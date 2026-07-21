import Foundation
import Metal
import simd

/// GPU runtime for volumetric (3D) NCAs: owns the voxel state buffers and the
/// step/life/raymarch pipelines. The 3D sibling of NCASimulation — kept separate
/// because dimensionality changes dispatch shape, indexing, and rendering wholesale.
final class NCASimulation3D {
    let device: MTLDevice
    let grid: Int

    private let queue: MTLCommandQueue
    private let stepPipeline: MTLComputePipelineState
    private let lifePipeline: MTLComputePipelineState
    private let renderPipeline: MTLComputePipelineState

    private var cur: MTLBuffer
    private var tmp: MTLBuffer
    private var next: MTLBuffer
    private var weightsBuffer: MTLBuffer
    private var filmBuffer: MTLBuffer

    private var fireRate: Float
    private var stepCounter: UInt32 = 0
    private var pendingDamage: (x: Float, y: Float, z: Float, radius: Float)?
    private let seed: (x: Int, y: Int, z: Int)

    let condCount: Int
    let hiddenCount: Int
    /// FiLM latent dimension (NC3M); when > 0, steer via `zTarget` / `setZ`.
    let zdim: Int
    private var filmMatrix: [Float] = []
    private var zCurrent: [Float] = []
    var zTarget: [Float] = []
    var condProvider: ((UInt32) -> (Float, Float, Float, Float))?
    /// Current phase angle for cyclic creatures (radians; the explorer's live dot).
    var currentTheta: Float { Float(stepCounter) * LainBehavior.omega }
    /// Camera orbit angle (radians); the view layer animates this.
    var azimuth: Float = 0
    var elevation: Float = 0.35

    struct Uniforms3D {
        var grid: Int32
        var fireRate: Float
        var step: UInt32
        var damageActive: Int32
        var damageX: Float
        var damageY: Float
        var damageZ: Float
        var damageRadius: Float
        var cond0: Float
        var cond1: Float
        var cond2: Float
        var cond3: Float
        var azimuth: Float
        var elevation: Float
    }

    init?(device: MTLDevice, weights: NCAWeights, grid: Int = 32,
          seed: (x: Int, y: Int, z: Int)? = nil) {
        guard weights.spatialDims == 3 else { return nil }
        self.device = device
        self.grid = grid
        self.fireRate = weights.fireRate
        self.condCount = weights.cond
        self.hiddenCount = weights.hidden
        self.zdim = weights.zdim
        self.filmMatrix = weights.film
        self.zCurrent = [Float](repeating: 0.5, count: weights.zdim)
        self.zTarget = self.zCurrent
        self.seed = seed ?? (grid / 2, grid / 2, grid / 2)

        guard let queue = device.makeCommandQueue() else { return nil }
        self.queue = queue

        guard let library = try? device.makeLibrary(
                  source: nca3dMetalSource(cond: weights.cond, hidden: weights.hidden,
                                           useFilm: weights.zdim > 0),
                  options: nil),
              let stepFn = library.makeFunction(name: "nca3d_step"),
              let lifeFn = library.makeFunction(name: "nca3d_life"),
              let renderFn = library.makeFunction(name: "nca3d_render"),
              let stepPS = try? device.makeComputePipelineState(function: stepFn),
              let lifePS = try? device.makeComputePipelineState(function: lifeFn),
              let renderPS = try? device.makeComputePipelineState(function: renderFn)
        else { return nil }
        stepPipeline = stepPS
        lifePipeline = lifePS
        renderPipeline = renderPS

        let stateBytes = grid * grid * grid * NCAWeights.channels * MemoryLayout<Float>.size
        guard let a = device.makeBuffer(length: stateBytes, options: .storageModeShared),
              let b = device.makeBuffer(length: stateBytes, options: .storageModeShared),
              let c = device.makeBuffer(length: stateBytes, options: .storageModeShared),
              let w = device.makeBuffer(bytes: weights.flat,
                                        length: weights.flat.count * MemoryLayout<Float>.size,
                                        options: .storageModeShared),
              let f = device.makeBuffer(length: max(2 * weights.hidden, 2) * MemoryLayout<Float>.size,
                                        options: .storageModeShared)
        else { return nil }
        cur = a; tmp = b; next = c; weightsBuffer = w; filmBuffer = f
        refreshFilm()

        reseed()
    }

    @discardableResult
    func updateWeights(_ weights: NCAWeights) -> Bool {
        guard weights.spatialDims == 3, weights.cond == condCount,
              weights.hidden == hiddenCount, weights.zdim == zdim,
              let w = device.makeBuffer(bytes: weights.flat,
                                        length: weights.flat.count * MemoryLayout<Float>.size,
                                        options: .storageModeShared) else { return false }
        weightsBuffer = w
        fireRate = weights.fireRate
        filmMatrix = weights.film
        refreshFilm()
        return true
    }

    /// Jump the latent immediately (headless tests / hard resets).
    func setZ(_ z: [Float]) {
        guard z.count == zdim else { return }
        zCurrent = z
        zTarget = z
        refreshFilm()
    }

    /// gamma/beta = filmW·z + filmB on the CPU; gamma rows tanh-bounded (training parity).
    private func refreshFilm() {
        guard zdim > 0, filmMatrix.count == 2 * hiddenCount * zdim + 2 * hiddenCount else { return }
        let ptr = filmBuffer.contents().bindMemory(to: Float.self, capacity: 2 * hiddenCount)
        let biasBase = 2 * hiddenCount * zdim
        for row in 0..<(2 * hiddenCount) {
            var acc = filmMatrix[biasBase + row]
            let rowBase = row * zdim
            for j in 0..<zdim { acc += filmMatrix[rowBase + j] * zCurrent[j] }
            ptr[row] = row < hiddenCount ? tanhf(acc) : acc
        }
    }

    /// Load raw RGBA voxels (z,y,x order, 4 floats each) into the visible channels —
    /// lets the raymarcher preview authored target volumes before any training exists.
    func loadStateRGBA(_ rgba: [Float]) {
        let voxels = grid * grid * grid
        guard rgba.count == voxels * 4 else { return }
        let ptr = cur.contents().bindMemory(to: Float.self, capacity: voxels * NCAWeights.channels)
        for i in 0..<voxels {
            for c in 0..<4 { ptr[i * NCAWeights.channels + c] = rgba[i * 4 + c] }
            for c in 4..<NCAWeights.channels { ptr[i * NCAWeights.channels + c] = 0 }
        }
    }

    func reseed() {
        let count = grid * grid * grid * NCAWeights.channels
        let ptr = cur.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count { ptr[i] = 0 }
        let base = ((seed.z * grid + seed.y) * grid + seed.x) * NCAWeights.channels
        for c in 3..<NCAWeights.channels { ptr[base + c] = 1.0 }
    }

    func damage(atVoxelX x: Float, y: Float, z: Float, radius: Float) {
        pendingDamage = (x, y, z, radius)
    }

    /// CPU-side ray pick: march the view ray (mirroring the shader camera) until
    /// alpha > 0.1; returns the hit voxel for damage placement, or nil on miss.
    func pick(ndcX: Float, ndcY: Float) -> (x: Float, y: Float, z: Float)? {
        let G = Float(grid)
        let center = SIMD3<Float>(repeating: G * 0.5)
        let ca = cos(azimuth), sa = sin(azimuth)
        let ce = cos(elevation), se = sin(elevation)
        let eye = center + 2.4 * G * SIMD3<Float>(sa * ce, se, ca * ce)
        let fwd = simd_normalize(center - eye)
        let right = simd_normalize(simd_cross(SIMD3<Float>(0, 1, 0), fwd))
        let up = simd_cross(fwd, right)
        let dir = simd_normalize(fwd * 1.9 + right * ndcX * 1.6 + up * ndcY * 1.6)

        let ptr = cur.contents().bindMemory(to: Float.self,
                                            capacity: grid * grid * grid * NCAWeights.channels)
        var t: Float = 0.5 * G
        while t < 4.0 * G {
            let p = eye + dir * t
            let xi = Int(p.x), yi = Int(p.y), zi = Int(p.z)
            if xi >= 0, yi >= 0, zi >= 0, xi < grid, yi < grid, zi < grid {
                let alpha = ptr[((zi * grid + yi) * grid + xi) * NCAWeights.channels + 3]
                if alpha > 0.1 { return (p.x, p.y, p.z) }
            }
            t += 0.6
        }
        return nil
    }

    private func makeUniforms() -> Uniforms3D {
        let d = pendingDamage
        let c = condProvider?(stepCounter) ?? (0, 0, 0, 0)
        return Uniforms3D(grid: Int32(grid), fireRate: fireRate, step: stepCounter,
                          damageActive: d == nil ? 0 : 1,
                          damageX: d?.x ?? 0, damageY: d?.y ?? 0, damageZ: d?.z ?? 0,
                          damageRadius: d?.radius ?? 0,
                          cond0: c.0, cond1: c.1, cond2: c.2, cond3: c.3,
                          azimuth: azimuth, elevation: elevation)
    }

    private func dispatch3D(_ enc: MTLComputeCommandEncoder, _ ps: MTLComputePipelineState) {
        enc.setComputePipelineState(ps)
        let tg = MTLSize(width: 4, height: 4, depth: 4)
        let n = (grid + 3) / 4
        enc.dispatchThreadgroups(MTLSize(width: n, height: n, depth: n),
                                 threadsPerThreadgroup: tg)
    }

    func step(count: Int, renderInto texture: MTLTexture? = nil) {
        guard let cmd = queue.makeCommandBuffer() else { return }

        if zdim > 0, zCurrent != zTarget {
            for j in 0..<zdim {
                zCurrent[j] += (zTarget[j] - zCurrent[j]) * 0.04
                if abs(zTarget[j] - zCurrent[j]) < 0.002 { zCurrent[j] = zTarget[j] }
            }
            refreshFilm()
        }

        for _ in 0..<count {
            var uniforms = makeUniforms()
            pendingDamage = nil
            stepCounter &+= 1

            guard let enc = cmd.makeComputeCommandEncoder() else { return }
            enc.setBuffer(cur, offset: 0, index: 0)
            enc.setBuffer(tmp, offset: 0, index: 1)
            enc.setBuffer(weightsBuffer, offset: 0, index: 2)
            enc.setBytes(&uniforms, length: MemoryLayout<Uniforms3D>.stride, index: 3)
            enc.setBuffer(filmBuffer, offset: 0, index: 4)
            dispatch3D(enc, stepPipeline)
            enc.endEncoding()

            guard let enc2 = cmd.makeComputeCommandEncoder() else { return }
            enc2.setBuffer(cur, offset: 0, index: 0)
            enc2.setBuffer(tmp, offset: 0, index: 1)
            enc2.setBuffer(next, offset: 0, index: 2)
            enc2.setBytes(&uniforms, length: MemoryLayout<Uniforms3D>.stride, index: 3)
            dispatch3D(enc2, lifePipeline)
            enc2.endEncoding()

            swap(&cur, &next)
        }

        if let texture {
            var uniforms = makeUniforms()
            guard let enc = cmd.makeComputeCommandEncoder() else { return }
            enc.setComputePipelineState(renderPipeline)
            enc.setBuffer(cur, offset: 0, index: 0)
            enc.setBytes(&uniforms, length: MemoryLayout<Uniforms3D>.stride, index: 1)
            enc.setTexture(texture, index: 0)
            let tg = MTLSize(width: 8, height: 8, depth: 1)
            enc.dispatchThreadgroups(
                MTLSize(width: (texture.width + 7) / 8, height: (texture.height + 7) / 8, depth: 1),
                threadsPerThreadgroup: tg)
            enc.endEncoding()
        }

        cmd.commit()
        if texture == nil { cmd.waitUntilCompleted() }
    }

    /// Raymarch into an offscreen texture and return RGBA8 bytes (headless tests).
    func renderOffscreen(size: Int = 256) -> [UInt8]? {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: size, height: size, mipmapped: false)
        desc.usage = [.shaderWrite, .shaderRead]
        guard let tex = device.makeTexture(descriptor: desc) else { return nil }
        step(count: 0, renderInto: tex)
        queue.makeCommandBuffer().map { $0.commit(); $0.waitUntilCompleted() }
        var bytes = [UInt8](repeating: 0, count: size * size * 4)
        tex.getBytes(&bytes, bytesPerRow: size * 4,
                     from: MTLRegionMake2D(0, 0, size, size), mipmapLevel: 0)
        return bytes
    }
}
