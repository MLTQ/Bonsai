import Foundation
import Metal

/// Runs the NCA on the GPU: owns the state buffers, weight buffer, and compute pipelines.
/// One instance per pet. All methods must be called from a single thread (the main thread).
final class NCASimulation {
    let device: MTLDevice
    let gridWidth: Int
    let gridHeight: Int

    private let queue: MTLCommandQueue
    private let stepPipeline: MTLComputePipelineState
    private let lifePipeline: MTLComputePipelineState
    private let renderPipeline: MTLComputePipelineState
    private let poolPipeline: MTLComputePipelineState

    // Three-buffer rotation: step reads `cur` -> writes `tmp`; life mask reads
    // `cur` (pre) + `tmp` (post) -> writes `next`; then cur and next swap.
    private var cur: MTLBuffer
    private var tmp: MTLBuffer
    private var next: MTLBuffer
    private var weightsBuffer: MTLBuffer
    private var filmBuffer: MTLBuffer
    private var poolBuffer: MTLBuffer

    private var fireRate: Float
    private var momentumDecay: Float
    private var stepCounter: UInt32 = 0
    private var pendingDamage: (x: Float, y: Float, radius: Float)?

    /// Number of conditioning channels the compiled shader expects (0 = static creature).
    let condCount: Int
    /// Update-rule width the shader was compiled with.
    let hiddenCount: Int
    /// FiLM latent dimension (0 = no FiLM). When > 0, set `zTarget` to steer mood.
    let zdim: Int
    /// Pooled feedback channels (NCAP). When > 0, nca_pool runs before every step.
    let npool: Int
    /// Physical state width and its position/output prefix. They differ only for
    /// NCA4, whose remaining channels are matched velocities.
    let stateChannelCount: Int
    let positionChannelCount: Int
    private var filmMatrix: [Float] = []   // (2*hidden, zdim) row-major + bias (2*hidden)
    private var zCurrent: [Float] = []
    /// Desired latent point; the simulation eases toward it (~2%/step) so moods morph.
    var zTarget: [Float] = []
    /// Called once per automaton step to supply conditioning values (phase, behavior...).
    /// Unused entries are ignored by the shader.
    var condProvider: ((UInt32) -> (Float, Float, Float, Float))?
    /// Render style: 0 plain, 1 CRT scanlines (Lain).
    var renderStyle: Int32 = 0
    /// Mirror the render horizontally (creature facing). State is untouched.
    var flipX: Bool = false
    /// Sharp silhouette at display time (bilinear alpha + smoothstep remap).
    /// Display-only: verification tools read the raw state and are unaffected.
    var crispEdges: Bool = true

    struct Uniforms {
        var width: Int32
        var height: Int32
        var fireRate: Float
        var step: UInt32
        var damageActive: Int32
        var damageX: Float
        var damageY: Float
        var damageRadius: Float
        var cond0: Float
        var cond1: Float
        var cond2: Float
        var cond3: Float
        var style: Int32
        var flipX: Int32
        var crisp: Int32
        var momentumDecay: Float
    }

    init?(device: MTLDevice, weights: NCAWeights, gridWidth: Int = 64, gridHeight: Int = 64) {
        self.device = device
        self.gridWidth = gridWidth
        self.gridHeight = gridHeight
        self.fireRate = weights.fireRate
        self.momentumDecay = weights.momentumDecay
        self.condCount = weights.cond
        self.hiddenCount = weights.hidden
        self.zdim = weights.zdim
        self.npool = weights.npool
        self.stateChannelCount = weights.stateChannels
        self.positionChannelCount = weights.positionChannels
        self.filmMatrix = weights.film
        self.zCurrent = [Float](repeating: 0.5, count: weights.zdim)
        self.zTarget = self.zCurrent

        guard let queue = device.makeCommandQueue() else { return nil }
        self.queue = queue

        guard let library = try? device.makeLibrary(
                  source: ncaMetalSource(cond: weights.cond, hidden: weights.hidden,
                                         useFilm: weights.zdim > 0, npool: weights.npool,
                                         stateChannels: weights.stateChannels,
                                         positionChannels: weights.positionChannels),
                  options: nil),
              let stepFn = library.makeFunction(name: "nca_step"),
              let lifeFn = library.makeFunction(name: "nca_life"),
              let renderFn = library.makeFunction(name: "nca_render"),
              let poolFn = library.makeFunction(name: "nca_pool"),
              let stepPS = try? device.makeComputePipelineState(function: stepFn),
              let lifePS = try? device.makeComputePipelineState(function: lifeFn),
              let renderPS = try? device.makeComputePipelineState(function: renderFn),
              let poolPS = try? device.makeComputePipelineState(function: poolFn)
        else { return nil }
        stepPipeline = stepPS
        lifePipeline = lifePS
        renderPipeline = renderPS
        poolPipeline = poolPS

        let stateBytes = gridWidth * gridHeight * stateChannelCount * MemoryLayout<Float>.size
        guard let a = device.makeBuffer(length: stateBytes, options: .storageModeShared),
              let b = device.makeBuffer(length: stateBytes, options: .storageModeShared),
              let c = device.makeBuffer(length: stateBytes, options: .storageModeShared),
              let w = device.makeBuffer(bytes: weights.flat,
                                        length: weights.flat.count * MemoryLayout<Float>.size,
                                        options: .storageModeShared),
              let f = device.makeBuffer(length: max(2 * weights.hidden, 2) * MemoryLayout<Float>.size,
                                        options: .storageModeShared),
              let g = device.makeBuffer(length: max(weights.npool, 1) * MemoryLayout<Float>.size,
                                        options: .storageModeShared)
        else { return nil }
        cur = a; tmp = b; next = c; weightsBuffer = w; filmBuffer = f; poolBuffer = g
        refreshFilm()

        reseed()
    }

    /// Jump the latent immediately (no easing) — headless tests and hard resets.
    func setZ(_ z: [Float]) {
        guard z.count == zdim else { return }
        zCurrent = z
        zTarget = z
        refreshFilm()
    }

    /// gamma/beta = filmW·z + filmB, computed CPU-side (uniform across cells, tiny).
    private func refreshFilm() {
        guard zdim > 0, filmMatrix.count == 2 * hiddenCount * zdim + 2 * hiddenCount else { return }
        let ptr = filmBuffer.contents().bindMemory(to: Float.self, capacity: 2 * hiddenCount)
        let biasBase = 2 * hiddenCount * zdim
        for row in 0..<(2 * hiddenCount) {
            var acc = filmMatrix[biasBase + row]
            let rowBase = row * zdim
            for j in 0..<zdim { acc += filmMatrix[rowBase + j] * zCurrent[j] }
            // gamma rows (first half) are tanh-bounded, matching train_manifold.py
            ptr[row] = row < hiddenCount ? tanhf(acc) : acc
        }
    }

    /// Swap in new weights (hot-reload while training is still running).
    /// Returns false if the weights' conditioning shape doesn't match the compiled
    /// shader — the caller must rebuild the simulation instead.
    @discardableResult
    func updateWeights(_ weights: NCAWeights) -> Bool {
        guard weights.cond == condCount, weights.hidden == hiddenCount, weights.zdim == zdim,
              weights.npool == npool, weights.stateChannels == stateChannelCount,
              weights.positionChannels == positionChannelCount,
              let w = device.makeBuffer(bytes: weights.flat,
                                        length: weights.flat.count * MemoryLayout<Float>.size,
                                        options: .storageModeShared) else { return false }
        weightsBuffer = w
        fireRate = weights.fireRate
        momentumDecay = weights.momentumDecay
        filmMatrix = weights.film
        refreshFilm()
        return true
    }

    /// Clear the grid and plant a single seed cell at the given grid position (default: center).
    func reseed(atX x: Int? = nil, y: Int? = nil) {
        let sx = x ?? gridWidth / 2
        let sy = y ?? gridHeight / 2
        let ptr = cur.contents().bindMemory(to: Float.self,
                                            capacity: gridWidth * gridHeight * stateChannelCount)
        for i in 0..<(gridWidth * gridHeight * stateChannelCount) { ptr[i] = 0 }
        let base = (sy * gridWidth + sx) * stateChannelCount
        // Position alpha/hidden channels start alive. NCA4 velocities remain zero.
        for c in 3..<positionChannelCount { ptr[base + c] = 1.0 }
    }

    /// Queue a circular damage zone; applied during the next step (the NCA then regrows).
    func damage(atGridX x: Float, gridY y: Float, radius: Float) {
        pendingDamage = (x, y, radius)
    }

    private func makeUniforms() -> Uniforms {
        let d = pendingDamage
        let c = condProvider?(stepCounter) ?? (0, 0, 0, 0)
        return Uniforms(width: Int32(gridWidth), height: Int32(gridHeight),
                        fireRate: fireRate, step: stepCounter,
                        damageActive: d == nil ? 0 : 1,
                        damageX: d?.x ?? 0, damageY: d?.y ?? 0, damageRadius: d?.radius ?? 0,
                        cond0: c.0, cond1: c.1, cond2: c.2, cond3: c.3,
                        style: renderStyle, flipX: flipX ? 1 : 0,
                        crisp: crispEdges ? 1 : 0, momentumDecay: momentumDecay)
    }

    private func dispatch(_ encoder: MTLComputeCommandEncoder, _ pipeline: MTLComputePipelineState,
                          width: Int, height: Int) {
        encoder.setComputePipelineState(pipeline)
        let tg = MTLSize(width: 8, height: 8, depth: 1)
        let grid = MTLSize(width: (width + 7) / 8, height: (height + 7) / 8, depth: 1)
        encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
    }

    /// Advance the automaton `count` steps and (optionally) render into a drawable texture,
    /// all in one command buffer.
    func step(count: Int, renderInto texture: MTLTexture? = nil) {
        guard let cmd = queue.makeCommandBuffer() else { return }

        // Ease the latent toward its target once per frame-batch; FiLM params refresh
        // CPU-side (uniform across cells, a few thousand MACs).
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

            if npool > 0 {
                // The global variable must be current before every step: one
                // threadgroup strides the grid and reduces in shared memory.
                guard let encP = cmd.makeComputeCommandEncoder() else { return }
                encP.setComputePipelineState(poolPipeline)
                encP.setBuffer(cur, offset: 0, index: 0)
                encP.setBuffer(poolBuffer, offset: 0, index: 1)
                encP.setBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 2)
                encP.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encP.endEncoding()
            }

            guard let enc = cmd.makeComputeCommandEncoder() else { return }
            enc.setBuffer(cur, offset: 0, index: 0)
            enc.setBuffer(tmp, offset: 0, index: 1)
            enc.setBuffer(weightsBuffer, offset: 0, index: 2)
            enc.setBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 3)
            enc.setBuffer(filmBuffer, offset: 0, index: 4)
            enc.setBuffer(poolBuffer, offset: 0, index: 5)
            dispatch(enc, stepPipeline, width: gridWidth, height: gridHeight)
            enc.endEncoding()

            guard let enc2 = cmd.makeComputeCommandEncoder() else { return }
            enc2.setBuffer(cur, offset: 0, index: 0)
            enc2.setBuffer(tmp, offset: 0, index: 1)
            enc2.setBuffer(next, offset: 0, index: 2)
            enc2.setBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 3)
            dispatch(enc2, lifePipeline, width: gridWidth, height: gridHeight)
            enc2.endEncoding()

            swap(&cur, &next)
        }

        if let texture {
            var uniforms = makeUniforms()
            guard let enc = cmd.makeComputeCommandEncoder() else { return }
            enc.setBuffer(cur, offset: 0, index: 0)
            enc.setBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.setTexture(texture, index: 0)
            dispatch(enc, renderPipeline, width: texture.width, height: texture.height)
            enc.endEncoding()
        }

        cmd.commit()
        if texture == nil { cmd.waitUntilCompleted() }
    }

    /// Copy the current RGBA (first 4 channels) off the GPU — used by the headless render test.
    func readRGBA() -> [Float] {
        queue.makeCommandBuffer().map { $0.commit(); $0.waitUntilCompleted() }
        let count = gridWidth * gridHeight * stateChannelCount
        let ptr = cur.contents().bindMemory(to: Float.self, capacity: count)
        var out = [Float](repeating: 0, count: gridWidth * gridHeight * 4)
        for i in 0..<(gridWidth * gridHeight) {
            for c in 0..<4 { out[i * 4 + c] = ptr[i * stateChannelCount + c] }
        }
        return out
    }
}
