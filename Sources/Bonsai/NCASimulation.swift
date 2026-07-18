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

    // Three-buffer rotation: step reads `cur` -> writes `tmp`; life mask reads
    // `cur` (pre) + `tmp` (post) -> writes `next`; then cur and next swap.
    private var cur: MTLBuffer
    private var tmp: MTLBuffer
    private var next: MTLBuffer
    private var weightsBuffer: MTLBuffer

    private var fireRate: Float
    private var stepCounter: UInt32 = 0
    private var pendingDamage: (x: Float, y: Float, radius: Float)?

    /// Number of conditioning channels the compiled shader expects (0 = static creature).
    let condCount: Int
    /// Called once per automaton step to supply conditioning values (phase, behavior...).
    /// Unused entries are ignored by the shader.
    var condProvider: ((UInt32) -> (Float, Float, Float, Float))?
    /// Render style: 0 plain, 1 CRT scanlines (Lain).
    var renderStyle: Int32 = 0

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
    }

    init?(device: MTLDevice, weights: NCAWeights, gridWidth: Int = 64, gridHeight: Int = 64) {
        self.device = device
        self.gridWidth = gridWidth
        self.gridHeight = gridHeight
        self.fireRate = weights.fireRate
        self.condCount = weights.cond

        guard let queue = device.makeCommandQueue() else { return nil }
        self.queue = queue

        guard let library = try? device.makeLibrary(source: ncaMetalSource(cond: weights.cond), options: nil),
              let stepFn = library.makeFunction(name: "nca_step"),
              let lifeFn = library.makeFunction(name: "nca_life"),
              let renderFn = library.makeFunction(name: "nca_render"),
              let stepPS = try? device.makeComputePipelineState(function: stepFn),
              let lifePS = try? device.makeComputePipelineState(function: lifeFn),
              let renderPS = try? device.makeComputePipelineState(function: renderFn)
        else { return nil }
        stepPipeline = stepPS
        lifePipeline = lifePS
        renderPipeline = renderPS

        let stateBytes = gridWidth * gridHeight * NCAWeights.channels * MemoryLayout<Float>.size
        guard let a = device.makeBuffer(length: stateBytes, options: .storageModeShared),
              let b = device.makeBuffer(length: stateBytes, options: .storageModeShared),
              let c = device.makeBuffer(length: stateBytes, options: .storageModeShared),
              let w = device.makeBuffer(bytes: weights.flat,
                                        length: weights.flat.count * MemoryLayout<Float>.size,
                                        options: .storageModeShared)
        else { return nil }
        cur = a; tmp = b; next = c; weightsBuffer = w

        reseed()
    }

    /// Swap in new weights (hot-reload while training is still running).
    /// Returns false if the weights' conditioning shape doesn't match the compiled
    /// shader — the caller must rebuild the simulation instead.
    @discardableResult
    func updateWeights(_ weights: NCAWeights) -> Bool {
        guard weights.cond == condCount,
              let w = device.makeBuffer(bytes: weights.flat,
                                        length: weights.flat.count * MemoryLayout<Float>.size,
                                        options: .storageModeShared) else { return false }
        weightsBuffer = w
        fireRate = weights.fireRate
        return true
    }

    /// Clear the grid and plant a single seed cell at the given grid position (default: center).
    func reseed(atX x: Int? = nil, y: Int? = nil) {
        let sx = x ?? gridWidth / 2
        let sy = y ?? gridHeight / 2
        let ptr = cur.contents().bindMemory(to: Float.self,
                                            capacity: gridWidth * gridHeight * NCAWeights.channels)
        for i in 0..<(gridWidth * gridHeight * NCAWeights.channels) { ptr[i] = 0 }
        let base = (sy * gridWidth + sx) * NCAWeights.channels
        for c in 3..<NCAWeights.channels { ptr[base + c] = 1.0 }
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
                        style: renderStyle)
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

        for _ in 0..<count {
            var uniforms = makeUniforms()
            pendingDamage = nil
            stepCounter &+= 1

            guard let enc = cmd.makeComputeCommandEncoder() else { return }
            enc.setBuffer(cur, offset: 0, index: 0)
            enc.setBuffer(tmp, offset: 0, index: 1)
            enc.setBuffer(weightsBuffer, offset: 0, index: 2)
            enc.setBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 3)
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
        let count = gridWidth * gridHeight * NCAWeights.channels
        let ptr = cur.contents().bindMemory(to: Float.self, capacity: count)
        var out = [Float](repeating: 0, count: gridWidth * gridHeight * 4)
        for i in 0..<(gridWidth * gridHeight) {
            for c in 0..<4 { out[i * 4 + c] = ptr[i * NCAWeights.channels + c] }
        }
        return out
    }
}
