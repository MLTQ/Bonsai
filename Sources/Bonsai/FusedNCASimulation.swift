import Foundation
import Metal

/// Multi-pass Metal runtime for the hard-routed fused 2D NCA.
final class FusedNCASimulation {
    let device: MTLDevice
    let gridWidth: Int
    let gridHeight: Int

    private let queue: MTLCommandQueue
    private let posePipeline: MTLComputePipelineState
    private let fieldsPipeline: MTLComputePipelineState
    private let predictPipeline: MTLComputePipelineState
    private let correctPipeline: MTLComputePipelineState
    private let lifePipeline: MTLComputePipelineState
    private let renderPipeline: MTLComputePipelineState

    private var cur: MTLBuffer
    private var tmp: MTLBuffer
    private var next: MTLBuffer
    private var transported: MTLBuffer
    private var predicted: MTLBuffer
    private var flows: MTLBuffer
    private var selectedSlots: MTLBuffer
    private var reaction: MTLBuffer
    private var weightsBuffer: MTLBuffer
    private var shape: FusedNCAWeights
    private var resetState: [Float]?

    private var stepCounter: UInt32 = 0
    private var cycleStep = 0
    private var currentExpert = 0
    private var pendingDamage: (x: Float, y: Float, radius: Float)?

    var crispEdges = true
    var flipX = false

    struct Uniforms {
        var width: Int32
        var height: Int32
        var fireRate: Float
        var maxFlow: Float
        var step: UInt32
        var expert: Int32
        var transition: Int32
        var progress: Float
        var damageActive: Int32
        var damageX: Float
        var damageY: Float
        var damageRadius: Float
        var crisp: Int32
        var flipX: Int32
    }

    init?(device: MTLDevice, weights: FusedNCAWeights) {
        self.device = device
        self.gridWidth = weights.grid
        self.gridHeight = weights.grid
        self.shape = weights
        guard let queue = device.makeCommandQueue() else { return nil }
        self.queue = queue
        guard let library = try? device.makeLibrary(
                  source: fusedNCAMetalSource(weights), options: nil),
              let poseFunction = library.makeFunction(name: "fused_pose_step"),
              let fieldsFunction = library.makeFunction(name: "fused_edge_fields"),
              let predictFunction = library.makeFunction(name: "fused_edge_predict"),
              let correctFunction = library.makeFunction(name: "fused_edge_correct"),
              let lifeFunction = library.makeFunction(name: "fused_life"),
              let renderFunction = library.makeFunction(name: "fused_render"),
              let posePipeline = try? device.makeComputePipelineState(function: poseFunction),
              let fieldsPipeline = try? device.makeComputePipelineState(function: fieldsFunction),
              let predictPipeline = try? device.makeComputePipelineState(function: predictFunction),
              let correctPipeline = try? device.makeComputePipelineState(function: correctFunction),
              let lifePipeline = try? device.makeComputePipelineState(function: lifeFunction),
              let renderPipeline = try? device.makeComputePipelineState(function: renderFunction)
        else { return nil }
        self.posePipeline = posePipeline
        self.fieldsPipeline = fieldsPipeline
        self.predictPipeline = predictPipeline
        self.correctPipeline = correctPipeline
        self.lifePipeline = lifePipeline
        self.renderPipeline = renderPipeline

        let cells = weights.grid * weights.grid
        let stateBytes = cells * weights.channels * MemoryLayout<Float>.size
        guard let cur = device.makeBuffer(length: stateBytes, options: .storageModeShared),
              let tmp = device.makeBuffer(length: stateBytes, options: .storageModeShared),
              let next = device.makeBuffer(length: stateBytes, options: .storageModeShared),
              let transported = device.makeBuffer(length: stateBytes, options: .storageModeShared),
              let predicted = device.makeBuffer(
                length: stateBytes * weights.slots, options: .storageModePrivate),
              let flows = device.makeBuffer(
                length: cells * weights.slots * 2 * MemoryLayout<Float>.size,
                options: .storageModePrivate),
              let selectedSlots = device.makeBuffer(
                length: cells * MemoryLayout<UInt32>.size, options: .storageModePrivate),
              let reaction = device.makeBuffer(length: stateBytes, options: .storageModePrivate),
              let weightsBuffer = device.makeBuffer(
                bytes: weights.flat,
                length: weights.flat.count * MemoryLayout<Float>.size,
                options: .storageModeShared)
        else { return nil }
        self.cur = cur
        self.tmp = tmp
        self.next = next
        self.transported = transported
        self.predicted = predicted
        self.flows = flows
        self.selectedSlots = selectedSlots
        self.reaction = reaction
        self.weightsBuffer = weightsBuffer
        reseed()
    }

    @discardableResult
    func updateWeights(_ weights: FusedNCAWeights) -> Bool {
        guard shape.isShapeCompatible(with: weights),
              let replacement = device.makeBuffer(
                bytes: weights.flat,
                length: weights.flat.count * MemoryLayout<Float>.size,
                options: .storageModeShared)
        else { return false }
        shape = weights
        weightsBuffer = replacement
        return true
    }

    @discardableResult
    func loadState(from path: String) -> Bool {
        guard let data = FileManager.default.contents(atPath: path), data.count >= 16,
              String(decoding: data.prefix(4), as: UTF8.self) == "NCS1"
        else { return false }
        func i32(_ offset: Int) -> Int {
            Int(data.withUnsafeBytes {
                $0.loadUnaligned(fromByteOffset: offset, as: Int32.self)
            })
        }
        let width = i32(4), height = i32(8), channels = i32(12)
        let count = gridWidth * gridHeight * shape.channels
        guard width == gridWidth, height == gridHeight, channels == shape.channels,
              data.count == 16 + count * MemoryLayout<Float>.size
        else { return false }
        var values = [Float](repeating: 0, count: count)
        data.withUnsafeBytes { raw in
            values.withUnsafeMutableBytes { destination in
                destination.baseAddress!.copyMemory(
                    from: raw.baseAddress!.advanced(by: 16), byteCount: count * 4)
            }
        }
        resetState = values
        reseed()
        return true
    }

    func reseed() {
        let count = gridWidth * gridHeight * shape.channels
        let destination = cur.contents().bindMemory(to: Float.self, capacity: count)
        if let resetState, resetState.count == count {
            resetState.withUnsafeBufferPointer { source in
                destination.update(from: source.baseAddress!, count: count)
            }
        } else {
            for index in 0..<count { destination[index] = 0 }
            let base = ((gridHeight / 2) * gridWidth + gridWidth / 2) * shape.channels
            for channel in 3..<shape.channels { destination[base + channel] = 1 }
        }
        stepCounter = 0
        cycleStep = 0
        currentExpert = 0
        pendingDamage = nil
    }

    func damage(atGridX x: Float, gridY y: Float, radius: Float) {
        pendingDamage = (x, y, radius)
    }

    private func uniforms() -> Uniforms {
        let transition = cycleStep < shape.transitionSteps
        let expert = transition ? currentExpert : (currentExpert + 1) % shape.experts
        let linear = transition
            ? Float(cycleStep + 1) / Float(shape.transitionSteps) : 0
        let progress = transition ? linear * linear * (3 - 2 * linear) : 0
        let damage = pendingDamage
        return Uniforms(
            width: Int32(gridWidth), height: Int32(gridHeight),
            fireRate: shape.fireRate, maxFlow: shape.maxFlow, step: stepCounter,
            expert: Int32(expert), transition: transition ? 1 : 0, progress: progress,
            damageActive: damage == nil ? 0 : 1,
            damageX: damage?.x ?? 0, damageY: damage?.y ?? 0,
            damageRadius: damage?.radius ?? 0,
            crisp: crispEdges ? 1 : 0, flipX: flipX ? 1 : 0)
    }

    private func dispatch2D(_ encoder: MTLComputeCommandEncoder,
                            _ pipeline: MTLComputePipelineState,
                            width: Int, height: Int) {
        encoder.setComputePipelineState(pipeline)
        let threads = MTLSize(width: 8, height: 8, depth: 1)
        let groups = MTLSize(width: (width + 7) / 8, height: (height + 7) / 8, depth: 1)
        encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
    }

    private func dispatchSlots(_ encoder: MTLComputeCommandEncoder) {
        encoder.setComputePipelineState(predictPipeline)
        let threads = MTLSize(width: 8, height: 8, depth: 1)
        let groups = MTLSize(
            width: (gridWidth + 7) / 8, height: (gridHeight + 7) / 8,
            depth: shape.slots)
        encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
    }

    func step(count: Int, renderInto texture: MTLTexture? = nil) {
        guard let command = queue.makeCommandBuffer() else { return }
        for _ in 0..<count {
            var uniform = uniforms()
            let transition = cycleStep < shape.transitionSteps
            pendingDamage = nil

            if transition {
                guard let fieldsEncoder = command.makeComputeCommandEncoder() else { return }
                fieldsEncoder.setBuffer(cur, offset: 0, index: 0)
                fieldsEncoder.setBuffer(flows, offset: 0, index: 1)
                fieldsEncoder.setBuffer(selectedSlots, offset: 0, index: 2)
                fieldsEncoder.setBuffer(reaction, offset: 0, index: 3)
                fieldsEncoder.setBuffer(weightsBuffer, offset: 0, index: 4)
                fieldsEncoder.setBytes(&uniform, length: MemoryLayout<Uniforms>.stride, index: 5)
                dispatch2D(fieldsEncoder, fieldsPipeline, width: gridWidth, height: gridHeight)
                fieldsEncoder.endEncoding()

                guard let predictEncoder = command.makeComputeCommandEncoder() else { return }
                predictEncoder.setBuffer(cur, offset: 0, index: 0)
                predictEncoder.setBuffer(flows, offset: 0, index: 1)
                predictEncoder.setBuffer(predicted, offset: 0, index: 2)
                predictEncoder.setBytes(&uniform, length: MemoryLayout<Uniforms>.stride, index: 3)
                dispatchSlots(predictEncoder)
                predictEncoder.endEncoding()

                guard let correctEncoder = command.makeComputeCommandEncoder() else { return }
                correctEncoder.setBuffer(cur, offset: 0, index: 0)
                correctEncoder.setBuffer(flows, offset: 0, index: 1)
                correctEncoder.setBuffer(selectedSlots, offset: 0, index: 2)
                correctEncoder.setBuffer(predicted, offset: 0, index: 3)
                correctEncoder.setBuffer(reaction, offset: 0, index: 4)
                correctEncoder.setBuffer(transported, offset: 0, index: 5)
                correctEncoder.setBuffer(tmp, offset: 0, index: 6)
                correctEncoder.setBytes(&uniform, length: MemoryLayout<Uniforms>.stride, index: 7)
                dispatch2D(correctEncoder, correctPipeline, width: gridWidth, height: gridHeight)
                correctEncoder.endEncoding()
            } else {
                guard let poseEncoder = command.makeComputeCommandEncoder() else { return }
                poseEncoder.setBuffer(cur, offset: 0, index: 0)
                poseEncoder.setBuffer(tmp, offset: 0, index: 1)
                poseEncoder.setBuffer(weightsBuffer, offset: 0, index: 2)
                poseEncoder.setBytes(&uniform, length: MemoryLayout<Uniforms>.stride, index: 3)
                dispatch2D(poseEncoder, posePipeline, width: gridWidth, height: gridHeight)
                poseEncoder.endEncoding()
            }

            guard let lifeEncoder = command.makeComputeCommandEncoder() else { return }
            lifeEncoder.setBuffer(transition ? transported : cur, offset: 0, index: 0)
            lifeEncoder.setBuffer(tmp, offset: 0, index: 1)
            lifeEncoder.setBuffer(next, offset: 0, index: 2)
            lifeEncoder.setBytes(&uniform, length: MemoryLayout<Uniforms>.stride, index: 3)
            dispatch2D(lifeEncoder, lifePipeline, width: gridWidth, height: gridHeight)
            lifeEncoder.endEncoding()
            swap(&cur, &next)

            stepCounter &+= 1
            cycleStep += 1
            if cycleStep >= shape.transitionSteps + shape.handoffSteps {
                cycleStep = 0
                currentExpert = (currentExpert + 1) % shape.experts
            }
        }

        if let texture {
            var uniform = uniforms()
            guard let encoder = command.makeComputeCommandEncoder() else { return }
            encoder.setBuffer(cur, offset: 0, index: 0)
            encoder.setBytes(&uniform, length: MemoryLayout<Uniforms>.stride, index: 1)
            encoder.setTexture(texture, index: 0)
            dispatch2D(encoder, renderPipeline, width: texture.width, height: texture.height)
            encoder.endEncoding()
        }
        command.commit()
        if texture == nil { command.waitUntilCompleted() }
    }

    func readRGBA() -> [Float] {
        queue.makeCommandBuffer().map { $0.commit(); $0.waitUntilCompleted() }
        let count = gridWidth * gridHeight * shape.channels
        let source = cur.contents().bindMemory(to: Float.self, capacity: count)
        var rgba = [Float](repeating: 0, count: gridWidth * gridHeight * 4)
        for cell in 0..<(gridWidth * gridHeight) {
            for channel in 0..<4 {
                rgba[cell * 4 + channel] = source[cell * shape.channels + channel]
            }
        }
        return rgba
    }
}
