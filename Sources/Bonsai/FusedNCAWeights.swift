import Foundation

/// Flat deployment form of the hard-routed four-pose/four-edge 2D NCA.
struct FusedNCAWeights {
    static let headerBytes = 48
    static let magic = "FX2D"

    let grid: Int
    let channels: Int
    let experts: Int
    let slots: Int
    let expertHidden: Int
    let flowHidden: Int
    let positionFrequencies: Int
    let maxFlow: Float
    let fireRate: Float
    let transitionSteps: Int
    let handoffSteps: Int
    let flat: [Float]

    var coordinateChannels: Int { 2 + 4 * positionFrequencies }
    var ruleInputs: Int { channels * 3 + coordinateChannels + 1 }

    enum LoadError: Error, CustomStringConvertible {
        case unreadable(String)
        case badMagic
        case truncated
        case shapeMismatch(String)

        var description: String {
            switch self {
            case .unreadable(let path): return "cannot read fused weights: \(path)"
            case .badMagic: return "not an FX2D fused weights file"
            case .truncated: return "fused weights file is truncated"
            case .shapeMismatch(let message): return "unsupported fused shape: \(message)"
            }
        }
    }

    static func load(from path: String) throws -> FusedNCAWeights {
        guard let data = FileManager.default.contents(atPath: path) else {
            throw LoadError.unreadable(path)
        }
        guard data.count >= headerBytes else { throw LoadError.truncated }
        guard String(decoding: data.prefix(4), as: UTF8.self) == magic else {
            throw LoadError.badMagic
        }
        func i32(_ offset: Int) -> Int {
            Int(data.withUnsafeBytes {
                $0.loadUnaligned(fromByteOffset: offset, as: Int32.self)
            })
        }
        func f32(_ offset: Int) -> Float {
            data.withUnsafeBytes {
                $0.loadUnaligned(fromByteOffset: offset, as: Float.self)
            }
        }
        let grid = i32(4), channels = i32(8), experts = i32(12), slots = i32(16)
        let expertHidden = i32(20), flowHidden = i32(24), frequencies = i32(28)
        let maxFlow = f32(32), fireRate = f32(36)
        let transitionSteps = i32(40), handoffSteps = i32(44)
        guard grid > 1, grid <= 512, channels == 16, experts == 4,
              slots > 0, slots <= 8, expertHidden > 0, expertHidden <= 1024,
              flowHidden > 0, flowHidden <= 512, frequencies >= 0, frequencies <= 8,
              maxFlow > 0, maxFlow <= 8, fireRate >= 0, fireRate <= 1,
              transitionSteps > 0, handoffSteps > 0
        else {
            throw LoadError.shapeMismatch(
                "grid=\(grid) ch=\(channels) experts=\(experts) slots=\(slots) " +
                "width=\(expertHidden)/\(flowHidden) fourier=\(frequencies)")
        }
        let coordinateChannels = 2 + 4 * frequencies
        let inputs = channels * 3 + coordinateChannels + 1
        func affineCount(hidden: Int, outputs: Int) -> Int {
            experts * (hidden * inputs + hidden + outputs * hidden + outputs)
        }
        let floatCount = affineCount(hidden: expertHidden, outputs: channels)
            + affineCount(hidden: flowHidden, outputs: slots * 2)
            + affineCount(hidden: flowHidden, outputs: slots)
            + affineCount(hidden: expertHidden, outputs: channels)
        guard data.count == headerBytes + floatCount * MemoryLayout<Float>.size else {
            throw LoadError.shapeMismatch(
                "payload bytes=\(data.count - headerBytes), expected=\(floatCount * 4)")
        }
        var flat = [Float](repeating: 0, count: floatCount)
        data.withUnsafeBytes { raw in
            flat.withUnsafeMutableBytes { destination in
                destination.baseAddress!.copyMemory(
                    from: raw.baseAddress!.advanced(by: headerBytes),
                    byteCount: floatCount * MemoryLayout<Float>.size)
            }
        }
        return FusedNCAWeights(
            grid: grid, channels: channels, experts: experts, slots: slots,
            expertHidden: expertHidden, flowHidden: flowHidden,
            positionFrequencies: frequencies, maxFlow: maxFlow, fireRate: fireRate,
            transitionSteps: transitionSteps, handoffSteps: handoffSteps, flat: flat)
    }

    func isShapeCompatible(with other: FusedNCAWeights) -> Bool {
        grid == other.grid && channels == other.channels && experts == other.experts
            && slots == other.slots && expertHidden == other.expertHidden
            && flowHidden == other.flowHidden
            && positionFrequencies == other.positionFrequencies
            && transitionSteps == other.transitionSteps
            && handoffSteps == other.handoffSteps
    }
}
