import Foundation

/// Parsed contents of a `.nca` weights file.
/// NCA1 (train_nca.py): magic, int32 channels, int32 hidden, float32 fireRate, then flat
/// float32 w1[hidden][channels*3], b1[hidden], w2[channels][hidden], b2[channels].
/// NCA2 (train_cyclic.py): same but an int32 cond-channel count follows hidden, and
/// w1 rows widen to channels*3 + cond (conditioning values appended after perception).
struct NCAWeights {
    static let channels = 16
    static let hidden = 128

    let fireRate: Float
    /// Number of conditioning channels (0 for NCA1 static creatures).
    let cond: Int
    /// w1, b1, w2, b2 concatenated in file order — uploaded to the GPU as one buffer.
    let flat: [Float]

    enum LoadError: Error, CustomStringConvertible {
        case unreadable(String)
        case badMagic
        case shapeMismatch(ch: Int32, hidden: Int32)
        case truncated

        var description: String {
            switch self {
            case .unreadable(let p): return "cannot read weights file: \(p)"
            case .badMagic: return "not an NCA1 weights file"
            case .shapeMismatch(let c, let h): return "unsupported shape ch=\(c) hidden=\(h)"
            case .truncated: return "weights file is truncated"
            }
        }
    }

    static func load(from path: String) throws -> NCAWeights {
        guard let data = FileManager.default.contents(atPath: path) else {
            throw LoadError.unreadable(path)
        }
        guard data.count > 16 else { throw LoadError.badMagic }
        let magic = data.prefix(4)
        let isV2 = magic == Data("NCA2".utf8)
        guard isV2 || magic == Data("NCA1".utf8) else { throw LoadError.badMagic }

        let ch = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 4, as: Int32.self) }
        let hid = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 8, as: Int32.self) }
        guard ch == channels, hid == hidden else {
            throw LoadError.shapeMismatch(ch: ch, hidden: hid)
        }
        var offset = 12
        var cond = 0
        if isV2 {
            cond = Int(data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 12, as: Int32.self) })
            guard cond >= 0, cond <= 4 else { throw LoadError.shapeMismatch(ch: ch, hidden: hid) }
            offset = 16
        }
        let fireRate = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: Float.self) }
        offset += 4

        let w1In = channels * 3 + cond
        let floatCount = hidden * w1In + hidden + channels * hidden + channels
        guard data.count >= offset + floatCount * 4 else { throw LoadError.truncated }

        var flat = [Float](repeating: 0, count: floatCount)
        data.withUnsafeBytes { raw in
            let src = raw.baseAddress!.advanced(by: offset)
            flat.withUnsafeMutableBytes { dst in
                dst.baseAddress!.copyMemory(from: src, byteCount: floatCount * 4)
            }
        }
        return NCAWeights(fireRate: fireRate, cond: cond, flat: flat)
    }

    /// Directory holding .nca files. Search order: $BONSAI_WEIGHTS_DIR, ./weights,
    /// <repo>/weights relative to the built executable, bundled Resources.
    static func weightsDir() -> String? {
        var candidates: [String] = []
        if let env = ProcessInfo.processInfo.environment["BONSAI_WEIGHTS_DIR"] {
            candidates.append(env)
        }
        candidates.append(FileManager.default.currentDirectoryPath + "/weights")
        let exe = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent()
        candidates.append(exe.appendingPathComponent("../../../weights").standardized.path)
        candidates.append(exe.appendingPathComponent("../Resources").standardized.path)
        return candidates.first {
            var isDir: ObjCBool = false
            return FileManager.default.fileExists(atPath: $0, isDirectory: &isDir) && isDir.boolValue
        }
    }

    /// Legacy convenience: the bonsai weights file.
    static func defaultPath() -> String? {
        if let env = ProcessInfo.processInfo.environment["BONSAI_WEIGHTS"],
           FileManager.default.fileExists(atPath: env) {
            return env
        }
        guard let dir = weightsDir() else { return nil }
        let p = dir + "/bonsai.nca"
        return FileManager.default.fileExists(atPath: p) ? p : nil
    }
}
