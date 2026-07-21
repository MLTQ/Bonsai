import Foundation

/// Parsed contents of a `.nca` weights file. Multiple formats, one loader:
/// NCA1 (train_nca.py):     magic, i32 ch, i32 hidden, f32 fire, w1[h][ch*3], b1, w2[ch][h], b2
/// NCA2 (train_cyclic.py):  + i32 cond after hidden; w1 rows widen to ch*3+cond
/// NCA3 (train_manifold.py): i32 ch, hidden, zdim; fire; w1[h][ch*3+2] (sin/cos phase),
///                           b1, w2, b2, then FiLM: filmW[2*hidden][zdim], filmB[2*hidden].
///                           Phase rides as 2 cond channels; z modulates via FiLM.
/// NCAP (pooled_nca.py):     + i32 npool after cond; w1 rows widen to ch*3+cond+npool.
///                           Breaks strict locality: the extra inputs are a spatial
///                           reduction over the whole grid, so the runtime must run
///                           nca_pool before every nca_step.
/// NCA4 (momentum_nca.py):   i32 stateCh, hidden, cond, positionCh; f32 fire, decay;
///                           w1[h][stateCh*3+cond], b1, w2[positionCh][h], b2.
///                           Requires stateCh == 2*positionCh; the second half stores
///                           velocities and the learned head emits forces.
struct NCAWeights {
    static let channels = 16

    /// Total channels stored per cell (32 for NCA4, 16 for legacy formats).
    let stateChannels: Int
    /// Channels advanced as positions. NCA4 has a matched velocity half; legacy
    /// formats set this equal to stateChannels and use residual updates.
    let positionChannels: Int
    let hidden: Int
    let fireRate: Float
    /// Velocity retention per step (NCA4 only; zero for residual formats).
    let momentumDecay: Float
    /// Conditioning channels appended to perception (NCA2: cond; NCA3: 2 = sin/cos).
    let cond: Int
    /// FiLM latent dimension (NCA3 only; 0 otherwise).
    let zdim: Int
    /// Globally-broadcast feedback channels (NCAP only; 0 otherwise). These are
    /// the alive-masked spatial mean of state channels [4, 4+npool), recomputed
    /// each step and appended to every cell's perception — see
    /// training/pooled_nca.py. Non-zero npool means the creature is not strictly
    /// local and needs the reduction pass before each step.
    let npool: Int
    /// 2 for planar creatures (3 perception kernels), 3 for volumetric (4 kernels:
    /// identity + Sobel x/y/z). Formats NC3D (static) and NC3C (cyclic) are 3D.
    let spatialDims: Int
    /// w1, b1, w2, b2 concatenated — uploaded to the GPU as one buffer.
    let flat: [Float]
    /// FiLM matrices, filmW row-major (2*hidden, zdim) then filmB (2*hidden). Empty unless NCA3.
    let film: [Float]

    enum LoadError: Error, CustomStringConvertible {
        case unreadable(String)
        case badMagic
        case shapeMismatch(String)
        case truncated

        var description: String {
            switch self {
            case .unreadable(let p): return "cannot read weights file: \(p)"
            case .badMagic: return "not a supported NCA weights file"
            case .shapeMismatch(let s): return "unsupported shape: \(s)"
            case .truncated: return "weights file is truncated"
            }
        }
    }

    static func load(from path: String) throws -> NCAWeights {
        guard let data = FileManager.default.contents(atPath: path) else {
            throw LoadError.unreadable(path)
        }
        guard data.count >= 4 else { throw LoadError.badMagic }
        let magic = String(decoding: data.prefix(4), as: UTF8.self)
        guard ["NCA1", "NCA2", "NCA3", "NCA4", "NCAP", "NC3D", "NC3C", "NC3M"].contains(magic) else {
            throw LoadError.badMagic
        }
        let headerBytes: Int
        switch magic {
        case "NCA4": headerBytes = 28
        case "NCAP": headerBytes = 24
        case "NCA2", "NCA3", "NC3C", "NC3M": headerBytes = 20
        default: headerBytes = 16
        }
        guard data.count >= headerBytes else { throw LoadError.truncated }
        let spatialDims = magic.hasPrefix("NC3") ? 3 : 2

        func i32(_ off: Int) -> Int {
            Int(data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: off, as: Int32.self) })
        }
        func f32(_ off: Int) -> Float {
            data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: off, as: Float.self) }
        }

        let ch = i32(4), hidden = i32(8)
        guard hidden > 0, hidden <= 1024 else {
            throw LoadError.shapeMismatch("hidden=\(hidden)")
        }
        if magic != "NCA4" && ch != channels {
            throw LoadError.shapeMismatch("ch=\(ch) hidden=\(hidden)")
        }
        var offset = 12
        var cond = 0, zdim = 0, npool = 0, positionChannels = ch
        var momentumDecay: Float = 0
        switch magic {
        case "NCA4":
            cond = i32(12)
            positionChannels = i32(16)
            guard (0...4).contains(cond) else { throw LoadError.shapeMismatch("cond=\(cond)") }
            guard positionChannels >= 4, ch == 2 * positionChannels, ch <= 64 else {
                throw LoadError.shapeMismatch("NCA4 stateCh=\(ch) positionCh=\(positionChannels)")
            }
            offset = 20
        case "NCAP":
            cond = i32(12)
            npool = i32(16)
            guard (0...4).contains(cond) else { throw LoadError.shapeMismatch("cond=\(cond)") }
            guard (1...8).contains(npool) else { throw LoadError.shapeMismatch("npool=\(npool)") }
            offset = 20
        case "NCA2", "NC3C":
            cond = i32(12)
            guard (0...4).contains(cond) else { throw LoadError.shapeMismatch("cond=\(cond)") }
            offset = 16
        case "NCA3", "NC3M":
            zdim = i32(12)
            cond = 2  // sin/cos phase channels
            guard (1...16).contains(zdim) else { throw LoadError.shapeMismatch("zdim=\(zdim)") }
            offset = 16
        default:
            break
        }
        let fireRate = f32(offset)
        offset += 4
        if magic == "NCA4" {
            momentumDecay = f32(offset)
            guard momentumDecay >= 0, momentumDecay <= 1 else {
                throw LoadError.shapeMismatch("momentumDecay=\(momentumDecay)")
            }
            offset += 4
        }

        let w1In = ch * (spatialDims == 3 ? 4 : 3) + cond + npool
        let outputChannels = magic == "NCA4" ? positionChannels : ch
        let baseCount = hidden * w1In + hidden + outputChannels * hidden + outputChannels
        let filmCount = zdim > 0 ? (2 * hidden * zdim + 2 * hidden) : 0
        guard data.count >= offset + (baseCount + filmCount) * 4 else { throw LoadError.truncated }

        func floats(_ start: Int, _ count: Int) -> [Float] {
            var out = [Float](repeating: 0, count: count)
            data.withUnsafeBytes { raw in
                let src = raw.baseAddress!.advanced(by: start)
                out.withUnsafeMutableBytes { dst in
                    dst.baseAddress!.copyMemory(from: src, byteCount: count * 4)
                }
            }
            return out
        }

        return NCAWeights(stateChannels: ch, positionChannels: positionChannels,
                          hidden: hidden, fireRate: fireRate, momentumDecay: momentumDecay,
                          cond: cond, zdim: zdim, npool: npool, spatialDims: spatialDims,
                          flat: floats(offset, baseCount),
                          film: floats(offset + baseCount * 4, filmCount))
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
