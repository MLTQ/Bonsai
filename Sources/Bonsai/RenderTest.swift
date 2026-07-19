import Foundation
import Metal
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

/// Headless verification: run the NCA without a window and write PNGs.
/// `--render-test out.png [steps] [weights]` — grow from seed, one snapshot.
/// `--render-seq outdir count stride [weights]` — warm up, then dump an animation
/// frame every `stride` steps (used to build verification GIFs of the cyclic NCA).
enum RenderTest {
    /// Volumetric grid edge for headless runs ($BONSAI_GRID3, default 32).
    private static func envGrid() -> Int {
        Int(ProcessInfo.processInfo.environment["BONSAI_GRID3"] ?? "32") ?? 32
    }

    private static func makeSim(weightsPath: String?) -> NCASimulation? {
        let path = weightsPath ?? NCAWeights.defaultPath()
        guard let path else {
            FileHandle.standardError.write(Data("no weights file found\n".utf8))
            return nil
        }
        guard let device = MTLCreateSystemDefaultDevice(),
              let weights = try? NCAWeights.load(from: path),
              let sim = NCASimulation(device: device, weights: weights)
        else {
            FileHandle.standardError.write(Data("failed to init simulation from \(path)\n".utf8))
            return nil
        }
        if weights.cond >= 2 {
            // Phase-conditioned creature: run its cycle (behavior flag ON for NCA2).
            sim.condProvider = { step in
                let theta = Float(step) * LainBehavior.omega
                return (sin(theta), cos(theta), 1.0, 0.0)
            }
        } else if weights.cond == 1 {
            // Single-flag creature (clockless or state-attractor). $BONSAI_STATE picks
            // the state for headless renders (default 1).
            let flag = Float(ProcessInfo.processInfo.environment["BONSAI_STATE"] ?? "1") ?? 1
            sim.condProvider = { _ in (flag, 0.0, 0.0, 0.0) }
        }
        if weights.zdim > 0 {
            // Manifold creature: pick z via $BONSAI_ANCHOR (name) or $BONSAI_Z (csv).
            let env = ProcessInfo.processInfo.environment
            if let csv = env["BONSAI_Z"] {
                sim.setZ(csv.split(separator: ",").compactMap { Float($0) })
            } else {
                let name = env["BONSAI_ANCHOR"] ?? "walk"
                if let z = AnchorFile.load()?.anchors[name] { sim.setZ(z) }
            }
        }
        print("weights: \(path) (cond \(weights.cond), zdim \(weights.zdim))")
        return sim
    }

    static func run(outputPath: String, steps: Int, weightsPath: String? = nil) -> Int32 {
        guard let sim = makeSim(weightsPath: weightsPath) else { return 1 }
        sim.step(count: steps)
        guard writePNG(sim: sim, to: outputPath) else { return 1 }
        print("wrote \(outputPath) after \(steps) steps")
        return 0
    }

    static func runSequence(outDir: String, count: Int, stride: Int,
                            weightsPath: String? = nil) -> Int32 {
        guard let sim = makeSim(weightsPath: weightsPath) else { return 1 }
        try? FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)
        sim.step(count: 300)  // grow to maturity first
        for i in 0..<count {
            sim.step(count: stride)
            let path = String(format: "%@/frame_%03d.png", outDir, i)
            guard writePNG(sim: sim, to: path) else { return 1 }
        }
        print("wrote \(count) frames (every \(stride) steps) to \(outDir)")
        return 0
    }

    /// Volumetric: grow for N steps, raymarch offscreen, write PNG.
    /// `bonsai --render-test3d out.png [steps] [weights] [azimuthDeg]`
    static func run3D(outputPath: String, steps: Int, weightsPath: String?,
                      azimuthDegrees: Float) -> Int32 {
        guard let path = weightsPath,
              let device = MTLCreateSystemDefaultDevice(),
              let weights = try? NCAWeights.load(from: path),
              let sim = NCASimulation3D(
                  device: device, weights: weights, grid: envGrid(),
                  seed: path.contains("bonsai3d")
                      ? (envGrid() / 2, envGrid() / 3, envGrid() / 2) : nil)
        else {
            FileHandle.standardError.write(Data("failed to init 3D simulation\n".utf8))
            return 1
        }
        if weights.cond >= 2 {
            sim.condProvider = { step in
                let theta = Float(step) * LainBehavior.omega
                return (sin(theta), cos(theta), 1.0, 0.0)
            }
        }
        if weights.zdim > 0 {
            let env = ProcessInfo.processInfo.environment
            if let csv = env["BONSAI_Z"] {
                sim.setZ(csv.split(separator: ",").compactMap { Float($0) })
            } else if let z = AnchorFile.load(named: "anchors_shoggoth3d.json")?
                        .anchors[env["BONSAI_ANCHOR"] ?? "walk"] {
                sim.setZ(z)
            }
        }
        sim.azimuth = azimuthDegrees * .pi / 180
        print("weights: \(path) (3D, cond \(weights.cond))")
        sim.step(count: steps)
        guard let bytes = sim.renderOffscreen(size: 512),
              writeRGBA8PNG(bytes: bytes, width: 512, height: 512, to: outputPath)
        else { return 1 }
        print("wrote \(outputPath) after \(steps) steps (azimuth \(azimuthDegrees) deg)")
        return 0
    }

    /// Volumetric animation evidence: one persistent sim, a PNG every `stride` steps
    /// while the camera orbits slowly. `--render-seq3d outdir count stride weights [azStartDeg]`
    static func runSequence3D(outDir: String, count: Int, stride: Int,
                              weightsPath: String?, azimuthDegrees: Float) -> Int32 {
        guard let path = weightsPath,
              let device = MTLCreateSystemDefaultDevice(),
              let weights = try? NCAWeights.load(from: path),
              let sim = NCASimulation3D(
                  device: device, weights: weights, grid: envGrid(),
                  seed: path.contains("bonsai3d")
                      ? (envGrid() / 2, envGrid() / 3, envGrid() / 2) : nil)
        else {
            FileHandle.standardError.write(Data("failed to init 3D simulation\n".utf8))
            return 1
        }
        if weights.cond >= 2 {
            sim.condProvider = { step in
                let theta = Float(step) * LainBehavior.omega
                return (sin(theta), cos(theta), 1.0, 0.0)
            }
        }
        if weights.zdim > 0 {
            let env = ProcessInfo.processInfo.environment
            if let csv = env["BONSAI_Z"] {
                sim.setZ(csv.split(separator: ",").compactMap { Float($0) })
            } else if let z = AnchorFile.load(named: "anchors_shoggoth3d.json")?
                        .anchors[env["BONSAI_ANCHOR"] ?? "walk"] {
                sim.setZ(z)
            }
        }
        try? FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)
        sim.azimuth = azimuthDegrees * .pi / 180
        sim.step(count: 300)
        for i in 0..<count {
            sim.step(count: stride)
            sim.azimuth += 0.02
            guard let bytes = sim.renderOffscreen(size: 256),
                  writeRGBA8PNG(bytes: bytes, width: 256, height: 256,
                                to: String(format: "%@/frame_%03d.png", outDir, i))
            else { return 1 }
        }
        print("wrote \(count) 3D frames (every \(stride) steps) to \(outDir)")
        return 0
    }

    /// Raymarch a raw authored volume (float32 RGBA, (z,y,x) order) with no NCA at all —
    /// art preview through the production renderer. `--render-vol vol.raw out.png [azDeg]`
    static func renderVolume(volumePath: String, outputPath: String,
                             azimuthDegrees: Float) -> Int32 {
        let grid = envGrid()
        guard let data = FileManager.default.contents(atPath: volumePath),
              data.count == grid * grid * grid * 4 * 4
        else {
            FileHandle.standardError.write(
                Data("volume file missing or wrong size for grid \(grid)\n".utf8))
            return 1
        }
        var rgba = [Float](repeating: 0, count: grid * grid * grid * 4)
        data.withUnsafeBytes { raw in
            rgba.withUnsafeMutableBytes { dst in
                dst.baseAddress!.copyMemory(from: raw.baseAddress!, byteCount: data.count)
            }
        }
        // Any 3D weights serve as a host; we never step the automaton.
        guard let wpath = NCAWeights.weightsDir().map({ $0 + "/bonsai3d.nca" }),
              let device = MTLCreateSystemDefaultDevice(),
              let weights = try? NCAWeights.load(from: wpath),
              let sim = NCASimulation3D(device: device, weights: weights, grid: grid)
        else {
            FileHandle.standardError.write(Data("no host weights for volume preview\n".utf8))
            return 1
        }
        sim.loadStateRGBA(rgba)
        sim.azimuth = azimuthDegrees * .pi / 180
        guard let bytes = sim.renderOffscreen(size: 512),
              writeRGBA8PNG(bytes: bytes, width: 512, height: 512, to: outputPath)
        else { return 1 }
        print("raymarched \(volumePath) (grid \(grid)) -> \(outputPath)")
        return 0
    }

    private static func writeRGBA8PNG(bytes: [UInt8], width: Int, height: Int,
                                      to outputPath: String) -> Bool {
        var pixels = bytes
        let ctx = CGContext(data: &pixels, width: width, height: height,
                            bitsPerComponent: 8, bytesPerRow: width * 4,
                            space: CGColorSpace(name: CGColorSpace.sRGB)!,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        guard let image = ctx?.makeImage(),
              let dest = CGImageDestinationCreateWithURL(
                  URL(fileURLWithPath: outputPath) as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { return false }
        CGImageDestinationAddImage(dest, image, nil)
        return CGImageDestinationFinalize(dest)
    }

    private static func writePNG(sim: NCASimulation, to outputPath: String, scale: Int = 4) -> Bool {
        let rgba = sim.readRGBA()
        let w = sim.gridWidth, h = sim.gridHeight
        var pixels = [UInt8](repeating: 0, count: w * scale * h * scale * 4)
        for y in 0..<(h * scale) {
            for x in 0..<(w * scale) {
                let src = ((y / scale) * w + (x / scale)) * 4
                let dst = (y * w * scale + x) * 4
                for c in 0..<4 {
                    pixels[dst + c] = UInt8(max(0, min(1, rgba[src + c])) * 255)
                }
            }
        }
        let ctx = CGContext(data: &pixels, width: w * scale, height: h * scale,
                            bitsPerComponent: 8, bytesPerRow: w * scale * 4,
                            space: CGColorSpace(name: CGColorSpace.sRGB)!,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        guard let image = ctx?.makeImage(),
              let dest = CGImageDestinationCreateWithURL(
                  URL(fileURLWithPath: outputPath) as CFURL, UTType.png.identifier as CFString, 1, nil)
        else {
            FileHandle.standardError.write(Data("failed to encode PNG\n".utf8))
            return false
        }
        CGImageDestinationAddImage(dest, image, nil)
        return CGImageDestinationFinalize(dest)
    }
}
