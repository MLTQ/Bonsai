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
