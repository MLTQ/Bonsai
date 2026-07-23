import CoreGraphics
import Foundation
import ImageIO
import Metal
import UniformTypeIdentifiers

/// Headless production-runtime evidence for an FX2D checkpoint.
enum FusedRenderTest {
    static func run(outputPath: String, steps: Int, weightsPath: String,
                    statePath: String) -> Int32 {
        guard let device = MTLCreateSystemDefaultDevice(),
              let weights = try? FusedNCAWeights.load(from: weightsPath),
              let simulation = FusedNCASimulation(device: device, weights: weights),
              simulation.loadState(from: statePath)
        else {
            FileHandle.standardError.write(
                Data("failed to initialize fused simulation\n".utf8))
            return 1
        }
        simulation.step(count: steps)
        let rgba = simulation.readRGBA()
        let scale = 4
        let width = simulation.gridWidth * scale
        let height = simulation.gridHeight * scale
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let source = ((y / scale) * simulation.gridWidth + x / scale) * 4
                let destination = (y * width + x) * 4
                let alpha = min(max(rgba[source + 3], 0), 1)
                for channel in 0..<3 {
                    pixels[destination + channel] = UInt8(
                        min(max(rgba[source + channel], 0), alpha) * 255)
                }
                pixels[destination + 3] = UInt8(alpha * 255)
            }
        }
        guard let context = CGContext(
                data: &pixels, width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: width * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
              let image = context.makeImage(),
              let destination = CGImageDestinationCreateWithURL(
                URL(fileURLWithPath: outputPath) as CFURL,
                UTType.png.identifier as CFString, 1, nil)
        else { return 1 }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { return 1 }
        print("wrote \(outputPath) after \(steps) fused steps")
        return 0
    }
}
