import AppKit

// Entry point. `bonsai` launches the desktop pet; `bonsai --render-test out.png [steps]`
// runs a headless growth rollout and writes a PNG (used to verify the Metal runtime).

let args = CommandLine.arguments
if args.count >= 6, args[1] == "--render-fused" {
    let steps = Int(args[3]) ?? 32
    exit(FusedRenderTest.run(
        outputPath: args[2], steps: steps, weightsPath: args[4], statePath: args[5]))
}
if args.count >= 2, args[1] == "--render-test" {
    let out = args.count >= 3 ? args[2] : "render_test.png"
    let steps = args.count >= 4 ? Int(args[3]) ?? 300 : 300
    let weights = args.count >= 5 ? args[4] : nil
    exit(RenderTest.run(outputPath: out, steps: steps, weightsPath: weights))
}
if args.count >= 2, args[1] == "--render-tex" {
    let out = args.count >= 3 ? args[2] : "render_tex.png"
    let steps = args.count >= 4 ? Int(args[3]) ?? 300 : 300
    let weights = args.count >= 5 ? args[4] : nil
    let crisp = args.count >= 6 ? args[5] == "1" : true
    exit(RenderTest.runTexture(outputPath: out, steps: steps, weightsPath: weights, crisp: crisp))
}
if args.count >= 4, args[1] == "--render-vol" {
    let azimuth = args.count >= 5 ? Float(args[4]) ?? 25 : 25
    exit(RenderTest.renderVolume(volumePath: args[2], outputPath: args[3],
                                 azimuthDegrees: azimuth))
}
if args.count >= 2, args[1] == "--render-test3d" {
    let out = args.count >= 3 ? args[2] : "render_test3d.png"
    let steps = args.count >= 4 ? Int(args[3]) ?? 300 : 300
    let weights = args.count >= 5 ? args[4] : NCAWeights.weightsDir().map { $0 + "/bonsai3d.nca" }
    let azimuth = args.count >= 6 ? Float(args[5]) ?? 30 : 30
    exit(RenderTest.run3D(outputPath: out, steps: steps, weightsPath: weights,
                          azimuthDegrees: azimuth))
}
if args.count >= 6, args[1] == "--render-seq3d" {
    let azimuth = args.count >= 7 ? Float(args[6]) ?? 30 : 30
    exit(RenderTest.runSequence3D(outDir: args[2], count: Int(args[3]) ?? 24,
                                  stride: Int(args[4]) ?? 10, weightsPath: args[5],
                                  azimuthDegrees: azimuth))
}
if args.count >= 5, args[1] == "--render-seq" {
    let outDir = args[2]
    let count = Int(args[3]) ?? 24
    let stride = Int(args[4]) ?? 10
    let weights = args.count >= 6 ? args[5] : nil
    exit(RenderTest.runSequence(outDir: outDir, count: count, stride: stride, weightsPath: weights))
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.accessory)  // no Dock icon; lives in the status bar
app.run()
