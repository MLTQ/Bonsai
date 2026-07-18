import AppKit

// Entry point. `bonsai` launches the desktop pet; `bonsai --render-test out.png [steps]`
// runs a headless growth rollout and writes a PNG (used to verify the Metal runtime).

let args = CommandLine.arguments
if args.count >= 2, args[1] == "--render-test" {
    let out = args.count >= 3 ? args[2] : "render_test.png"
    let steps = args.count >= 4 ? Int(args[3]) ?? 300 : 300
    let weights = args.count >= 5 ? args[4] : nil
    exit(RenderTest.run(outputPath: out, steps: steps, weightsPath: weights))
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
