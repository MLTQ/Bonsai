import AppKit
import Metal
import QuartzCore

/// CAMetalLayer host and mouse interaction surface for a fused 2D simulation.
final class FusedPetView: NSView {
    private let sim: FusedNCASimulation
    private var metalLayer: CAMetalLayer { layer as! CAMetalLayer }
    private var timer: Timer?
    private var mouseDownPoint: NSPoint?
    private var didDrag = false
    private var paused = false
    var stepsPerTick = 2

    init(simulation: FusedNCASimulation, frame: NSRect) {
        self.sim = simulation
        super.init(frame: frame)
        wantsLayer = true
        let center = NSWorkspace.shared.notificationCenter
        center.addObserver(forName: NSWorkspace.screensDidSleepNotification, object: nil,
                           queue: .main) { [weak self] _ in self?.paused = true }
        center.addObserver(forName: NSWorkspace.screensDidWakeNotification, object: nil,
                           queue: .main) { [weak self] _ in self?.paused = false }
    }

    required init?(coder: NSCoder) { fatalError("not used") }

    override func makeBackingLayer() -> CALayer {
        let layer = CAMetalLayer()
        layer.device = sim.device
        layer.pixelFormat = .bgra8Unorm
        layer.framebufferOnly = false
        layer.isOpaque = false
        return layer
    }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        guard window != nil else { timer?.invalidate(); timer = nil; return }
        metalLayer.contentsScale = window?.backingScaleFactor ?? 2
        updateDrawableSize()
        timer = Timer.scheduledTimer(withTimeInterval: 1.0 / 30.0, repeats: true) {
            [weak self] _ in self?.tick()
        }
        RunLoop.main.add(timer!, forMode: .common)
    }

    override func layout() {
        super.layout()
        updateDrawableSize()
    }

    private func updateDrawableSize() {
        let scale = metalLayer.contentsScale
        metalLayer.drawableSize = CGSize(
            width: bounds.width * scale, height: bounds.height * scale)
    }

    private func tick() {
        guard !paused else { return }
        guard let drawable = metalLayer.nextDrawable() else {
            sim.step(count: stepsPerTick)
            return
        }
        sim.step(count: stepsPerTick, renderInto: drawable.texture)
        drawable.present()
    }

    private func gridPoint(for point: NSPoint) -> (x: Float, y: Float) {
        var x = Float(point.x / bounds.width) * Float(sim.gridWidth)
        if sim.flipX { x = Float(sim.gridWidth) - x }
        let y = Float(1 - point.y / bounds.height) * Float(sim.gridHeight)
        return (x, y)
    }

    override func mouseDown(with event: NSEvent) {
        mouseDownPoint = event.locationInWindow
        didDrag = false
    }

    override func mouseDragged(with event: NSEvent) {
        guard let start = mouseDownPoint, !didDrag else { return }
        let point = event.locationInWindow
        if hypot(point.x - start.x, point.y - start.y) > 4 {
            didDrag = true
            window?.performDrag(with: event)
        }
    }

    override func mouseUp(with event: NSEvent) {
        defer { mouseDownPoint = nil }
        guard !didDrag else { return }
        let point = gridPoint(for: convert(event.locationInWindow, from: nil))
        sim.damage(atGridX: point.x, gridY: point.y, radius: 10)
    }

    override func rightMouseDown(with event: NSEvent) {
        let menu = NSMenu()
        menu.addItem(withTitle: "Reset to Pose 0", action: #selector(reseed), keyEquivalent: "")
        menu.addItem(.separator())
        menu.addItem(withTitle: "Quit Bonsai", action: #selector(quit), keyEquivalent: "")
        menu.items.forEach { $0.target = self }
        NSMenu.popUpContextMenu(menu, with: event, for: self)
    }

    @objc private func reseed() { sim.reseed() }
    @objc private func quit() { NSApp.terminate(nil) }
}
