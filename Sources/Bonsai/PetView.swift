import AppKit
import Metal
import QuartzCore

/// The pet's face to the world: hosts a CAMetalLayer, ticks the simulation on a timer,
/// and translates mouse gestures into pet interactions (drag to move, click to poke).
final class PetView: NSView {
    private let sim: NCASimulation
    private let behavior: CreatureBehavior?
    private var metalLayer: CAMetalLayer { layer as! CAMetalLayer }
    private var timer: Timer?
    private var mouseDownPoint: NSPoint?
    private var didDrag = false
    private var paused = false

    /// Automaton steps per display tick; 2 at 30 fps matches the training regime's pace.
    var stepsPerTick = 2

    init(simulation: NCASimulation, behavior: CreatureBehavior?, frame: NSRect) {
        self.sim = simulation
        self.behavior = behavior
        super.init(frame: frame)
        wantsLayer = true

        // Don't burn GPU while the displays are asleep.
        let nc = NSWorkspace.shared.notificationCenter
        nc.addObserver(forName: NSWorkspace.screensDidSleepNotification, object: nil,
                       queue: .main) { [weak self] _ in self?.paused = true }
        nc.addObserver(forName: NSWorkspace.screensDidWakeNotification, object: nil,
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
        metalLayer.contentsScale = window?.backingScaleFactor ?? 2.0
        updateDrawableSize()
        timer = Timer.scheduledTimer(withTimeInterval: 1.0 / 30.0, repeats: true) { [weak self] _ in
            self?.tick()
        }
        RunLoop.main.add(timer!, forMode: .common)  // keep animating during window drags
    }

    override func layout() {
        super.layout()
        updateDrawableSize()
    }

    private func updateDrawableSize() {
        let scale = metalLayer.contentsScale
        metalLayer.drawableSize = CGSize(width: bounds.width * scale, height: bounds.height * scale)
    }

    private func tick() {
        guard !paused else { return }
        behavior?.tick(sim: sim, window: window)
        guard let drawable = metalLayer.nextDrawable() else {
            sim.step(count: stepsPerTick)
            return
        }
        sim.step(count: stepsPerTick, renderInto: drawable.texture)
        drawable.present()
    }

    // MARK: - Interactions

    /// Convert a view-space point (bottom-left origin) to grid coordinates (top-left
    /// origin), accounting for the render mirror so pokes land where the eye sees them.
    private func gridPoint(for viewPoint: NSPoint) -> (x: Float, y: Float) {
        var gx = Float(viewPoint.x / bounds.width) * Float(sim.gridWidth)
        if sim.flipX { gx = Float(sim.gridWidth) - gx }
        let gy = Float(1.0 - viewPoint.y / bounds.height) * Float(sim.gridHeight)
        return (gx, gy)
    }

    override func mouseDown(with event: NSEvent) {
        mouseDownPoint = event.locationInWindow
        didDrag = false
    }

    override func mouseDragged(with event: NSEvent) {
        guard let start = mouseDownPoint, !didDrag else { return }
        let p = event.locationInWindow
        if hypot(p.x - start.x, p.y - start.y) > 4 {
            didDrag = true
            window?.performDrag(with: event)
        }
    }

    override func mouseUp(with event: NSEvent) {
        defer { mouseDownPoint = nil }
        guard !didDrag else { return }
        let p = convert(event.locationInWindow, from: nil)
        let g = gridPoint(for: p)
        sim.damage(atGridX: g.x, gridY: g.y, radius: 7)
    }

    override func rightMouseDown(with event: NSEvent) {
        let menu = NSMenu()
        menu.addItem(withTitle: "Regrow from Seed", action: #selector(reseed), keyEquivalent: "")
        menu.addItem(.separator())
        menu.addItem(withTitle: "Quit Bonsai", action: #selector(quit), keyEquivalent: "")
        menu.items.forEach { $0.target = self }
        NSMenu.popUpContextMenu(menu, with: event, for: self)
    }

    @objc private func reseed() { sim.reseed() }
    @objc private func quit() { NSApp.terminate(nil) }
}
