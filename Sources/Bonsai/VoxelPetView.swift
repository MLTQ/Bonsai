import AppKit
import Metal
import QuartzCore

/// The volumetric pet's window into the world: raymarched CAMetalLayer with a
/// slowly orbiting camera. Scroll to spin, drag to move the window, click to
/// blast a spherical crater into the creature (it heals from the inside out).
/// For cyclic 3D creatures (cond >= 3) it also runs idle/walk episodes and
/// glides the window along the Dock rail while walking.
final class VoxelPetView: NSView {
    private let sim: NCASimulation3D
    private let cyclic: Bool
    private var metalLayer: CAMetalLayer { layer as! CAMetalLayer }
    private var timer: Timer?
    private var mouseDownPoint: NSPoint?
    private var didDrag = false
    private var paused = false

    private var walking = false
    private var direction: CGFloat = 1
    private var nextEpisode = Date().addingTimeInterval(.random(in: 10...25))

    // Manifold (zdim > 0): anchor autopilot + control.json steering, 3D edition.
    private let anchorFile3D = AnchorFile.load(named: "anchors_shoggoth3d.json")
    private var autopilotPausedUntil = Date.distantPast
    private var nextDrift = Date().addingTimeInterval(.random(in: 20...60))
    private var lastControlCheck = Date.distantPast
    private var controlMTime: Date?

    var stepsPerTick = 2
    /// Idle camera drift (radians/tick). The creature always turns to be seen.
    var orbitRate: Float = 0.006

    init(simulation: NCASimulation3D, cyclic: Bool, frame: NSRect) {
        self.sim = simulation
        self.cyclic = cyclic
        super.init(frame: frame)
        wantsLayer = true
        if simulation.condCount >= 2 {
            // Phase clock for cyclic (cond=3, +behavior flag) and manifold (cond=2)
            // creatures alike; the shader reads only its compiled cond width.
            simulation.condProvider = { [weak self] step in
                let theta = Float(step) * LainBehavior.omega
                return (sin(theta), cos(theta), (self?.walking ?? false) ? 1.0 : 0.0, 0.0)
            }
        }
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
        RunLoop.main.add(timer!, forMode: .common)
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
        sim.azimuth += orbitRate
        if sim.zdim > 0 { manifoldTick() }
        if cyclic || sim.zdim > 0 { walkTick() }
        guard let drawable = metalLayer.nextDrawable() else {
            sim.step(count: stepsPerTick)
            return
        }
        sim.step(count: stepsPerTick, renderInto: drawable.texture)
        drawable.present()
    }

    private func manifoldTick() {
        let now = Date()
        if now.timeIntervalSince(lastControlCheck) > 1.0 {
            lastControlCheck = now
            if let z = readControl() {
                sim.zTarget = z
                autopilotPausedUntil = now.addingTimeInterval(300)
            }
        }
        if now >= autopilotPausedUntil, now >= nextDrift,
           let anchors = anchorFile3D?.anchors, let z = anchors.values.randomElement() {
            sim.zTarget = z
            nextDrift = now.addingTimeInterval(.random(in: 25...90))
        }
        // Walkness (z[0]) drives the Dock commute for manifold creatures.
        walking = (sim.zTarget.first ?? 0) > 0.6
    }

    private func readControl() -> [Float]? {
        guard let dir = NCAWeights.weightsDir() else { return nil }
        let path = dir + "/control.json"
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: path),
              let mtime = attrs[.modificationDate] as? Date else { return nil }
        if let seen = controlMTime, mtime <= seen { return nil }
        controlMTime = mtime
        guard let data = FileManager.default.contents(atPath: path),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        if let name = obj["anchor"] as? String, let z = anchorFile3D?.anchors[name] { return z }
        if let arr = obj["z"] as? [Double] { return arr.map { Float(min(max($0, 0), 1)) } }
        return nil
    }

    private func walkTick() {
        let now = Date()
        // Pure cyclic creatures toggle walk episodes on a timer; manifold creatures'
        // `walking` is set by manifoldTick from the walkness factor.
        if sim.zdim == 0, now >= nextEpisode {
            walking.toggle()
            if walking, let window, let screen = window.screen ?? NSScreen.main {
                direction = window.frame.midX > screen.visibleFrame.midX ? -1 : 1
            }
            nextEpisode = now.addingTimeInterval(
                walking ? .random(in: 10...25) : .random(in: 12...35))
        }
        guard walking, let window, let screen = window.screen ?? NSScreen.main else { return }
        let vis = screen.visibleFrame
        var origin = window.frame.origin
        origin.y += (vis.minY - 20 - origin.y) * 0.05
        origin.x += direction * 0.7
        if origin.x < vis.minX - 40 { direction = 1 }
        if origin.x > vis.maxX - window.frame.width + 40 { direction = -1 }
        window.setFrameOrigin(origin)
        // Face the direction of travel: steer the orbit toward the profile view.
        let target: Float = direction > 0 ? .pi / 2 : -.pi / 2
        sim.azimuth += (target - sim.azimuth) * 0.02
    }

    // MARK: - Interactions

    override func scrollWheel(with event: NSEvent) {
        sim.azimuth += Float(event.scrollingDeltaX + event.scrollingDeltaY) * 0.01
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
        let ndcX = Float(p.x / bounds.width - 0.5)
        let ndcY = Float(p.y / bounds.height - 0.5)  // view is bottom-left origin; camera up matches
        if let hit = sim.pick(ndcX: ndcX, ndcY: ndcY) {
            sim.damage(atVoxelX: hit.x, y: hit.y, z: hit.z, radius: 4.5)
        }
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
