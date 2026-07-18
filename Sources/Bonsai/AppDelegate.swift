import AppKit
import Metal

/// Wires everything together: transparent floating window, status-bar item,
/// creature loading/switching, and hot-reload of the current creature's weights.
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var window: NSWindow!
    private var statusItem: NSStatusItem!
    private var sim: NCASimulation?
    private var sim3D: NCASimulation3D?
    private var behavior: CreatureBehavior?
    private var currentCreature: Creature?
    private var weightsMTime: Date?
    private var reloadTimer: Timer?

    static let windowSize: CGFloat = 256

    func applicationDidFinishLaunching(_ notification: Notification) {
        let available = Creature.registry.filter { $0.isAvailable }
        guard !available.isEmpty else {
            fatalErrorAlert("""
            No weights files found.
            Train one first: cd training && python3 train_nca.py
            Or set BONSAI_WEIGHTS_DIR to a directory of .nca files.
            """)
            return
        }
        makeWindow()

        let savedName = UserDefaults.standard.string(forKey: "creature")
        let creature = available.first { $0.name == savedName } ?? available[0]
        load(creature: creature)

        makeStatusItem()
        watchWeights()
    }

    // MARK: - Creatures

    private func load(creature: Creature) {
        guard let path = creature.path,
              let device = MTLCreateSystemDefaultDevice(),
              let weights = try? NCAWeights.load(from: path)
        else {
            fatalErrorAlert("Failed to load creature '\(creature.name)'")
            return
        }
        let size = Self.windowSize
        let frame = NSRect(x: 0, y: 0, width: size, height: size)

        if creature.volumetric {
            guard let sim3D = NCASimulation3D(device: device, weights: weights,
                                              seed: creature.seed3D) else {
                fatalErrorAlert("Failed to init 3D simulation for '\(creature.name)'")
                return
            }
            self.sim3D = sim3D
            self.sim = nil
            self.behavior = nil
            window.contentView = VoxelPetView(simulation: sim3D,
                                              cyclic: weights.cond >= 3, frame: frame)
        } else {
            guard let sim = NCASimulation(device: device, weights: weights) else {
                fatalErrorAlert("Failed to init simulation for '\(creature.name)'")
                return
            }
            sim.renderStyle = creature.renderStyle
            let behavior = creature.makeBehavior()
            if let behavior {
                sim.condProvider = { [weak behavior] step in
                    behavior?.cond(step: step) ?? (0, 0, 0, 0)
                }
            }
            self.sim = sim
            self.sim3D = nil
            self.behavior = behavior
            window.contentView = PetView(simulation: sim, behavior: behavior, frame: frame)
        }
        self.currentCreature = creature
        self.weightsMTime = fileMTime(path)
        UserDefaults.standard.set(creature.name, forKey: "creature")
    }

    @objc private func switchCreature(_ sender: NSMenuItem) {
        guard let creature = sender.representedObject as? Creature else { return }
        load(creature: creature)
        rebuildStatusMenu()
    }

    // MARK: - Window / status item

    private func makeWindow() {
        let size = Self.windowSize
        let screen = NSScreen.main?.visibleFrame ?? NSRect(x: 0, y: 0, width: 1440, height: 900)
        let origin = NSPoint(x: screen.maxX - size - 40, y: screen.minY + 40)
        window = NSWindow(contentRect: NSRect(origin: origin, size: NSSize(width: size, height: size)),
                          styleMask: [.borderless], backing: .buffered, defer: false)
        window.isOpaque = false
        window.backgroundColor = .clear
        window.hasShadow = false
        window.level = .floating
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        window.isMovableByWindowBackground = false  // PetView arbitrates click-vs-drag
        window.setFrameAutosaveName("BonsaiPetWindow")
        window.makeKeyAndOrderFront(nil)
    }

    private func makeStatusItem() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        statusItem.button?.title = "🌱"
        rebuildStatusMenu()
    }

    private func rebuildStatusMenu() {
        let menu = NSMenu()
        let creatureMenu = NSMenu()
        for creature in Creature.registry where creature.isAvailable {
            let item = NSMenuItem(title: creature.name, action: #selector(switchCreature(_:)),
                                  keyEquivalent: "")
            item.representedObject = creature
            item.target = self
            item.state = creature.name == currentCreature?.name ? .on : .off
            creatureMenu.addItem(item)
        }
        let creatureItem = NSMenuItem(title: "Creature", action: nil, keyEquivalent: "")
        menu.addItem(creatureItem)
        menu.setSubmenu(creatureMenu, for: creatureItem)
        menu.addItem(withTitle: "Regrow from Seed", action: #selector(reseed), keyEquivalent: "r")
        menu.addItem(withTitle: "Reload Weights", action: #selector(reloadWeights), keyEquivalent: "")
        menu.addItem(.separator())
        menu.addItem(withTitle: "Quit Bonsai", action: #selector(quit), keyEquivalent: "q")
        menu.items.forEach { if $0.action != nil { $0.target = self } }
        statusItem.menu = menu
    }

    // MARK: - Weights hot-reload

    /// Poll the current creature's weights file; when training writes a new
    /// checkpoint, swap it in live (or rebuild the sim if its shape changed).
    private func watchWeights() {
        reloadTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            guard let self, let path = self.currentCreature?.path,
                  let mtime = self.fileMTime(path) else { return }
            if self.weightsMTime.map({ mtime > $0 }) ?? true {
                self.weightsMTime = mtime
                self.reloadWeights()
            }
        }
    }

    private func fileMTime(_ path: String) -> Date? {
        (try? FileManager.default.attributesOfItem(atPath: path))?[.modificationDate] as? Date
    }

    @objc private func reseed() {
        sim?.reseed()
        sim3D?.reseed()
    }

    @objc private func reloadWeights() {
        guard let creature = currentCreature, let path = creature.path,
              let weights = try? NCAWeights.load(from: path) else { return }
        let ok = creature.volumetric ? sim3D?.updateWeights(weights) : sim?.updateWeights(weights)
        if ok != true {
            load(creature: creature)  // shape changed: rebuild
        }
    }

    @objc private func quit() { NSApp.terminate(nil) }

    private func fatalErrorAlert(_ message: String) {
        let alert = NSAlert()
        alert.messageText = "Bonsai"
        alert.informativeText = message
        alert.runModal()
        NSApp.terminate(nil)
    }
}
