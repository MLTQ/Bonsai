import AppKit
import Foundation

/// A creature = a weights file + how to condition and present it.
/// The registry is the single place new creatures get added.
struct Creature {
    let name: String
    let fileName: String
    let renderStyle: Int32
    let makeBehavior: () -> CreatureBehavior?
    /// Volumetric creatures run on NCASimulation3D + VoxelPetView.
    var volumetric: Bool = false
    /// Seed voxel for volumetric creatures (must match the trainer's seed position).
    var seed3D: (x: Int, y: Int, z: Int)? = nil

    static let registry: [Creature] = [
        Creature(name: "Bonsai", fileName: "bonsai.nca", renderStyle: 0, makeBehavior: { nil }),
        Creature(name: "Lain", fileName: "lain.nca", renderStyle: 1, makeBehavior: { LainBehavior() }),
        Creature(name: "Shoggoth", fileName: "shoggoth.nca", renderStyle: 0,
                 makeBehavior: { ShoggothBehavior() }),
        Creature(name: "Manifold", fileName: "shoggoth_manifold.nca", renderStyle: 0,
                 makeBehavior: { ManifoldBehavior() }),
        Creature(name: "Bonsai 3D", fileName: "bonsai3d.nca", renderStyle: 0,
                 makeBehavior: { nil }, volumetric: true, seed3D: (16, 10, 16)),
        Creature(name: "Shoggoth Mk. III", fileName: "shoggoth3d.nca", renderStyle: 0,
                 makeBehavior: { nil }, volumetric: true),
        Creature(name: "Shoggoth Mk. IV", fileName: "shoggoth3d_manifold.nca", renderStyle: 0,
                 makeBehavior: { nil }, volumetric: true),
    ]

    var path: String? {
        NCAWeights.weightsDir().map { $0 + "/" + fileName }
    }

    var isAvailable: Bool {
        path.map { FileManager.default.fileExists(atPath: $0) } ?? false
    }
}

/// Per-creature autonomy: supplies conditioning values each automaton step and may
/// act each display tick — on the simulation (mood switches, glitches) and on the
/// window (locomotion; window may be nil in headless render tests).
protocol CreatureBehavior: AnyObject {
    func cond(step: UInt32) -> (Float, Float, Float, Float)
    func tick(sim: NCASimulation, window: NSWindow?)
}

/// Lain: phase-conditioned cyclic creature. Mostly stares; sometimes murmurs
/// (behavior flag 1) for a while; occasionally tears a small hole of static in
/// herself, which heals. Timing is deliberately irregular.
final class LainBehavior: CreatureBehavior {
    /// Must match training: one animation cycle = 240 automaton steps.
    static let omega = 2.0 * Float.pi / 240.0

    private var talking = false
    private var nextMoodSwitch = Date().addingTimeInterval(.random(in: 15...45))
    private var nextGlitch = Date().addingTimeInterval(.random(in: 40...120))

    func cond(step: UInt32) -> (Float, Float, Float, Float) {
        let theta = Float(step) * Self.omega
        return (sin(theta), cos(theta), talking ? 1.0 : 0.0, 0.0)
    }

    func tick(sim: NCASimulation, window: NSWindow?) {
        let now = Date()
        if now >= nextMoodSwitch {
            talking.toggle()
            nextMoodSwitch = now.addingTimeInterval(
                talking ? .random(in: 5...15) : .random(in: 20...60))
        }
        if now >= nextGlitch {
            sim.damage(atGridX: .random(in: 18...46), gridY: .random(in: 16...44),
                       radius: .random(in: 2.5...4.5))
            nextGlitch = now.addingTimeInterval(.random(in: 40...120))
        }
    }
}

/// Named points in the behavior manifold, exported by training/manifold_shoggoth.py.
struct AnchorFile: Decodable {
    let z_spec: [String]
    let anchors: [String: [Float]]

    static func load(named name: String = "anchors_shoggoth.json") -> AnchorFile? {
        guard let dir = NCAWeights.weightsDir(),
              let data = FileManager.default.contents(atPath: dir + "/" + name)
        else { return nil }
        return try? JSONDecoder().decode(AnchorFile.self, from: data)
    }
}

/// The manifold shoggoth: its mood is a point z in a 10-D factor space. An
/// autopilot drifts between named anchors; an external controller (an LLM, a
/// projector daemon, or you with echo) can take the wheel by writing
/// weights/control.json: {"anchor": "dread"} or {"z": [10 floats 0..1]}.
/// External commands pause the autopilot for a few minutes, then it resumes.
/// When the mood's walkness factor (z[0]) is high, it commutes along the Dock.
final class ManifoldBehavior: CreatureBehavior {
    static let omega = 2.0 * Float.pi / 240.0
    private static let footMargin: CGFloat = 32
    private static let speed: CGFloat = 0.75

    private let anchorFile = AnchorFile.load()
    private var autopilotPausedUntil = Date.distantPast
    private var nextDrift = Date().addingTimeInterval(.random(in: 20...60))
    private var lastControlCheck = Date.distantPast
    private var controlMTime: Date?
    private var direction: CGFloat = 1

    func cond(step: UInt32) -> (Float, Float, Float, Float) {
        let theta = Float(step) * Self.omega
        return (sin(theta), cos(theta), 0, 0)
    }

    func tick(sim: NCASimulation, window: NSWindow?) {
        let now = Date()

        // External control channel (poll at 1 Hz).
        if now.timeIntervalSince(lastControlCheck) > 1.0 {
            lastControlCheck = now
            if let z = readControl() {
                sim.zTarget = z
                autopilotPausedUntil = now.addingTimeInterval(300)
            }
        }

        // Autopilot: drift between anchors when nobody is steering.
        if now >= autopilotPausedUntil, now >= nextDrift,
           let anchors = anchorFile?.anchors, let z = anchors.values.randomElement() {
            sim.zTarget = z
            nextDrift = now.addingTimeInterval(.random(in: 25...90))
        }

        // High walkness -> commute along the Dock, facing the way it's going.
        let walkness = sim.zTarget.first ?? 0
        let walking = walkness > 0.6
        sim.flipX = walking && direction < 0
        guard walking, let window, let screen = window.screen ?? NSScreen.main else { return }
        let vis = screen.visibleFrame
        var origin = window.frame.origin
        let railY = vis.minY - Self.footMargin
        origin.y += (railY - origin.y) * 0.06
        origin.x += direction * Self.speed * CGFloat((walkness - 0.6) / 0.4)
        if origin.x < vis.minX - 40 { direction = 1 }
        if origin.x > vis.maxX - window.frame.width + 40 { direction = -1 }
        window.setFrameOrigin(origin)
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
        if let name = obj["anchor"] as? String, let z = anchorFile?.anchors[name] {
            return z
        }
        if let arr = obj["z"] as? [Double] {
            return arr.map { Float(min(max($0, 0), 1)) }
        }
        return nil
    }
}

/// The latent-space shoggoth: writhes in place, and periodically commutes along
/// the top edge of the Dock. Walk episodes settle the window down to the Dock
/// rail, glide horizontally in sync with the trained gait, and turn around at
/// screen edges (mirroring the render so it faces where it's going).
final class ShoggothBehavior: CreatureBehavior {
    /// Must match training: one gait cycle = 240 automaton steps.
    static let omega = 2.0 * Float.pi / 240.0
    /// Tentacle tips sit ~8 grid px above the sprite bottom; at 4x window scale
    /// the window hangs 32 px below the Dock rail so the tips touch it.
    private static let footMargin: CGFloat = 32
    private static let speed: CGFloat = 0.75   // px per tick ≈ 22 px/s at 30 fps

    private var walking = false
    private var direction: CGFloat = 1          // +1 right, -1 left
    private var nextEpisode = Date().addingTimeInterval(.random(in: 8...20))

    func cond(step: UInt32) -> (Float, Float, Float, Float) {
        let theta = Float(step) * Self.omega
        return (sin(theta), cos(theta), walking ? 1.0 : 0.0, 0.0)
    }

    func tick(sim: NCASimulation, window: NSWindow?) {
        let now = Date()
        if now >= nextEpisode {
            walking.toggle()
            if walking, let window, let screen = window.screen ?? NSScreen.main {
                // Head toward the side with more room.
                let vis = screen.visibleFrame
                let mid = window.frame.midX
                direction = mid > vis.midX ? -1 : 1
            }
            nextEpisode = now.addingTimeInterval(
                walking ? .random(in: 10...25) : .random(in: 10...30))
        }
        // The art faces right; mirror when heading left.
        sim.flipX = walking && direction < 0

        guard walking, let window, let screen = window.screen ?? NSScreen.main else { return }
        let vis = screen.visibleFrame
        var origin = window.frame.origin

        // Settle down to the Dock rail, then glide.
        let railY = vis.minY - Self.footMargin
        origin.y += (railY - origin.y) * 0.06
        origin.x += direction * Self.speed

        // Turn around before walking off-screen.
        if origin.x < vis.minX - 40 { direction = 1 }
        if origin.x > vis.maxX - window.frame.width + 40 { direction = -1 }
        window.setFrameOrigin(origin)
    }
}
