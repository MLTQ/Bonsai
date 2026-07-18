import Foundation

/// A creature = a weights file + how to condition and present it.
/// The registry is the single place new creatures get added.
struct Creature {
    let name: String
    let fileName: String
    let renderStyle: Int32
    let makeBehavior: () -> CreatureBehavior?

    static let registry: [Creature] = [
        Creature(name: "Bonsai", fileName: "bonsai.nca", renderStyle: 0, makeBehavior: { nil }),
        Creature(name: "Lain", fileName: "lain.nca", renderStyle: 1, makeBehavior: { LainBehavior() }),
    ]

    var path: String? {
        NCAWeights.weightsDir().map { $0 + "/" + fileName }
    }

    var isAvailable: Bool {
        path.map { FileManager.default.fileExists(atPath: $0) } ?? false
    }
}

/// Per-creature autonomy: supplies conditioning values each automaton step and may
/// act on the simulation each display tick (mood switches, self-inflicted glitches).
protocol CreatureBehavior: AnyObject {
    func cond(step: UInt32) -> (Float, Float, Float, Float)
    func tick(sim: NCASimulation)
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

    func tick(sim: NCASimulation) {
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
