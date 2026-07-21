import AppKit

/// The state-space explorer: a small floating panel showing a 2D embedding of the
/// creature's mood manifold (built by tools/make_statemap.py). Drag the cursor and
/// the creature morphs — the panel inverts 2D -> 10D by inverse-distance-weighted
/// nearest neighbors and writes weights/control.json, so it steers through the same
/// channel as the trace daemon and the autopilot politely yields.
final class StateMapPanel: NSPanel {
    private let mapView: StateMapView

    /// `live` supplies the creature's CURRENT position: a [x, y] in map
    /// coordinates, or a full z vector (projected onto the map by kNN).
    static func make(for creature: Creature?,
                     live: (() -> [Double]?)? = nil) -> StateMapPanel? {
        if let name = creature?.stateMapName, let map = StateMap.load(named: name) {
            return StateMapPanel(map: map, live: live)
        }
        if let states = creature?.flagStates {
            return StateMapPanel(map: StateMap.islands(states), live: live)
        }
        if creature?.volumetric == true {
            // Cyclic creature: its true state space is a phase circle, not the
            // shoggoth manifold the old fallback showed (Max caught this twice).
            return StateMapPanel(map: StateMap.phaseRing(), live: live)
        }
        return StateMap.load(named: "statemap_2d.json").map { StateMapPanel(map: $0, live: live) }
    }

    private init(map: StateMap, live: (() -> [Double]?)? = nil) {
        mapView = StateMapView(map: map)
        mapView.live = live
        let rect = NSRect(x: 0, y: 0, width: 300, height: 320)
        super.init(contentRect: rect,
                   styleMask: [.titled, .closable, .utilityWindow, .nonactivatingPanel],
                   backing: .buffered, defer: false)
        title = "State Space"
        isFloatingPanel = true
        level = .floating
        contentView = mapView
        setFrameAutosaveName("BonsaiStateMap")
    }
}

struct StateMap: Decodable {
    let method: String
    let points: [[Double]]
    let z: [[Double]]
    let anchors: [String: [Double]]
    /// islands mode only: control anchor name per point (steer sends anchors, not z)
    var pointControls: [String]? = nil

    private enum CodingKeys: String, CodingKey { case method, points, z, anchors }

    init(method: String, points: [[Double]], z: [[Double]],
         anchors: [String: [Double]], pointControls: [String]? = nil) {
        self.method = method
        self.points = points
        self.z = z
        self.anchors = anchors
        self.pointControls = pointControls
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        method = try c.decode(String.self, forKey: .method)
        points = try c.decode([[Double]].self, forKey: .points)
        z = try c.decode([[Double]].self, forKey: .z)
        anchors = try c.decode([String: [Double]].self, forKey: .anchors)
        pointControls = nil
    }

    static func load(named name: String) -> StateMap? {
        guard let dir = NCAWeights.weightsDir(),
              let data = FileManager.default.contents(atPath: dir + "/" + name)
        else { return nil }
        return try? JSONDecoder().decode(StateMap.self, from: data)
    }

    /// The honest map for a phase-conditioned cyclic creature: its state IS a
    /// point on a circle. Display-only (steer() ignores method "ring").
    static func phaseRing() -> StateMap {
        var pts: [[Double]] = []
        for i in 0..<96 {
            let a = 2.0 * Double.pi * Double(i) / 96.0
            pts.append([0.5 + 0.38 * cos(a), 0.5 + 0.38 * sin(a)])
        }
        return StateMap(method: "ring", points: pts,
                        z: Array(repeating: [0.0], count: pts.count),
                        anchors: ["cycle": [0.5, 0.5]])
    }

    /// Two-island map for flag creatures: labeled clusters and the road between.
    static func islands(_ states: [(String, String)]) -> StateMap {
        var pts: [[Double]] = []
        var controls: [String] = []
        var anchors: [String: [Double]] = [:]
        let centers: [[Double]] = [[0.25, 0.5], [0.75, 0.5]]
        var rng = SystemRandomNumberGenerator()
        for (i, (label, control)) in states.prefix(2).enumerated() {
            anchors[label] = centers[i]
            for _ in 0..<24 {
                let dx = Double.random(in: -0.07...0.07, using: &rng)
                let dy = Double.random(in: -0.07...0.07, using: &rng)
                pts.append([centers[i][0] + dx, centers[i][1] + dy])
                controls.append(control)
            }
        }
        for t in stride(from: 0.1, through: 0.9, by: 0.05) {
            pts.append([0.25 + 0.5 * t, 0.5 + Double.random(in: -0.02...0.02, using: &rng)])
            controls.append(t < 0.5 ? states[0].1 : states[min(1, states.count - 1)].1)
        }
        return StateMap(method: "islands", points: pts,
                        z: Array(repeating: [0.0], count: pts.count),
                        anchors: anchors, pointControls: controls)
    }
}

final class StateMapView: NSView {
    private let map: StateMap
    private var cursor = CGPoint(x: 0.5, y: 0.5)
    private var lastWrite = Date.distantPast
    /// Live creature position: [x, y] map coords, or a z vector to kNN-project.
    var live: (() -> [Double]?)?
    private var liveTimer: Timer?

    init(map: StateMap) {
        self.map = map
        super.init(frame: .zero)
        liveTimer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) {
            [weak self] t in
            guard let self else { t.invalidate(); return }
            if self.live != nil { self.needsDisplay = true }
        }
    }

    deinit { liveTimer?.invalidate() }

    /// Where the creature IS right now, in normalized map coordinates.
    private func livePoint() -> [Double]? {
        guard let v = live?() else { return nil }
        if v.count == 2 { return v }
        // z vector: inverse of steer() — kNN in z-space, blend the 2D positions.
        var dists: [(Double, Int)] = []
        for (i, z) in map.z.enumerated() {
            var d = 0.0
            for j in 0..<min(z.count, v.count) { d += (z[j] - v[j]) * (z[j] - v[j]) }
            dists.append((d, i))
        }
        dists.sort { $0.0 < $1.0 }
        let k = min(8, dists.count)
        guard k > 0 else { return nil }
        var x = 0.0, y = 0.0, total = 0.0
        for j in 0..<k {
            let w = 1.0 / (dists[j].0 + 1e-6)
            x += map.points[dists[j].1][0] * w
            y += map.points[dists[j].1][1] * w
            total += w
        }
        return [x / total, y / total]
    }

    required init?(coder: NSCoder) { fatalError("not used") }

    override func draw(_ dirtyRect: NSRect) {
        NSColor(calibratedWhite: 0.12, alpha: 1).setFill()
        bounds.fill()
        let inset = bounds.insetBy(dx: 16, dy: 16)

        NSColor(calibratedWhite: 0.45, alpha: 0.5).setFill()
        for p in map.points {
            let pt = place(p, in: inset)
            NSBezierPath(ovalIn: NSRect(x: pt.x - 1.2, y: pt.y - 1.2, width: 2.4, height: 2.4)).fill()
        }

        let labelAttrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 10, weight: .semibold),
            .foregroundColor: NSColor(calibratedRed: 0.92, green: 0.62, blue: 0.45, alpha: 1),
        ]
        for (name, p) in map.anchors {
            let pt = place(p, in: inset)
            NSColor(calibratedRed: 0.85, green: 0.47, blue: 0.34, alpha: 1).setFill()
            NSBezierPath(ovalIn: NSRect(x: pt.x - 3, y: pt.y - 3, width: 6, height: 6)).fill()
            (name as NSString).draw(at: NSPoint(x: pt.x + 5, y: pt.y - 5), withAttributes: labelAttrs)
        }

        if let lp = livePoint() {
            let pt = place(lp, in: inset)
            // the creature's actual position: green, pulsing
            let phase = 0.5 + 0.5 * sin(Date().timeIntervalSince1970 * 4)
            NSColor(calibratedRed: 0.35, green: 0.9, blue: 0.55, alpha: 0.35 + 0.3 * phase).setFill()
            NSBezierPath(ovalIn: NSRect(x: pt.x - 8, y: pt.y - 8, width: 16, height: 16)).fill()
            NSColor(calibratedRed: 0.35, green: 0.95, blue: 0.55, alpha: 1).setFill()
            NSBezierPath(ovalIn: NSRect(x: pt.x - 4, y: pt.y - 4, width: 8, height: 8)).fill()
        }

        let cpt = place([Double(cursor.x), Double(cursor.y)], in: inset)
        NSColor.white.setStroke()
        let ring = NSBezierPath(ovalIn: NSRect(x: cpt.x - 7, y: cpt.y - 7, width: 14, height: 14))
        ring.lineWidth = 2
        ring.stroke()
        NSColor(calibratedRed: 0.99, green: 0.9, blue: 0.6, alpha: 1).setFill()
        NSBezierPath(ovalIn: NSRect(x: cpt.x - 3.5, y: cpt.y - 3.5, width: 7, height: 7)).fill()
    }

    private func place(_ p: [Double], in rect: NSRect) -> NSPoint {
        NSPoint(x: rect.minX + rect.width * CGFloat(p[0]),
                y: rect.minY + rect.height * CGFloat(p[1]))
    }

    override func mouseDown(with event: NSEvent) { drag(event) }
    override func mouseDragged(with event: NSEvent) { drag(event) }

    private func drag(_ event: NSEvent) {
        let inset = bounds.insetBy(dx: 16, dy: 16)
        let p = convert(event.locationInWindow, from: nil)
        cursor = CGPoint(x: min(max((p.x - inset.minX) / inset.width, 0), 1),
                         y: min(max((p.y - inset.minY) / inset.height, 0), 1))
        needsDisplay = true
        steer()
    }

    /// kNN (k=8) inverse-distance-weighted blend of the neighbors' z vectors.
    private func steer() {
        guard map.method != "ring" else { return }   // phase ring is display-only
        guard Date().timeIntervalSince(lastWrite) > 0.2 else { return }
        lastWrite = Date()
        let cx = Double(cursor.x), cy = Double(cursor.y)
        var dists: [(Double, Int)] = []
        for (i, p) in map.points.enumerated() {
            let d = (p[0] - cx) * (p[0] - cx) + (p[1] - cy) * (p[1] - cy)
            dists.append((d, i))
        }
        dists.sort { $0.0 < $1.0 }
        let k = min(8, dists.count)
        var weights = [Double](repeating: 0, count: k)
        var total = 0.0
        for j in 0..<k {
            let w = 1.0 / (dists[j].0 + 1e-6)
            weights[j] = w
            total += w
        }
        let zdim = map.z[0].count
        var z = [Double](repeating: 0, count: zdim)
        for j in 0..<k {
            let src = map.z[dists[j].1]
            for d in 0..<zdim { z[d] += src[d] * weights[j] / total }
        }
        guard let dir = NCAWeights.weightsDir() else { return }
        let payload: [String: Any]
        if let controls = map.pointControls {
            payload = ["anchor": controls[dists[0].1]]   // islands: send the nearest anchor
        } else {
            payload = ["z": z.map { (($0 * 1000).rounded() / 1000) }]
        }
        guard let data = try? JSONSerialization.data(withJSONObject: payload) else { return }
        try? data.write(to: URL(fileURLWithPath: dir + "/control.json"))
    }
}
