// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "Bonsai",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "Bonsai",
            path: "Sources/Bonsai",
            exclude: ["AppDelegate.md", "Creature.md", "main.md", "NCAShaders.md",
                      "NCASimulation.md", "NCAWeights.md", "PetView.md", "RenderTest.md",
                      "FusedNCAWeights.md", "FusedNCAShaders.md",
                      "FusedNCASimulation.md", "FusedPetView.md",
                      "FusedRenderTest.md", "StateMapPanel.md"]
        )
    ]
)
