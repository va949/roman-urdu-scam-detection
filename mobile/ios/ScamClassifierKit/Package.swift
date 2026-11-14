// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "ScamClassifierKit",
    platforms: [.iOS(.v16), .macOS(.v13)],
    products: [
        .library(name: "ScamClassifierKit", targets: ["ScamClassifierKit"]),
    ],
    targets: [
        .target(
            name: "ScamClassifierKit",
            resources: [.copy("Resources/scam_model.json")]
        ),
        .testTarget(
            name: "ScamClassifierKitTests",
            dependencies: ["ScamClassifierKit"],
            resources: [.copy("Resources/parity_cases.json")]
        ),
    ]
)
