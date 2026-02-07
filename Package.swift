// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "VoxtralFoundation",
    products: [
        .library(
            name: "VoxtralFoundation",
            targets: ["VoxtralC"]
        )
    ],
    targets: [
        .target(
            name: "VoxtralC",
            path: "Sources/VoxtralC",
            publicHeadersPath: "include",
            cSettings: [
                .define("USE_BLAS", .when(platforms: [.macOS, .iOS, .tvOS, .watchOS, .visionOS])),
                .define("VOXTRAL_ENABLE_METAL", .when(platforms: [.macOS, .iOS, .tvOS, .watchOS, .visionOS]))
            ],
            linkerSettings: [
                .linkedFramework("Accelerate", .when(platforms: [.macOS, .iOS, .tvOS, .watchOS, .visionOS]))
            ]
        )
    ]
)
