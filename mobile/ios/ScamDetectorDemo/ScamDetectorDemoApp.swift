import SwiftUI
import ScamClassifierKit

@main
struct ScamDetectorDemoApp: App {
    @StateObject private var store = ClassifierStore()

    var body: some Scene {
        WindowGroup {
            TabView {
                InboxView()
                    .tabItem { Label("Inbox", systemImage: "tray.full") }
                CheckView()
                    .tabItem { Label("Check", systemImage: "text.magnifyingglass") }
                AboutView()
                    .tabItem { Label("About", systemImage: "info.circle") }
            }
            .environmentObject(store)
        }
    }
}

/// Loads the bundled model once and times every prediction.
@MainActor
final class ClassifierStore: ObservableObject {
    @Published private(set) var classifier: ScamClassifier?
    @Published private(set) var loadError: String?
    @Published private(set) var loadMilliseconds: Double = 0
    let modelSizeBytes: Int

    init() {
        let url = ScamClassifier.bundledModelURL
        modelSizeBytes = url.flatMap { try? FileManager.default.attributesOfItem(atPath: $0.path)[.size] as? Int } ?? 0
        let start = DispatchTime.now()
        do {
            classifier = try ScamClassifier()
        } catch {
            loadError = "Could not load the model: \(error)"
        }
        loadMilliseconds = Self.milliseconds(since: start)
    }

    /// Prediction plus how long it took on this device.
    func classify(_ text: String) -> (prediction: ScamPrediction, milliseconds: Double)? {
        guard let classifier else { return nil }
        let start = DispatchTime.now()
        let prediction = classifier.predict(text)
        return (prediction, Self.milliseconds(since: start))
    }

    static func milliseconds(since start: DispatchTime) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000
    }
}
