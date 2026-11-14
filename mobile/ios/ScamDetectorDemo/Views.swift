import SwiftUI
import ScamClassifierKit

// MARK: - Inbox

struct InboxView: View {
    @EnvironmentObject private var store: ClassifierStore

    var body: some View {
        NavigationStack {
            List {
                if let error = store.loadError {
                    Text(error).foregroundStyle(.red)
                }
                Section {
                    ForEach(DemoInbox.messages) { message in
                        NavigationLink(value: message) { MessageRow(message: message) }
                    }
                } footer: {
                    Text("Every verdict is computed on this phone. No message leaves the device.")
                }
            }
            .navigationTitle("Inbox")
            .navigationDestination(for: DemoMessage.self) { MessageDetailView(message: $0) }
        }
    }
}

struct MessageRow: View {
    @EnvironmentObject private var store: ClassifierStore
    let message: DemoMessage

    var body: some View {
        let result = store.classify(message.text)
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(message.sender).font(.headline)
                Spacer()
                Text(message.time).font(.caption).foregroundStyle(.secondary)
            }
            Text(message.text).font(.subheadline).foregroundStyle(.secondary).lineLimit(2)
            if let result {
                VerdictBadge(prediction: result.prediction)
            }
        }
        .padding(.vertical, 2)
    }
}

struct MessageDetailView: View {
    @EnvironmentObject private var store: ClassifierStore
    let message: DemoMessage

    var body: some View {
        let result = store.classify(message.text)
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text(message.sender).font(.title2.bold())
                Text(message.text)
                    .padding()
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 16))
                if let result {
                    ResultPanel(prediction: result.prediction, milliseconds: result.milliseconds)
                    let correct = result.prediction.isScam == message.expectedScam
                    let truth: String = message.expectedScam ? "a scam" : "legitimate"
                    let note: String = correct ? "Model verdict matches the true label"
                                               : "Model is wrong here: this message is " + truth
                    Label(note, systemImage: correct ? "checkmark.circle" : "exclamationmark.triangle")
                        .font(.footnote)
                        .foregroundStyle(correct ? .green : .orange)
                }
            }
            .padding()
        }
        .navigationBarTitleDisplayMode(.inline)
    }
}

// MARK: - Check a message

struct CheckView: View {
    @EnvironmentObject private var store: ClassifierStore
    @State private var text = ""

    var body: some View {
        NavigationStack {
            Form {
                Section("Message") {
                    TextEditor(text: $text)
                        .frame(minHeight: 120)
                        .autocorrectionDisabled()
                        .textInputAutocapitalization(.never)
                }
                if !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, let result = store.classify(text) {
                    Section("Verdict") {
                        ResultPanel(prediction: result.prediction, milliseconds: result.milliseconds)
                    }
                }
                Section("Try an example") {
                    ForEach(DemoInbox.samples, id: \.self) { sample in
                        Button(sample) { text = sample }
                            .font(.footnote)
                    }
                }
            }
            .navigationTitle("Check a message")
        }
    }
}

// MARK: - About

struct AboutView: View {
    @EnvironmentObject private var store: ClassifierStore

    var body: some View {
        NavigationStack {
            List {
                Section("Model") {
                    row("Type", "TF-IDF + logistic regression")
                    row("Features", store.classifier.map { "\($0.featureCount)" } ?? "-")
                    row("Model file", ByteCountFormatter.string(fromByteCount: Int64(store.modelSizeBytes), countStyle: .file))
                    row("Load time", String(format: "%.0f ms", store.loadMilliseconds))
                    row("Scam threshold", store.classifier.map { String(format: "%.3f", $0.threshold) } ?? "-")
                    row("Runs", "Fully on-device, offline")
                }
                Section("Limitations") {
                    Text("Trained on 1,000 synthetic-style Roman Urdu/English SMS. On hand-written messages in other styles it misses about 40% of scams and flags about 23% of legitimate messages, and it has never seen English scams. This is a demo of on-device deployment, not a production filter.")
                        .font(.footnote)
                }
                Section("Credit") {
                    Text("Training data: Roman Urdu Phishing SMS Dataset for Smishing Detection Research, Ghufran Siddiq (Zenodo, 2026), doi:10.5281/zenodo.21810885, CC BY 4.0.")
                        .font(.footnote)
                }
            }
            .navigationTitle("About")
        }
    }

    private func row(_ title: String, _ value: String) -> some View {
        HStack { Text(title); Spacer(); Text(value).foregroundStyle(.secondary) }
    }
}

// MARK: - Shared pieces

struct VerdictBadge: View {
    let prediction: ScamPrediction

    var body: some View {
        Label(title, systemImage: prediction.isScam ? "exclamationmark.shield.fill" : "checkmark.shield.fill")
            .font(.caption.bold())
            .padding(.horizontal, 8).padding(.vertical, 3)
            .background((prediction.isScam ? Color.red : Color.green).opacity(0.15), in: Capsule())
            .foregroundStyle(prediction.isScam ? .red : .green)
    }

    private var title: String {
        let percent = String(format: "(%.0f%%)", prediction.probability * 100)
        return (prediction.isScam ? "Likely scam " : "Looks safe ") + percent
    }
}

struct ResultPanel: View {
    let prediction: ScamPrediction
    let milliseconds: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            VerdictBadge(prediction: prediction)
            ProgressView(value: prediction.probability) {
                Text("Scam probability")
            } currentValueLabel: {
                Text(String(format: "%.1f%% (threshold %.1f%%)", prediction.probability * 100, prediction.threshold * 100))
            }
            .tint(prediction.isScam ? .red : .green)
            Text(String(format: "Computed on-device in %.2f ms", milliseconds))
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}
