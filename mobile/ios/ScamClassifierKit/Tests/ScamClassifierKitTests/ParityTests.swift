import XCTest
@testable import ScamClassifierKit

/// The Swift classifier must reproduce scikit-learn's probabilities (exported by
/// src/export_mobile.py into parity_cases.json) to within 1e-9 on every case.
final class ParityTests: XCTestCase {
    private struct ParityFile: Decodable {
        struct Case: Decodable { let text: String; let proba: Double }
        let threshold: Double
        let cases: [Case]
    }

    private func loadCases() throws -> ParityFile {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "parity_cases", withExtension: "json"))
        return try JSONDecoder().decode(ParityFile.self, from: Data(contentsOf: url))
    }

    func testMatchesScikitLearn() throws {
        let parity = try loadCases()
        let classifier = try ScamClassifier()
        XCTAssertEqual(classifier.threshold, parity.threshold, accuracy: 1e-12)
        XCTAssertGreaterThan(parity.cases.count, 100)

        var worst = 0.0
        var worstText = ""
        for c in parity.cases {
            let diff = abs(classifier.probability(of: c.text) - c.proba)
            if diff > worst { worst = diff; worstText = c.text }
        }
        print("parity: \(parity.cases.count) cases, max |swift - sklearn| = \(worst)")
        XCTAssertLessThan(worst, 1e-9, "worst case: \(worstText.debugDescription)")
    }

    func testVerdictsMatchThreshold() throws {
        let parity = try loadCases()
        let classifier = try ScamClassifier()
        for c in parity.cases {
            XCTAssertEqual(classifier.predict(c.text).isScam, c.proba >= parity.threshold, c.text)
        }
    }

    func testTokenisationEdgeCases() {
        let s = { (t: String) in Array(t.unicodeScalars) }
        // a word shorter than n is emitted once, then n stops growing
        XCTAssertEqual(ScamClassifier.charWbNgrams(s("a"), 2, 5), [" a", "a ", " a "])
        // tokens need >= 2 word characters; punctuation splits tokens
        XCTAssertEqual(ScamClassifier.wordNgrams(s("a ok, j@zz"), 1, 2), ["ok", "zz", "ok zz"])
        XCTAssertEqual(ScamClassifier.splitOnWhitespace(s("  x \t y\n")).count, 2)
    }

    func testLatency() throws {
        let classifier = try ScamClassifier()
        let text = "Apka HBL account block ho gaya hai. Abhi OTP bataen warna account band ho jaye ga."
        let start = Date()
        for _ in 0..<1_000 { _ = classifier.probability(of: text) }
        let ms = Date().timeIntervalSince(start) * 1000 / 1_000
        print(String(format: "latency: %.3f ms per message (1000 runs)", ms))
        XCTAssertLessThan(ms, 50)
    }
}
