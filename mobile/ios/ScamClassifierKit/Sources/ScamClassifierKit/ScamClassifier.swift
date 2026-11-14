import Foundation

/// On-device scam-SMS classifier: TF-IDF (character 2-5-grams within words + word 1-2-grams)
/// followed by logistic regression, exported from scikit-learn by `src/export_mobile.py`.
///
/// Every step mirrors scikit-learn exactly (and `reference_proba` in export_mobile.py), so
/// probabilities match Python to ~1e-15. Text is processed as Unicode scalars, like Python
/// code points, not as Swift grapheme clusters. Keep the two implementations in lock-step;
/// `ParityTests` fails if they drift.
public final class ScamClassifier: @unchecked Sendable {
    public let threshold: Double
    public let featureCount: Int
    public let sourceModel: String

    private let intercept: Double
    private let char: Block
    private let word: Block

    /// URL of the model bundled with this package.
    public static var bundledModelURL: URL? {
        Bundle.module.url(forResource: "scam_model", withExtension: "json")
    }

    /// Loads the bundled model.
    public convenience init() throws {
        guard let url = Self.bundledModelURL else { throw ScamClassifierError.modelNotFound }
        try self.init(contentsOf: url)
    }

    public init(contentsOf url: URL) throws {
        let model = try JSONDecoder().decode(ModelFile.self, from: Data(contentsOf: url))
        guard model.format == "tfidf-logreg-v1" else { throw ScamClassifierError.unsupportedFormat(model.format) }
        threshold = model.threshold
        intercept = model.intercept
        char = try Block(model.char)
        word = try Block(model.word)
        featureCount = model.char.terms.count + model.word.terms.count
        sourceModel = model.meta?.source_model ?? "unknown"
    }

    /// Probability that `text` is a scam, in [0, 1].
    public func probability(of text: String) -> Double {
        let scalars = Array(text.lowercased().unicodeScalars)
        var z = intercept
        z += char.score(Self.charWbNgrams(scalars, char.minN, char.maxN))
        z += word.score(Self.wordNgrams(scalars, word.minN, word.maxN))
        return 1.0 / (1.0 + exp(-z))
    }

    public func predict(_ text: String) -> ScamPrediction {
        let p = probability(of: text)
        return ScamPrediction(probability: p, isScam: p >= threshold, threshold: threshold)
    }

    // MARK: - Feature extraction (mirrors scikit-learn)

    /// scikit-learn `analyzer="char_wb"`: n-grams inside each word padded with one space each side.
    static func charWbNgrams(_ text: [Unicode.Scalar], _ minN: Int, _ maxN: Int) -> [String] {
        var out: [String] = []
        for w in splitOnWhitespace(text) {
            let padded: [Unicode.Scalar] = [" "] + w + [" "]
            for n in minN...maxN {
                var offset = 0
                out.append(string(padded, offset, n))
                while offset + n < padded.count {
                    offset += 1
                    out.append(string(padded, offset, n))
                }
                if offset == 0 { break }  // word shorter than n: counted once, stop growing n
            }
        }
        return out
    }

    /// Tokens are maximal runs of >= 2 word characters (`(?u)\b\w\w+\b`), then word n-grams.
    static func wordNgrams(_ text: [Unicode.Scalar], _ minN: Int, _ maxN: Int) -> [String] {
        var tokens: [String] = []
        var current: [Unicode.Scalar] = []
        for s in text {
            if isWordChar(s) {
                current.append(s)
            } else {
                if current.count >= 2 { tokens.append(string(current, 0, current.count)) }
                current.removeAll(keepingCapacity: true)
            }
        }
        if current.count >= 2 { tokens.append(string(current, 0, current.count)) }

        var out: [String] = []
        for n in minN...maxN where tokens.count >= n {
            for i in 0...(tokens.count - n) {
                out.append(tokens[i..<(i + n)].joined(separator: " "))
            }
        }
        return out
    }

    /// Python's `str.split()`: runs of whitespace separate words; empty words are dropped.
    static func splitOnWhitespace(_ text: [Unicode.Scalar]) -> [[Unicode.Scalar]] {
        var words: [[Unicode.Scalar]] = []
        var current: [Unicode.Scalar] = []
        for s in text {
            if isPythonSpace(s) {
                if !current.isEmpty { words.append(current); current = [] }
            } else {
                current.append(s)
            }
        }
        if !current.isEmpty { words.append(current) }
        return words
    }

    /// Python `str.isspace()`: Unicode White_Space plus the ASCII separators U+001C-U+001F.
    static func isPythonSpace(_ s: Unicode.Scalar) -> Bool {
        s.properties.isWhitespace || (0x1C...0x1F).contains(s.value)
    }

    /// Python regex `\w` on str: `isalnum()` (letters and numbers by category) or underscore.
    static func isWordChar(_ s: Unicode.Scalar) -> Bool {
        if s == "_" { return true }
        switch s.properties.generalCategory {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter,
             .decimalNumber, .letterNumber, .otherNumber:
            return true
        default:
            return s.properties.numericType != nil
        }
    }

    /// Python slice `w[offset:offset+n]` (clamped at the end), as a String.
    static func string(_ s: [Unicode.Scalar], _ offset: Int, _ n: Int) -> String {
        var view = String.UnicodeScalarView()
        view.append(contentsOf: s[offset..<min(offset + n, s.count)])
        return String(view)
    }
}

public struct ScamPrediction: Sendable {
    public let probability: Double
    public let isScam: Bool
    public let threshold: Double
}

public enum ScamClassifierError: Error {
    case modelNotFound
    case unsupportedFormat(String)
    case malformedModel(String)
}

// MARK: - Model file

private struct ModelFile: Decodable {
    struct BlockFile: Decodable {
        let analyzer: String
        let ngram_range: [Int]
        let terms: [String]
        let idf: [Double]
        let coef: [Double]
    }
    struct Meta: Decodable { let source_model: String? }
    let format: String
    let threshold: Double
    let intercept: Double
    let char: BlockFile
    let word: BlockFile
    let meta: Meta?
}

/// One TF-IDF vectoriser plus its slice of the logistic-regression weights.
private struct Block {
    let minN: Int
    let maxN: Int
    let index: [String: Int]
    let idf: [Double]
    let coef: [Double]

    init(_ f: ModelFile.BlockFile) throws {
        guard f.ngram_range.count == 2, f.terms.count == f.idf.count, f.terms.count == f.coef.count else {
            throw ScamClassifierError.malformedModel("block \(f.analyzer) has inconsistent sizes")
        }
        minN = f.ngram_range[0]
        maxN = f.ngram_range[1]
        idf = f.idf
        coef = f.coef
        var index: [String: Int] = [:]
        index.reserveCapacity(f.terms.count)
        for (i, term) in f.terms.enumerated() { index[term] = i }
        self.index = index
    }

    /// Sublinear TF x IDF, L2-normalised within this block, dotted with the weights.
    func score(_ grams: [String]) -> Double {
        var counts: [Int: Int] = [:]
        for g in grams {
            if let i = index[g] { counts[i, default: 0] += 1 }
        }
        var weighted: [(Int, Double)] = []
        weighted.reserveCapacity(counts.count)
        var sumSquares = 0.0
        for (i, c) in counts {
            let v = (1.0 + log(Double(c))) * idf[i]
            weighted.append((i, v))
            sumSquares += v * v
        }
        guard sumSquares > 0 else { return 0 }
        let norm = sumSquares.squareRoot()
        return weighted.reduce(0.0) { $0 + $1.1 / norm * coef[$1.0] }
    }
}
