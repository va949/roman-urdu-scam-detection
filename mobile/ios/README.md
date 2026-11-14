# iOS on-device demo

A small SwiftUI app that shows the scam classifier running **entirely on an iPhone**, offline. It is a deployment showcase, not a Messages filter: it classifies a built-in synthetic inbox and anything you type.

- **Inbox:** 12 synthetic messages, each with an on-device verdict; the detail view says whether the model got it right.
- **Check:** paste or type a message and get a live verdict, scam probability and on-device latency.
- **About:** model size, feature count, load time, threshold, limitations, data credit.

## What runs on the phone

`ScamClassifierKit` (a Swift package) is a hand port of the scikit-learn pipeline: TF-IDF over character 2–5-grams within words plus word 1–2-grams, then logistic regression. The model ships as a 494 KB JSON file (`scam_model.json`, 10,833 features) exported by `src/export_mobile.py` from `models/tfidf_lr_aug.joblib` (TF-IDF + augmentation, seed 0, the most robust model in the benchmark).

**No phone numbers are published.** The export first prunes the 2,066 word terms containing 7+ consecutive digits (phone numbers, OTPs and transaction IDs memorised from single training messages). This changes no verdict on any of the 1,000 dataset messages (largest probability change 0.012).

**Parity:** `swift test` checks that Swift reproduces scikit-learn's probabilities to within 1e-9 on 205 cases (the 40 probes, each probe under all six perturbations, and edge cases such as Urdu script, emoji and empty text). The same algorithm, re-implemented in pure Python inside the export script, matches scikit-learn to 3e-16.

## Build an unsigned IPA on GitHub Actions (no Mac needed)

`.github/workflows/ios-demo.yml` runs on a macOS runner on every push that touches `mobile/ios/` (or manually from the Actions tab, "Run workflow"):

1. `swift test`: Swift vs scikit-learn parity tests; the build stops if they fail
2. `xcodegen generate`: creates the Xcode project from `project.yml`
3. `xcodebuild` for a generic iOS device with code signing disabled
4. zips `Payload/ScamDetectorDemo.app` into `ScamDetectorDemo-unsigned.ipa` and uploads it as a workflow artifact (kept 30 days)

Download it from the run's page, under **Artifacts**. On a private repository, macOS runner minutes count 10x against the free Actions quota; public repositories are free.

## Installing it on an iPhone

An **unsigned** IPA cannot be installed as-is; iOS only runs signed apps. Options:

- **Sideloadly or AltStore** (Windows or Mac): re-sign the IPA with your own Apple ID and install over USB. With a free Apple ID the app expires after 7 days and must be re-signed.
- **A Mac with Xcode:** `brew install xcodegen && xcodegen generate` in `mobile/ios`, open `ScamDetectorDemo.xcodeproj`, pick your personal team under Signing, and run on your phone or the simulator.
- **A paid Apple Developer account** lets the workflow sign the build itself (add a certificate and provisioning profile as repository secrets); not set up here.

## Updating the model

After retraining, re-export and commit the two JSON files; the next push rebuilds the IPA and re-runs the parity tests:

```bash
python src/export_mobile.py --model models/tfidf_lr_aug.joblib
```

## Limitations

The model is trained on 1,000 synthetic-style messages. On hand-written messages in other styles it misses about 40% of scams and flags about 23% of legitimate ones, and it has never seen an English scam (see the main README). Good for showing that on-device deployment is cheap and fast; not good enough to protect real users yet.
