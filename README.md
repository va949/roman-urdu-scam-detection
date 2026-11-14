# Roman Urdu Scam-SMS Benchmark

Can models catch **new** scam SMS in Roman Urdu/English, or do they just memorise old templates?

People in Pakistan receive constant fraud SMS: fake BISP/Ehsaas payouts, "your JazzCash account is blocked", prize scams. These messages are heavily templated. Evaluating with a random train/test split leaks near-duplicates across the split and inflates scores toward 99–100%. This project evaluates every model on **message templates it has never seen**.

**Status:** work in progress. TF-IDF and fine-tuned XLM-R results on the Zenodo dataset, plus a robustness suite, are below. Short version: both models score 90–95% on held-out templates but only ~60–68% macro-F1 on hand-written messages in other styles; the dataset, not the model, is the limit. The rest of the model ladder is in [PLAN.md](PLAN.md).

## Results so far

Dataset: 1,000 messages (405 scam, 595 legit), 16 message types. The file has **no template IDs**, so templates are recovered by masking numbers/links and grouping messages whose word 3-gram Jaccard similarity is ≥ 0.6 (`src/templates.py`), giving 68 templates. No template mixes scam and legit messages.

### 1. Random splits make the task look solved

TF-IDF (char 2–5 + word 1–2 n-grams), 5-fold cross-validation, mean +/- std across folds:

| Model | Split | Macro-F1 | Scam recall | False-alarm rate |
|---|---|---|---|---|
| LogReg | random | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.000 +/- 0.000 |
| LogReg | **unseen templates** | **0.931 +/- 0.106** | 0.964 +/- 0.072 | 0.099 +/- 0.198 |
| LogReg | unseen message types | 0.983 +/- 0.035 | 0.952 +/- 0.096 | 0.000 +/- 0.000 |
| LinearSVC | random | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.000 +/- 0.000 |
| LinearSVC | **unseen templates** | **0.950 +/- 0.072** | 0.964 +/- 0.072 | 0.068 +/- 0.136 |
| LinearSVC | unseen message types | 0.976 +/- 0.049 | 0.935 +/- 0.130 | 0.000 +/- 0.000 |

A random split scores a perfect 1.000 because near-copies of every test message are in the training set. Holding out whole templates drops macro-F1 to 0.93–0.95, with a large spread between folds. The "unseen message types" split (whole `Source_Type` held out) has only 16 groups, so each fold depends heavily on which types land together; don't read its higher score as "easier".

### 2. XLM-R vs TF-IDF on held-out templates

Template-aware train/val/test holdout (~70/15/15 by template, both classes in every part), seeds 0–2, identical splits for both models. Hyperparameters, early stopping and the decision threshold are chosen on validation; test is scored once per seed. Mean +/- std over the 3 seeds:

| Model | Params | Macro-F1 | Scam recall | False-alarm rate | ROC-AUC | Train time | Inference (msgs/s) |
|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg | ~5.5k features | 0.901 +/- 0.036 | 0.917 +/- 0.076 | 0.105 +/- 0.102 | 0.994 +/- 0.005 | seconds (CPU) | not measured yet |
| **XLM-R base** (fine-tuned) | 278M | **0.950 +/- 0.052** | **1.000 +/- 0.000** | **0.082 +/- 0.087** | **1.000 +/- 0.000** | 40–110 s (RTX 4060, 5.2 GB VRAM) | 613 GPU / 34 CPU |

Per seed (test macro-F1): TF-IDF 0.914 / 0.852 / 0.937; XLM-R 0.878 / 1.000 / 0.972.

What this says:
- **XLM-R ranks every test scam above every legit message on all three seeds (ROC-AUC 1.000)**, and never misses a scam. TF-IDF ranks almost as well (0.994) but misses 8% of scams.
- **Nearly all remaining errors come from the decision threshold, not the model.** XLM-R seed 0 is perfectly separable yet raises 18 false alarms (20%) at the threshold validation chose; TF-IDF seed 1 has ROC-AUC 1.000 yet macro-F1 0.852. Validation is saturated (macro-F1 1.000 at almost every threshold), so it cannot place the threshold for unseen templates. Threshold choice is the bottleneck on this data.
- On this dataset, separating scam from legit on unseen templates is close to solved in ranking terms. The harder, more realistic test needs genuinely new messages (PLAN.md, Week 2).

### 3. Warning sign: XLM-R's perfect score does not survive rewording

A quick sanity probe: 10 short messages written by hand in a different style from the dataset (6 scam, 4 legit), scored by the seed-0 models. **This is anecdotal; the probes were not collected from real inboxes.**

| Probe (true label) | XLM-R p(scam) | TF-IDF p(scam) |
|---|---|---|
| "Aap ka JazzCash account block ho gaya hai, foran verify karein" (scam) | **0.00** ✗ | 0.63 |
| same + a fake link (scam) | **0.00** ✗ | 0.62 |
| same + "…warna account band ho jayega" + link (scam) | 1.00 | 0.79 |
| English "Easypaisa account will be suspended… share the 4-digit code" (scam) | **0.00** ✗ | 0.27 ✗ |
| prize, BISP payout scams (scam) | 1.00, 1.00 | 0.73, 0.72 |
| salary credit, real OTP, bill reminder, chat (legit) | 0.00–0.02 | 0.12–0.33 |
| **correct** | **7 / 10** | **9 / 10** |

XLM-R fits the dataset's phrasing (e.g. "Apka… acount… block ho gaya h… **warna**") so tightly that dropping the threat clause flips a confident 1.00 into a confident 0.00. TF-IDF is less confident but more robust here. Section 4 replaces this anecdote with a systematic robustness suite.

### 4. Robustness suite and augmentation

**Why the dataset is easy:** it has strong surface shortcuts. No scam contains "aap" (44% of legit messages do) or "your" (34% of legit), every scam spells "apka", phone numbers appear in 56% of scams and 0% of legit, and **there are no English scams at all**: every English message is a legit OTP or salary SMS.

**The suite** (`src/perturb.py`, `src/robustness.py`) scores each model on:
- the test set rewritten by one label-preserving perturbation at a time. **Seen** families (also used for augmentation): `spelling` (apka ↔ aap ka, hai/h/ha, krein/karein, …), `drop_threat` (remove the "warna…" clause), `swap_entity` (banks, wallets, programs, including names absent from the dataset; new phone numbers), `currency` (PKR ↔ Rs ↔ rupay, 50,000 ↔ 50000). **Unseen** families (never used in training): `char_tricks` (J@zzCash, 0TP, split or doubled letters) and `casing_punct` (all lowercase or caps, no punctuation).
- **40 hand-written probes** (`data/probes/handwritten_v1.csv`, 20 scam, 20 legit): spaced "aap" scams, English scams, scam types missing from the dataset (SIM block, parcel fee, job, loan), and legit messages that start with "Apka". Written by hand for this project, not collected from real inboxes.

**Augmentation** (`--augment 2`): two perturbed copies of every training message (and one of every validation message), using only the seen families.

Macro-F1, mean +/- std over seeds 0–2 (reproduce with `python src/compare.py`):

| Test condition | TF-IDF | TF-IDF + aug | XLM-R | XLM-R + aug |
|---|---|---|---|---|
| clean test | 0.901 +/- 0.036 | 0.912 +/- 0.052 | **0.950 +/- 0.052** | 0.795 +/- 0.076 |
| spelling | **0.950 +/- 0.022** | 0.924 +/- 0.044 | 0.916 +/- 0.046 | 0.809 +/- 0.071 |
| drop_threat | 0.901 +/- 0.036 | 0.912 +/- 0.052 | **0.950 +/- 0.052** | 0.795 +/- 0.076 |
| swap_entity | 0.896 +/- 0.035 | 0.912 +/- 0.052 | **0.955 +/- 0.046** | 0.795 +/- 0.076 |
| currency | 0.863 +/- 0.012 | **0.912 +/- 0.052** | 0.868 +/- 0.030 | 0.795 +/- 0.076 |
| char_tricks (unseen) | 0.901 +/- 0.028 | 0.917 +/- 0.049 | **0.950 +/- 0.031** | 0.800 +/- 0.065 |
| casing_punct (unseen) | 0.897 +/- 0.036 | **0.917 +/- 0.057** | 0.812 +/- 0.017 | 0.791 +/- 0.055 |
| **hand-written probes** | 0.663 +/- 0.030 | **0.678 +/- 0.006** | 0.602 +/- 0.020 | 0.587 +/- 0.044 |

Probe ROC-AUC: TF-IDF 0.718, TF-IDF + aug **0.739**, XLM-R 0.652, XLM-R + aug 0.678.

What this says:
- **The suite exposes the shortcuts.** Swapping "PKR" for "Rs" raises false alarms from 10.5% to 16.9% (TF-IDF) and from 8.2% to 22.1% (XLM-R); spelling swaps raise XLM-R's false alarms to 13.5%; lowercasing and stripping punctuation drops XLM-R's scam recall to 82% (TF-IDF already lowercases, so it barely notices).
- **Augmentation makes TF-IDF invariant to the perturbations, including the two it never saw** (all rows 0.91–0.92), and nudges its probe score up. It is a modest, real gain.
- **Augmentation made XLM-R worse at its chosen thresholds:** false alarms rose to 32–35% on every condition, while ROC-AUC stayed ~0.996. With the spelling shortcut removed, unseen legit templates look more scam-like, and the saturated validation set still cannot place the threshold. Two of three seeds early-stopped after one epoch.
- **Every model fails the probes** (macro-F1 ≤ 0.68). All four miss **all 5 English scams**, because the training data has none, and most of the new scam types (job, loan). No amount of perturbing this dataset fixes that: **the ceiling is the data, not the model.** New, real messages (PLAN.md, Week 2), including English and mixed scams, are the next step.

### Method notes
- **Threshold rule:** best validation macro-F1 over 0.20–0.80; ties go to the middle of the tied range. (An earlier version took the first tie, which always meant 0.20, the most trigger-happy setting.)
- **XLM-R settings** (`lr 2e-5`, fp32, batch 16, max 10 epochs, early stopping on validation loss with patience 3) were picked from validation curves only, with test not scored (`--no-test`). Lowering the learning rate to 1e-5 or using bf16 was less stable.
- **Seed 2 overfits after one epoch:** training loss keeps falling (0.019 → 0.0005) while validation loss jumps from 0.21 to 2.24 and validation collapses to one class. Early stopping keeps the epoch-1 weights. This split leaves training with few, large templates, which the model memorises.
- **Caveats:** 3 seeds; templates are derived, not hand-labelled; val/test tend to receive the smaller templates (each part only takes templates that fit under its share), so train can end up with few templates (10 on seed 1).

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 1. Smoke test on a tiny synthetic sample (no real data needed)
python src/make_sample.py
python src/baseline.py --data data/sample/sample.csv

# 2. Real data: download phishing_dataset_1000.xlsx from
#    https://doi.org/10.5281/zenodo.21810885 into data/raw/, then:
python src/templates.py data/raw/phishing_dataset_1000.xlsx      # template-count sweep over thresholds
python src/baseline.py --data data/raw/phishing_dataset_1000.xlsx
```

The baseline reports each model on three splits:
- **random:** stratified K-fold, the way most prior work evaluated
- **template:** grouped K-fold, so no template appears in both train and test
- **type:** grouped K-fold over message types (`Source_Type`), when the data has them

### Train our own model and run inference
```bash
# template-aware train/val/test (~70/15/15, both classes in every part); tunes on val, then scores
# test + the robustness suite once per seed; writes results/tfidf.json
python src/train.py --data data/raw/phishing_dataset_1000.xlsx --seeds 0 1 2 --out models/tfidf_lr.joblib
python src/train.py --data data/raw/phishing_dataset_1000.xlsx --seeds 0 1 2 --augment 2 --out models/tfidf_lr_aug.joblib
python src/predict.py --model models/tfidf_lr.joblib "Aap ka JazzCash account block ho gaya hai, foran verify karein"
python src/perturb.py "Apka HBL account block ho gaya hai"   # see what each perturbation does
python src/compare.py --metric false_alarm_rate               # comparison table from results/*.json
```

### Fine-tune XLM-R (GPU recommended)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130   # match your CUDA version
pip install -r requirements-dl.txt

# same splits as train.py for the same seeds; downloads xlm-roberta-base (~1.1 GB) on first run
python src/xlmr.py --data data/raw/phishing_dataset_1000.xlsx --seeds 0 1 2 --out models/xlmr
python src/xlmr.py --data data/raw/phishing_dataset_1000.xlsx --seeds 0 1 2 --augment 2 --out models/xlmr_aug
python src/xlmr.py --data data/raw/phishing_dataset_1000.xlsx --seeds 0 1 2 --no-test   # choose settings on val only
python src/predict.py --model models/xlmr "Aap ka JazzCash account block ho gaya hai, foran verify karein"
```

### On-device demo (iOS)
The TF-IDF + augmentation model also runs on an iPhone: a 494 KB JSON model (memorised phone numbers and IDs pruned out), a Swift port that matches scikit-learn to within 1e-9 (checked by `swift test`), and a SwiftUI demo app. GitHub Actions builds an unsigned IPA on a macOS runner, so no Mac is needed. See [mobile/ios/README.md](mobile/ios/README.md).

```bash
python src/export_mobile.py --model models/tfidf_lr_aug.joblib   # re-export after retraining
```

## Layout
```
data/raw/           downloaded datasets (git-ignored)
data/sample/        synthetic smoke-test data
data/probes/        hand-written probe messages (synthetic)
src/data.py         loading, column detection, derived templates
src/templates.py    template recovery by masked-shingle similarity
src/splits.py       random, template-aware and type-aware splits
src/baseline.py     TF-IDF baselines + cross-validated evaluation
src/train.py        train/val/test training of the TF-IDF model (+ --augment)
src/xlmr.py         XLM-R fine-tuning with the same protocol (+ --augment)
src/metrics.py      metrics and the validation threshold rule
src/perturb.py      label-preserving perturbations (robustness tests + augmentation)
src/robustness.py   scores a model on perturbed test sets and the probes
src/compare.py      comparison tables from results/*.json
src/export_mobile.py  exports the TF-IDF model to JSON for the iOS app (+ parity cases)
mobile/ios/         Swift classifier package, SwiftUI demo app, XcodeGen spec
.github/workflows/  ios-demo.yml: parity tests + unsigned IPA on a macOS runner
src/predict.py      inference with a saved model (TF-IDF .joblib or XLM-R folder)
src/make_sample.py  synthetic sample generator
```

## Data and credit
Uses the *Roman Urdu Phishing SMS Dataset for Smishing Detection Research* by Ghufran Siddiq (Zenodo, 2026, DOI [10.5281/zenodo.21810885](https://doi.org/10.5281/zenodo.21810885)), licensed CC BY 4.0. The dataset is not redistributed here; download it from Zenodo.
