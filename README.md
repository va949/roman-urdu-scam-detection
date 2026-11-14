# Roman Urdu Scam-SMS Benchmark

Can models catch **new** scam SMS in Roman Urdu/English, or do they just memorise old templates?

People in Pakistan receive constant fraud SMS: fake BISP/Ehsaas payouts, "your JazzCash account is blocked", prize scams. These messages are heavily templated. Evaluating with a random train/test split leaks near-duplicates across the split and inflates scores toward 99–100%. This project evaluates every model on **message templates it has never seen**.

**Status:** work in progress. TF-IDF and fine-tuned XLM-R results on the Zenodo dataset are below; the rest of the model ladder is in [PLAN.md](PLAN.md).

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

XLM-R fits the dataset's phrasing (e.g. "Apka… acount… block ho gaya h… **warna**") so tightly that dropping the threat clause flips a confident 1.00 into a confident 0.00. TF-IDF is less confident but more robust here. Template-held-out scores on this dataset overstate real-world robustness, which is the case for the robustness suite and new data in PLAN.md.

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
# template-aware train/val/test (~70/15/15, both classes in every part); tunes on val, reports test once
python src/train.py --data data/raw/phishing_dataset_1000.xlsx --out models/tfidf_lr.joblib
python src/predict.py --model models/tfidf_lr.joblib "Aap ka JazzCash account block ho gaya hai, foran verify karein"
```

### Fine-tune XLM-R (GPU recommended)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130   # match your CUDA version
pip install -r requirements-dl.txt

# same splits as train.py for the same seeds; downloads xlm-roberta-base (~1.1 GB) on first run
python src/xlmr.py --data data/raw/phishing_dataset_1000.xlsx --seeds 0 1 2 --out models/xlmr
python src/xlmr.py --data data/raw/phishing_dataset_1000.xlsx --seeds 0 1 2 --no-test   # choose settings on val only
python src/predict.py --model models/xlmr "Aap ka JazzCash account block ho gaya hai, foran verify karein"
```

## Layout
```
data/raw/           downloaded datasets (git-ignored)
data/sample/        synthetic smoke-test data
src/data.py         loading, column detection, derived templates
src/templates.py    template recovery by masked-shingle similarity
src/splits.py       random, template-aware and type-aware splits
src/baseline.py     TF-IDF baselines + cross-validated evaluation
src/train.py        train/val/test training of the TF-IDF model; shared threshold + metrics helpers
src/xlmr.py         XLM-R fine-tuning with the same protocol
src/predict.py      inference with a saved model (TF-IDF .joblib or XLM-R folder)
src/make_sample.py  synthetic sample generator
```

## Data and credit
Uses the *Roman Urdu Phishing SMS Dataset for Smishing Detection Research* by Ghufran Siddiq (Zenodo, 2026, DOI [10.5281/zenodo.21810885](https://doi.org/10.5281/zenodo.21810885)), licensed CC BY 4.0. The dataset is not redistributed here; download it from Zenodo.
