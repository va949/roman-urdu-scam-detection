# Roman Urdu Scam-SMS Benchmark

Can models catch **new** scam SMS in Roman Urdu/English, or do they just memorise old templates?

People in Pakistan receive constant fraud SMS: fake BISP/Ehsaas payouts, "your JazzCash account is blocked", prize scams. These messages are heavily templated. Evaluating with a random train/test split leaks near-duplicates across the split and inflates scores toward 99–100%. This project evaluates every model on **scam templates it has never seen**.

**Status:** work in progress. See [PLAN.md](PLAN.md).

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
python src/baseline.py --data data/raw/phishing_dataset_1000.xlsx --show-columns
python src/baseline.py --data data/raw/phishing_dataset_1000.xlsx
```

The baseline reports each model twice:
- **random split:** stratified K-fold, the way most prior work evaluated
- **template split:** grouped K-fold, so no scam template appears in both train and test

The gap between the two is the first result.

### Train our own model and run inference
```bash
# template-aware train/val/test (70/15/15); tunes on val, reports test once
python src/train.py --data data/sample/sample.csv --out models/tfidf_lr.joblib
python src/predict.py --model models/tfidf_lr.joblib "Aap ka JazzCash account block ho gaya hai, foran verify karein"
```

## Layout
```
data/raw/        downloaded datasets (git-ignored)
data/sample/     synthetic smoke-test data
src/data.py      loading + column detection
src/splits.py    random vs template-aware splits
src/baseline.py  TF-IDF baselines + evaluation
src/make_sample.py
```

## Data and credit
Uses the *Roman Urdu Phishing SMS Dataset for Smishing Detection Research* by Ghufran Siddiq (Zenodo, 2026, DOI [10.5281/zenodo.21810885](https://doi.org/10.5281/zenodo.21810885)), licensed CC BY 4.0.
