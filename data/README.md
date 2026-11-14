# Data

- `raw/`: downloaded datasets. Git-ignored; never commit them.
  - `phishing_dataset_1000.xlsx` from https://doi.org/10.5281/zenodo.21810885 (CC BY 4.0, Ghufran Siddiq, 2026).
- `sample/`: synthetic smoke-test data made by `src/make_sample.py`. Not real messages.
- `probes/handwritten_v1.csv`: 40 hand-written probe messages (20 scam, 20 legit) in styles and scam types the Zenodo set lacks. Synthetic, written for this project, not collected from real inboxes; phone numbers are `03XX-XXXXXXX` placeholders and links use the reserved `.invalid` domain.
- (Week 2) `new/`: your newly collected messages, **redacted** (no real numbers, CNICs, names or working links).
