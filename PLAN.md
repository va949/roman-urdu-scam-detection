# Roman Urdu Scam-SMS Benchmark: Plan

**Goal:** an honest benchmark of how well models detect Roman Urdu/English scam SMS **on scam templates they have never seen**, including the first fine-tuned small LLM for this task.

**Hardware:** RTX 4060 Laptop (8 GB VRAM). Enough for TF-IDF, XLM-R base, and 4-bit QLoRA fine-tuning of a ~4B model with Unsloth.

## What already exists (and what's left)

| Already done | By whom | What's still open |
|---|---|---|
| 1,000-message Roman Urdu/English phishing set with template IDs, CC BY 4.0 | Siddiq, Zenodo 2026 ([DOI 10.5281/zenodo.21810885](https://doi.org/10.5281/zenodo.21810885)) | Small; one source |
| Template-aware split + text-normalization study with classical ML | Same author (companion paper) | No transformers or LLMs |
| SVM / Naive Bayes on 1,050 hand-built fraud SMS (~99% acc.) | Ayaz, Nizamani & Chandio, 2024 | Dataset not public; random splits likely inflate results |
| Many student "scam or not" apps | GitHub | No rigorous evaluation |

**Our contribution:**
1. **New labelled test data** from different sources/time, so we test on genuinely new scam templates.
2. **Model ladder** under the same template-aware protocol: TF-IDF → XLM-R → prompted LLM → **QLoRA-fine-tuned small LLM**.
3. **Robustness tests:** spelling variation, spacing and character tricks, and number or link swaps that scammers use.
4. **Error analysis** of which scam types each model misses, and its false-alarm rate on real bank and telecom messages.

## Finding #0 (14 Nov 2026): the Zenodo file has no template IDs and is heavily templated
- Columns: `ID, Raw_text, Labels, Source_Type, Normalized_text`. No template column, despite the description.
- Recovered templates by masking numbers and links, then grouping by word-shingle Jaccard similarity (`src/templates.py`):
  16 clusters at 0.4 (≈ the 16 `Source_Type`s), 68 at 0.6, 184 at 0.7.
- Several categories are a **single template**: Bank OTP (143 messages), Wallet Transfer (143), ATM Alert (55), Utility Reminder (46).
- ⇒ Random-split scores will be near 100% and meaningless. Report three splits:
  **random** (comparison with prior work), **template** (derived, threshold 0.6), **type** (whole `Source_Type` held out, the honest test).
- ⇒ Newly collected real messages (Week 2) are essential, not optional.

## Milestones (about 5 weeks of evenings)

### Week 1: Data foundation
- [ ] Download the Zenodo XLSX (`data/raw/`), inspect its columns, write `data/DATA_CARD.md`.
- [ ] Run `src/baseline.py` and compare against the companion paper's numbers.
- [ ] Record the **random split vs template split** gap. This is your first finding.

### Week 2: New data (the part nobody else has)
- [ ] Collect 300–500 new messages: your own and family inbox (with consent), public reports (PTA/FIA advisories, news screenshots), and genuine messages from banks, JazzCash/Easypaisa and telecoms.
- [ ] **Redact all personal data** (names, numbers, CNICs, live links → placeholders).
- [ ] Write `data/LABELING_GUIDE.md`. Label each message, assign a template ID, and have a second person label a 100-message subset to measure agreement (Cohen's κ).

### Week 3: Baselines and our own model (no LLM)
- [ ] TF-IDF (character + word n-grams) with Logistic Regression and SVM. Already scaffolded (`baseline.py`, `train.py`).
- [ ] **Our own model from scratch:** a small character-level CNN in PyTorch (about 1M parameters, trains in minutes on the 4060).
  - Template-aware **train/val/test (70/15/15)**: early stopping and hyperparameter tuning on val only, test reported once.
  - Save checkpoint → `predict.py` inference, with speed measured in messages per second on CPU.
- [ ] XLM-R base fine-tune on the same split.
- [ ] Prompted LLM, zero- and few-shot (local model), on the same test set.

### Week 4: Fine-tune a small LLM
- [ ] QLoRA fine-tune of Gemma 4 E4B (or another ~4B model) with Unsloth on the 4060.
- [ ] Evaluate on held-out templates **and** on your new Week 2 data.
- [ ] Log VRAM, training time and inference speed (shows engineering judgement).

### Week 5: Analysis and write-up
- [ ] Robustness suite: run perturbed versions of the test messages through every model.
- [ ] Results table, confusion matrices, a gallery of each model's errors.
- [ ] Polish the README, write a model card, build a small demo (Gradio). Optionally a short arXiv-style report.

## CV line to aim for
> Built a template-aware benchmark for Roman Urdu scam-SMS detection (1.4k messages, 400+ newly labelled); a QLoRA-fine-tuned 4B LLM reached **X** macro-F1 on unseen scam templates vs **Y** for TF-IDF and **Z** for prompted LLMs.

## Model ladder (all on the same template-aware test set)

| Model | LLM? | Built by us? | Why it's there |
|---|---|---|---|
| TF-IDF + LogReg/SVM | No | Trained | Classical baseline, matches prior work |
| Char-CNN from scratch | No | **Designed and trained** | Shows deep-learning fundamentals; tiny and fast enough for on-device use |
| XLM-R base fine-tune | Encoder | Fine-tuned | Standard multilingual transformer |
| Prompted LLM (zero/few-shot) | Yes | No training | Does a general model get this "for free"? |
| QLoRA-fine-tuned ~4B LLM | Yes | Fine-tuned | Best-case accuracy; costs memory and speed |

The interview story is the trade-off: accuracy on unseen templates vs model size vs speed.

## Rules
- Never publish real phone numbers, CNICs or working scam links.
- Credit the Zenodo dataset (CC BY 4.0) wherever it is used.
- Never report random-split scores as the headline number.
