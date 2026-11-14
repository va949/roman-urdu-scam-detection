"""Fine-tune XLM-R (multilingual transformer) with the same template-aware protocol as train.py.

- Same train/val/test split as train.py for the same --seed.
- Early stopping on validation loss; decision threshold tuned on validation.
- Test is scored once per seed. Inference speed is measured on GPU and CPU.

Usage:
    python src/xlmr.py --data data/raw/phishing_dataset_1000.xlsx --seeds 0 1 2 --out models/xlmr
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

import data as data_mod
import robustness
from metrics import full, pick_threshold
from train import split_texts

BF16 = True  # set from --precision in main()


def amp(device):
    """bf16 autocast on CUDA (weights and optimiser stay fp32); a no-op otherwise."""
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda" and BF16)


def batches(enc, labels, idx, bs, shuffle, rng=None):
    idx = rng.permutation(idx) if shuffle else idx
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        yield ({k: v[b] for k, v in enc.items()}, torch.as_tensor(labels[b]))


@torch.no_grad()
def scam_probs(model, enc, idx, bs, device):
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        with amp(device):
            logits = model(**{k: v[b].to(device) for k, v in enc.items()}).logits
        out.append(torch.softmax(logits.float(), -1)[:, 1].cpu())
    return torch.cat(out).numpy()


def mean_loss(model, enc, labels, idx, bs, device, loss_fn):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in batches(enc, labels, idx, bs, shuffle=False):
            with amp(device):
                logits = model(**{k: v.to(device) for k, v in x.items()}).logits
            total += loss_fn(logits.float(), y.to(device)).item() * len(y)
    return total / len(idx)


def speed(model, enc, idx, device, bs=32, n=128):
    """Messages per second for inference only (tokenisation excluded)."""
    idx = idx[:n]
    model.to(device)
    scam_probs(model, enc, idx[:bs], bs, device)  # warm-up
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    scam_probs(model, enc, idx, bs, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return round(len(idx) / (time.perf_counter() - t), 1)


def make_proba(model, tok, max_len, device):
    """predict_proba(list[str]) for arbitrary texts (used by the robustness suite and predict.py)."""
    def predict_proba(texts):
        texts = list(texts)
        enc = dict(tok(texts, truncation=True, max_length=max_len, padding="longest", return_tensors="pt"))
        return scam_probs(model, enc, np.arange(len(texts)), 64, device)
    return predict_proba


def run(df, seed, a, tok, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    # same split (and, with --augment, the same augmented copies) as train.py for this seed
    (otr, ova, ote), (Xtr, ytr), (Xva, yva), (Xte, yte) = split_texts(df, seed, a.augment)
    texts = Xtr + Xva + Xte
    y = np.concatenate([ytr, yva, yte])
    tr = np.arange(len(Xtr)); va = np.arange(len(Xva)) + len(Xtr); te = np.arange(len(Xte)) + len(Xtr) + len(Xva)
    print(f"\n=== seed {seed}: train={len(tr)} val={len(va)} test={len(te)} "
          f"({'augmented x' + str(a.augment) if a.augment else 'no augmentation'}) | templates "
          f"{df.group.iloc[otr].nunique()}/{df.group.iloc[ova].nunique()}/{df.group.iloc[ote].nunique()}")

    enc = dict(tok(texts, truncation=True, max_length=a.max_len, padding="longest", return_tensors="pt"))
    model = AutoModelForSequenceClassification.from_pretrained(a.model_name, num_labels=2).to(device)

    # balanced class weights from TRAIN only, like class_weight="balanced" in the TF-IDF models
    counts = np.bincount(y[tr], minlength=2)
    weights = torch.tensor(len(tr) / (2 * counts), dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * -(-len(tr) // a.batch)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    best, bad, history, t0 = None, 0, [], time.perf_counter()
    for epoch in range(1, a.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, yb in batches(enc, y, tr, a.batch, shuffle=True, rng=rng):
            with amp(device):
                logits = model(**{k: v.to(device) for k, v in x.items()}).logits
            loss = loss_fn(logits.float(), yb.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            train_loss += loss.item() * len(yb)
        train_loss /= len(tr)  # running mean while training (dropout on)
        vl = mean_loss(model, enc, y, va, a.batch * 2, device, loss_fn)
        vf1 = f1_score(y[va], (scam_probs(model, enc, va, a.batch * 2, device) >= 0.5).astype(int),
                       average="macro", zero_division=0)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4), "val_loss": round(vl, 4),
                        "val_macro_f1": round(float(vf1), 4)})
        print(f"  epoch {epoch:>2}  train loss {train_loss:.4f}  val loss {vl:.4f}  val macro-F1@0.5 {vf1:.3f}")
        if best is None or vl < best[0]:
            best, bad = (vl, epoch, copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})), 0
        else:
            bad += 1
            if bad >= a.patience:
                print(f"  early stop (no val-loss improvement for {a.patience} epochs)")
                break
    train_secs = round(time.perf_counter() - t0, 1)
    model.load_state_dict(best[2])

    pv = scam_probs(model, enc, va, a.batch * 2, device)
    thr = pick_threshold(y[va], pv)
    result = {
        "seed": seed, "best_epoch": best[1], "threshold": round(thr, 3), "train_seconds": train_secs,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else None,
        "val": full(y[va], pv, thr), "test": None, "history": history,
    }
    if a.no_test:  # choosing settings: never look at test
        print(f"  best epoch {best[1]}, threshold {thr:.3f} | test not scored (--no-test)  ({train_secs}s)")
        return model, enc, te, thr, result
    pt = scam_probs(model, enc, te, a.batch * 2, device)  # the one look at test
    rob = robustness.evaluate(make_proba(model, tok, a.max_len, device), thr, Xte, yte, seed, clean_probs=pt)
    t = result["test"] = rob["clean"]
    result["robustness"] = rob
    print(f"  best epoch {best[1]}, threshold {thr:.3f} | TEST macro-F1 {t['macro_f1']:.3f}  "
          f"accuracy {t['accuracy']:.3f}  recall {t['scam_recall']:.3f}  false alarms {t['false_alarm_rate']:.3f}  "
          f"ROC-AUC {t['roc_auc']:.3f}  | probes macro-F1 {rob['probes']['macro_f1']:.3f}  ({train_secs}s)")
    return model, enc, te, thr, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--text-col"); ap.add_argument("--label-col"); ap.add_argument("--group-col")
    ap.add_argument("--model-name", default="FacebookAI/xlm-roberta-base")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)  # chosen on validation curves, see README
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--out", default="models/xlmr", help="model of the FIRST seed is saved here")
    ap.add_argument("--precision", choices=["bf16", "fp32"], default="fp32",
                    help="fp32 was the most stable on validation; bf16 is ~2x faster")
    ap.add_argument("--no-test", action="store_true",
                    help="score validation only (for choosing settings); nothing is saved")
    ap.add_argument("--augment", type=int, default=0,
                    help="perturbed copies per train message, SEEN families only (0 = off)")
    ap.add_argument("--results", default=None, help="default: results/xlmr[_augK].json")
    a = ap.parse_args()
    global BF16
    BF16 = a.precision == "bf16"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'} | model: {a.model_name}")
    df = data_mod.load(a.data, a.text_col, a.label_col, a.group_col)
    tok = AutoTokenizer.from_pretrained(a.model_name)
    lengths = [len(ids) for ids in tok(df["text"].tolist())["input_ids"]]
    print(f"tokens/message: median {int(np.median(lengths))}, max {max(lengths)} (truncating at {a.max_len})")

    results = []
    for i, seed in enumerate(a.seeds):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model, enc, te, thr, r = run(df, seed, a, tok, device)
        if i == 0 and not a.no_test:
            r["msgs_per_sec"] = {"gpu": speed(model, enc, te, device) if device.type == "cuda" else None,
                                 "cpu": speed(model, enc, te, torch.device("cpu"))}
            print(f"  inference speed (msgs/s): {r['msgs_per_sec']}")
            Path(a.out).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(a.out); tok.save_pretrained(a.out)
            (Path(a.out) / "threshold.json").write_text(json.dumps({"threshold": thr, "seed": seed, "max_len": a.max_len}))
            print(f"  saved model to {a.out}")
        results.append(r)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if a.no_test:
        summary = {k: (round(float(np.mean([r["val"][k] for r in results])), 4),
                       round(float(np.std([r["val"][k] for r in results])), 4))
                   for k in ("macro_f1", "accuracy", "scam_recall", "false_alarm_rate")}
        print(f"\nVAL over seeds {a.seeds} (mean, std): {summary}")
        return
    print(f"\nXLM-R{' + augmentation x' + str(a.augment) if a.augment else ''}, mean over seeds {a.seeds}:")
    print(robustness.table([r["robustness"] for r in results]))
    out = Path(a.results or f"results/xlmr{'_aug' + str(a.augment) if a.augment else ''}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": a.model_name, "args": vars(a),
                               "summary": robustness.summarize([r["robustness"] for r in results]),
                               "runs": results}, indent=2))
    print(f"saved {out}")


def load_predictor(path):
    """For predict.py: returns (predict_proba(texts) -> np.ndarray, threshold)."""
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    meta = json.loads((Path(path) / "threshold.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return make_proba(model, tok, meta["max_len"], device), meta["threshold"]


if __name__ == "__main__":
    main()
