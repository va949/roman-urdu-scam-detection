"""Label-preserving perturbations of Roman Urdu/English SMS, for robustness tests and augmentation.

The Zenodo set has strong surface shortcuts: no scam contains "aap" or "your" (44% and 34%
of legit messages do), every scam spells "apka", phone numbers appear in 56% of scams and
0% of legit messages. A model can score ~100% by learning the dataset's spelling. These
perturbations change the surface but not the meaning, so a robust model's verdict should
not move.

Families:
- SEEN (used for training augmentation AND tested): spelling, drop_threat, swap_entity, currency
- UNSEEN (tested only, never used in training): char_tricks, casing_punct

Each family is fn(text, rng) -> text, deterministic given rng.
"""
from __future__ import annotations

import random
import re

# ---- spelling: interchangeable Roman Urdu spellings ---------------------------------------
# Every member of a group can replace any other. Multi-word members are matched first.
SPELLING_GROUPS = [
    ["apka", "aap ka", "aapka", "ap ka"],
    ["apke", "aap ke", "aapke", "aap k", "ap ke"],
    ["apki", "aap ki", "aapki", "ap ki"],
    ["hai", "h", "ha", "hy", "he"],
    ["hain", "hn", "han", "hen"],
    ["karein", "krein", "karin", "kren", "krain", "karen", "krin"],
    ["k liye", "ke liye", "k lie", "ke liay", "kay liye"],
    ["account", "acount", "acc", "a/c"],
    ["kar", "kr"], ["krne", "karne"], ["krwane", "karwane"], ["dia", "diya"], ["gya", "gaya"],
    ["inam", "inaam"], ["mein", "me", "main", "mai"], ["par", "pr", "pe"],
    ["bataen", "batayein", "batain"], ["dain", "dein", "den"], ["nahi", "nahin", "nai"],
    ["abhi", "abi"], ["foran", "fauran"], ["rabta", "raabta"], ["ho gaya", "ho gya", "hogaya"],
    ["jaye ga", "jayega", "jaega"],
]
_SPELL_MAP = {m: g for g in SPELLING_GROUPS for m in g}
_SPELL_RX = re.compile(r"(?<![\w/])(" + "|".join(
    re.escape(m) for m in sorted(_SPELL_MAP, key=len, reverse=True)) + r")(?![\w/])", re.I)


def _match_case(src: str, new: str) -> str:
    if src.isupper() and len(src) > 1:
        return new.upper()
    return new[0].upper() + new[1:] if src[0].isupper() else new


def spelling(text: str, rng: random.Random, p: float = 0.7) -> str:
    def sub(m):
        word = m.group(0)
        if rng.random() > p:
            return word
        options = [o for o in _SPELL_MAP[word.lower()] if o != word.lower()]
        return _match_case(word, rng.choice(options))
    return _SPELL_RX.sub(sub, text)


# ---- drop_threat: remove the "warna ..." (or else ...) clause -----------------------------
_THREAT_RX = re.compile(r"\s*\b(warna|nahi to|nahi toh|otherwise)\b[^.!?]*", re.I)


def drop_threat(text: str, rng: random.Random) -> str:
    return re.sub(r"\s+([.!?])", r"\1", _THREAT_RX.sub("", text))


# ---- swap_entity: banks, wallets, programs, utilities, phone numbers -----------------------
# Pools include names that never appear in the dataset.
ENTITY_POOLS = [
    ["HBL", "UBL", "MCB", "Allied Bank", "State Bank", "Meezan Bank", "Bank Alfalah", "Faysal Bank",
     "Askari Bank", "BankIslami", "Standard Chartered"],
    ["JazzCash", "Easypaisa", "NayaPay", "SadaPay", "Upaisa", "Zindigi", "Konnect"],
    ["BISP", "Ehsaas Program", "Jeeto Pakistan", "Benazir Income Support", "Inaam Ghar",
     "PM Relief Fund", "Kamyab Jawan Program", "Sehat Card"],
    ["WASA", "IESCO", "PTCL", "LESCO", "K-Electric", "Sui Gas", "MEPCO", "FESCO", "SNGPL"],
]
_ENTITY_MAP = {e: pool for pool in ENTITY_POOLS for e in pool}
_ENTITY_RX = re.compile(r"\b(" + "|".join(re.escape(e) for e in sorted(_ENTITY_MAP, key=len, reverse=True)) + r")\b")
_PHONE_RX = re.compile(r"\b03\d{8,9}\b")


def swap_entity(text: str, rng: random.Random) -> str:
    text = _ENTITY_RX.sub(lambda m: rng.choice([e for e in _ENTITY_MAP[m.group(0)] if e != m.group(0)]), text)
    return _PHONE_RX.sub(lambda m: "03" + "".join(rng.choice("0123456789") for _ in range(len(m.group(0)) - 2)), text)


# ---- currency: PKR / Rs / rupay, and digit grouping ----------------------------------------
_CUR_RX = re.compile(r"\b(PKR|Rs\.?|rs|rupay|rupees)(?=\W|$)")
_NUM_RX = re.compile(r"\b\d{1,3}(?:,\d{3})+\b|\b\d{4,6}\b")


def currency(text: str, rng: random.Random) -> str:
    text = _CUR_RX.sub(lambda m: rng.choice(["PKR", "Rs", "Rs.", "rupay", "rupees"]), text)

    def regroup(m):
        n = m.group(0).replace(",", "")
        if len(n) > 6:  # leave IDs and long codes alone
            return m.group(0)
        return f"{int(n):,}" if rng.random() < 0.5 else n
    return _NUM_RX.sub(regroup, text)


# ---- UNSEEN: char_tricks (scammers' obfuscation) -------------------------------------------
_LOOKALIKE = {"a": "@", "o": "0", "i": "1", "e": "3", "s": "$", "l": "1"}


def char_tricks(text: str, rng: random.Random, p: float = 0.25) -> str:
    def trick(word):
        if len(word) < 4 or not word.isalpha() or rng.random() > p:
            return word
        kind = rng.choice(["lookalike", "split", "double"])
        if kind == "lookalike":
            idx = [i for i, ch in enumerate(word) if ch.lower() in _LOOKALIKE]
            if not idx:
                return word
            i = rng.choice(idx)
            return word[:i] + _LOOKALIKE[word[i].lower()] + word[i + 1:]
        i = rng.randrange(1, len(word) - 1)
        return word[:i] + (" " if kind == "split" else word[i]) + word[i:]
    return re.sub(r"[A-Za-z]+", lambda m: trick(m.group(0)), text)


# ---- UNSEEN: casing_punct ------------------------------------------------------------------
def casing_punct(text: str, rng: random.Random) -> str:
    text = re.sub(r"(?<!\d)[.,]|[.,](?!\d)|[:!?]", " ", text)  # keep 50,000 and 2.5 intact
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower() if rng.random() < 0.7 else text.upper()


SEEN = {"spelling": spelling, "drop_threat": drop_threat, "swap_entity": swap_entity, "currency": currency}
UNSEEN = {"char_tricks": char_tricks, "casing_punct": casing_punct}
FAMILIES = {**SEEN, **UNSEEN}


def augment(texts, labels, k: int = 2, seed: int = 0, families=SEEN):
    """k perturbed copies per message, each from a random mix of `families` (>= 1 applied).

    Returns (texts, labels, source_index) for the copies only; copies identical to the
    original are dropped.
    """
    rng = random.Random(seed)
    names = list(families)
    out_t, out_y, out_i = [], [], []
    for i, (t, y) in enumerate(zip(texts, labels)):
        seen = {t}
        for _ in range(k):
            for _attempt in range(5):
                chosen = [n for n in names if rng.random() < 0.5] or [rng.choice(names)]
                new = t
                for n in chosen:
                    new = families[n](new, rng)
                if new not in seen:
                    seen.add(new); out_t.append(new); out_y.append(y); out_i.append(i)
                    break
    return out_t, out_y, out_i


if __name__ == "__main__":
    import sys
    rng = random.Random(0)
    demo = sys.argv[1] if len(sys.argv) > 1 else (
        "Apka HBL ka account suspicious activity ki waja se block ho gaya hai. "
        "Abhi OTP bataen 0349287760 par warna account hamesha k liye band ho jaye ga.")
    print("original     :", demo)
    for name, fn in FAMILIES.items():
        print(f"{name:<13}:", fn(demo, rng))
