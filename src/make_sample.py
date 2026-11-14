"""Write a tiny SYNTHETIC dataset for smoke-testing the pipeline. Not real messages."""
import csv
import random
from pathlib import Path

random.seed(0)
SCAM = {
    "bisp": "Mubarak ho! Aap ki BISP raqam {amt} Rs mansoor ho gayi hai. Hasil karne ke liye {num} par call karein.",
    "wallet_block": "Aap ka {wallet} account block kar diya gaya hai. Foran verify karein: {link}",
    "prize": "Congratulations! Aap ne {prize} jeet liya hai. Inaam lene ke liye {num} pe rabta karein.",
    "otp": "{wallet} se: aap ka code {code} hai, ye code humare agent ko batayein taake account active rahe.",
}
LEGIT = {
    "otp_real": "{code} is your {wallet} verification code. Kisi ke saath share na karein.",
    "bill": "Aap ka bijli ka bill {amt} Rs {date} tak jama karwa dein. Shukriya.",
    "txn": "{wallet}: Rs {amt} aap ke account mein receive ho gaye. Balance check karne ke liye app kholein.",
    "friend": "Bhai {date} ko shaam 7 bajay milte hain, {place} pe.",
}
FILL = dict(
    amt=lambda: str(random.choice([5000, 9000, 12500, 25000])), num=lambda: "03XX-XXXXXXX",
    link=lambda: "hxxp://example[.]invalid/verify", wallet=lambda: random.choice(["JazzCash", "Easypaisa", "Bank"]),
    prize=lambda: random.choice(["motorcycle", "iPhone", "50,000 Rs"]), code=lambda: str(random.randint(1000, 9999)),
    date=lambda: random.choice(["Monday", "15 tareekh", "kal"]), place=lambda: random.choice(["Liberty", "Mall Road"]),
)


def render(t):
    return t.format(**{k: f() for k, f in FILL.items() if "{" + k + "}" in t})


rows = []
for label, bank in (("phishing", SCAM), ("legitimate", LEGIT)):
    for tid, tpl in bank.items():
        for _ in range(15):
            rows.append({"Raw_Text": render(tpl), "Label": label, "Template_ID": tid})
random.shuffle(rows)
out = Path("data/sample/sample.csv")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["Raw_Text", "Label", "Template_ID"]); w.writeheader(); w.writerows(rows)
print(f"wrote {len(rows)} synthetic rows to {out}")
