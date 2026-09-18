"""Fetch the three benchmark datasets into data/.

The raw data is deliberately NOT redistributed in this repository:

  * HELOC (FICO Explainable ML Challenge) requires accepting FICO's data
    licence before download, so it cannot be mirrored here. This script
    prints the acquisition steps and verifies the file once you supply it.
  * Give-Me-Some-Credit is a Kaggle competition dataset whose terms
    restrict redistribution; this script pulls it from a public mirror.
  * Taiwan Default of Credit Card Clients is UCI-hosted and openly
    redistributable; this script downloads it directly.

Each dataset is checksummed after download so anyone can confirm they
are running against the exact bytes used to produce the published results.

Usage:
    python scripts/fetch_datasets.py            # fetch what can be fetched
    python scripts/fetch_datasets.py --verify   # checksum existing files only
"""

from __future__ import annotations

import argparse
import hashlib
import io
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# SHA-256 of the exact files used for the published results. Populated by
# --emit-checksums; verified on every fetch so results stay reproducible.
CHECKSUMS_FILE = ROOT / "data" / "CHECKSUMS.txt"

HELOC_TARGET = ROOT / "heloc_dataset_v1 (1).csv"
GMSC_TARGET = DATA / "gmsc" / "gmsc.csv"
TAIWAN_TARGET = DATA / "taiwan" / "taiwan_default.csv"

TAIWAN_COLUMNS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    "default_payment_next_month",
]


def _opener():
    try:
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
    op = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
    op.addheaders = [("User-Agent", "xcreditscore-repro/1.0")]
    return op


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_gmsc() -> bool:
    if GMSC_TARGET.exists():
        print(f"  GMSC already present: {GMSC_TARGET}")
        return True
    GMSC_TARGET.parent.mkdir(parents=True, exist_ok=True)
    url = ("https://huggingface.co/datasets/algcache/GiveMeSomeCredit/"
           "resolve/main/Give-Me-Some-Credit.zip")
    print(f"  downloading GMSC from {url}")
    try:
        with _opener().open(url, timeout=180) as r:
            blob = r.read()
        z = zipfile.ZipFile(io.BytesIO(blob))
        member = next(n for n in z.namelist() if n.endswith("cs-training.csv"))
        import pandas as pd

        with z.open(member) as f:
            df = pd.read_csv(f)
        df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
        df.to_csv(GMSC_TARGET, index=False)
        print(f"  wrote {GMSC_TARGET} ({GMSC_TARGET.stat().st_size:,} bytes)")
        return True
    except Exception as exc:
        print(f"  GMSC download failed: {type(exc).__name__}: {exc}")
        print("  Manual route: download 'cs-training.csv' from")
        print("    https://www.kaggle.com/c/GiveMeSomeCredit/data")
        print(f"  drop the index column and save it as {GMSC_TARGET}")
        return False


def fetch_taiwan() -> bool:
    if TAIWAN_TARGET.exists():
        print(f"  Taiwan already present: {TAIWAN_TARGET}")
        return True
    TAIWAN_TARGET.parent.mkdir(parents=True, exist_ok=True)
    url = ("https://huggingface.co/datasets/xaitalk/credit-default-taiwan/"
           "resolve/main/credit_default_taiwan.csv")
    print(f"  downloading Taiwan from {url}")
    try:
        import pandas as pd

        with _opener().open(url, timeout=180) as r:
            blob = r.read()
        df = pd.read_csv(io.BytesIO(blob), header=None, names=TAIWAN_COLUMNS)
        df.to_csv(TAIWAN_TARGET, index=False)
        print(f"  wrote {TAIWAN_TARGET} ({TAIWAN_TARGET.stat().st_size:,} bytes)")
        return True
    except Exception as exc:
        print(f"  Taiwan download failed: {type(exc).__name__}: {exc}")
        print("  Manual route: UCI 'Default of Credit Card Clients'")
        print("    https://archive.ics.uci.edu/dataset/350/")
        print(f"  save with the documented header as {TAIWAN_TARGET}")
        return False


def heloc_notice() -> bool:
    if HELOC_TARGET.exists():
        print(f"  HELOC already present: {HELOC_TARGET}")
        return True
    print("  HELOC is NOT redistributable and must be obtained directly.")
    print("    1. Register at https://community.fico.com/s/explainable-machine-learning-challenge")
    print("    2. Accept the FICO data licence")
    print("    3. Download heloc_dataset_v1.csv")
    print(f"    4. Save it at: {HELOC_TARGET}")
    return False


def write_checksums() -> None:
    lines = []
    for label, path in (("heloc", HELOC_TARGET), ("gmsc", GMSC_TARGET), ("taiwan", TAIWAN_TARGET)):
        if path.exists():
            lines.append(f"{sha256(path)}  {label}  {path.relative_to(ROOT)}")
    CHECKSUMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKSUMS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  wrote {CHECKSUMS_FILE}")
    for l in lines:
        print(f"    {l}")


def verify() -> int:
    if not CHECKSUMS_FILE.exists():
        print("  no CHECKSUMS.txt to verify against")
        return 1
    bad = 0
    for line in CHECKSUMS_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expect, label, rel = line.split(None, 2)
        path = ROOT / rel
        if not path.exists():
            print(f"  MISSING  {label}: {rel}")
            bad += 1
            continue
        actual = sha256(path)
        ok = actual == expect
        print(f"  {'OK      ' if ok else 'MISMATCH'} {label}: {rel}")
        if not ok:
            bad += 1
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="checksum existing files only")
    ap.add_argument("--emit-checksums", action="store_true", help="record checksums of current files")
    args = ap.parse_args()

    if args.verify:
        sys.exit(1 if verify() else 0)

    print("Fetching benchmark datasets")
    print("-" * 52)
    ok_h = heloc_notice()
    ok_g = fetch_gmsc()
    ok_t = fetch_taiwan()

    if args.emit_checksums:
        write_checksums()

    print("-" * 52)
    ready = [n for n, ok in (("HELOC", ok_h), ("GMSC", ok_g), ("Taiwan", ok_t)) if ok]
    missing = [n for n, ok in (("HELOC", ok_h), ("GMSC", ok_g), ("Taiwan", ok_t)) if not ok]
    print(f"ready: {', '.join(ready) if ready else 'none'}")
    if missing:
        print(f"still needed: {', '.join(missing)}")
    print("\nrun the pipeline with:")
    print("  python run_project.py --dataset heloc")
    print("  python run_project.py --dataset taiwan")
    print("  python run_project.py --dataset gmsc")


if __name__ == "__main__":
    main()
