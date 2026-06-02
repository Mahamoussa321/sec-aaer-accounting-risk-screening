import os
import pandas as pd
from pathlib import Path

BASE = Path("sec_financial_statements/extracted")
out_rows = []

for qdir in sorted(BASE.iterdir()):
    sub_path = qdir / "sub.txt"
    if not sub_path.exists():
        continue

    print("Reading", sub_path)

    sub = pd.read_csv(sub_path, sep="\t", dtype=str, low_memory=False)

    keep_cols = [
        "adsh", "cik", "name", "sic", "countryba", "stprba",
        "form", "period", "fy", "fp", "filed", "accepted"
    ]
    keep_cols = [c for c in keep_cols if c in sub.columns]

    sub = sub[keep_cols].copy()
    sub["quarter_folder"] = qdir.name

    out_rows.append(sub)

filings = pd.concat(out_rows, ignore_index=True)

filings.to_csv("sec_company_filings_2024_2026.csv", index=False, encoding="utf-8-sig")

print("\nSaved: sec_company_filings_2024_2026.csv")
print("Rows:", len(filings))
print(filings.head())