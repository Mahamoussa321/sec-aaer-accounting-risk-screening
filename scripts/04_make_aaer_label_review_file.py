#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_aaer_label_review_file.py

Creates a manual-review file for AAER labels.

Input expected in the same folder:
  outputs/aaer_quarterly_sivc_benchmark/aaer_proxy_labels_from_fuzzy_match.csv

Fallback input:
  outputs/aaer_sivc_benchmark/aaer_proxy_labels_from_fuzzy_match.csv

Output:
  aaer_label_review_template.csv

How to use:
  1) python make_aaer_label_review_file.py
  2) Open aaer_label_review_template.csv in Excel
  3) Review each row
  4) Set keep_label = 1 for true public-company AAER matches
     Set keep_label = 0 for false/non-company matches
  5) Save the reviewed file as:
       aaer_labels_reviewed.csv
  6) Rerun:
       python run_final_aaer_quarterly_sivc_benchmark.py
"""

from pathlib import Path
import pandas as pd
import re
import numpy as np

ROOT = Path(".")
CANDIDATE_FILES = [
    ROOT / "outputs" / "aaer_quarterly_sivc_benchmark" / "aaer_proxy_labels_from_fuzzy_match.csv",
    ROOT / "outputs" / "aaer_sivc_benchmark" / "aaer_proxy_labels_from_fuzzy_match.csv",
]
OUT = ROOT / "aaer_label_review_template.csv"

EXCLUDE_HINTS = [
    " CPA", " C P A", " CERTIFIED PUBLIC ACCOUNTANT", " LLP", " AUDIT",
    " ACCOUNTANTS", " ESQ", " ATTORNEY", " LAW", " ORDER DISMISSING",
    " ORDER REGARDING", " ORDER POSTPONING", " ORDER TERMINATING",
]

COMPANY_HINTS = [
    " INC", " CORP", " CORPORATION", " COMPANY", " CO.", " HOLDINGS",
    " LTD", " PLC", " GROUP", " TECHNOLOGIES", " ENERGY", " PHARMACEUTICALS",
    " SYSTEMS", " INTERNATIONAL", " INDUSTRIES",
]

def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find AAER proxy label file. Expected one of:\n" +
        "\n".join(str(p) for p in paths)
    )

def parse_aaer_number(x):
    if pd.isna(x):
        return ""
    m = re.findall(r"AAER-\d+", str(x))
    return "; ".join(sorted(set(m)))

def review_hint(text):
    s = str(text).upper()
    if any(k in s for k in EXCLUDE_HINTS):
        return "REVIEW CAREFULLY: may be auditor/CPA/law/order, not public-company issuer"
    if any(k in s for k in COMPANY_HINTS):
        return "LIKELY COMPANY MATCH"
    return "REVIEW"

def main():
    src = first_existing(CANDIDATE_FILES)
    print("Reading:", src)

    df = pd.read_csv(src, dtype=str)

    required = ["cik", "aaer_year"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column in candidate file: {c}")

    # Harmonize column names from earlier scripts
    if "company_name" not in df.columns and "sec_company_name" in df.columns:
        df["company_name"] = df["sec_company_name"]
    if "aaer_details" not in df.columns and "details" in df.columns:
        df["aaer_details"] = df["details"]
    if "aaer_numbers" not in df.columns:
        df["aaer_numbers"] = df.get("aaer_details", "").apply(parse_aaer_number)

    df["cik"] = df["cik"].astype(str).str.replace(r"\D", "", regex=True).str.lstrip("0")
    df["aaer_year"] = pd.to_numeric(df["aaer_year"], errors="coerce")
    df = df.dropna(subset=["cik", "aaer_year"]).copy()
    df["aaer_year"] = df["aaer_year"].astype(int)

    # One row per company-year-details
    keep_cols = [
        "keep_label",
        "review_status",
        "review_notes",
        "cik",
        "company_name",
        "aaer_year",
        "match_score",
        "aaer_date",
        "aaer_numbers",
        "aaer_details",
        "aaer_links",
        "label_source",
    ]

    df["keep_label"] = ""
    df["review_status"] = df.get("aaer_details", "").apply(review_hint)
    df["review_notes"] = ""

    for c in keep_cols:
        if c not in df.columns:
            df[c] = ""

    # Sort likely strongest matches first
    if "match_score" in df.columns:
        df["match_score_num"] = pd.to_numeric(df["match_score"], errors="coerce")
    else:
        df["match_score_num"] = np.nan

    out = (
        df[keep_cols + ["match_score_num"]]
        .drop_duplicates(subset=["cik", "aaer_year", "aaer_details"])
        .sort_values(["review_status", "match_score_num", "company_name", "aaer_year"], ascending=[True, False, True, True])
        .drop(columns=["match_score_num"])
    )

    out.to_csv(OUT, index=False, encoding="utf-8-sig")

    print("\nDONE")
    print("Saved:", OUT)
    print("Rows for review:", len(out))
    print("\nNext:")
    print("1) Open aaer_label_review_template.csv in Excel.")
    print("2) Put keep_label = 1 for true public-company issuer matches.")
    print("3) Put keep_label = 0 for false matches, CPA-only/audit-firm-only/order-only rows, or unrelated entities.")
    print("4) Save as aaer_labels_reviewed.csv in this same folder.")
    print("5) Rerun the quarterly benchmark script.")

if __name__ == "__main__":
    main()
