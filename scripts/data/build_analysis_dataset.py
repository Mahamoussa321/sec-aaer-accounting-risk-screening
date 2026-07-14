#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build the SEC accounting-risk analysis dataset directly from the extracted
SEC Financial Statement Data Sets.

Expected extracted structure
----------------------------
<EXTRACTED_ROOT>/
    2009q1/sub.txt
    2009q1/num.txt
    2009q2/sub.txt
    2009q2/num.txt
    ...

Outputs in the repository root
------------------------------
sec_financial_features_2009_2026.csv

If aaer_labels_reviewed.csv is present:
outputs/aaer_enhanced_accounting_benchmark/
    enhanced_10k_10q_analysis_dataset.csv
    dataset_build_summary.json

If aaer_labels_reviewed.csv is absent but sec_aaer_index.csv is present:
aaer_label_review_template.csv

After manually reviewing the template, save it as aaer_labels_reviewed.csv
and rerun this script.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz, process
except ImportError:
    fuzz = None
    process = None


# ---------------------------------------------------------------------------
# XBRL tag mappings. Earlier entries have higher priority.
# ---------------------------------------------------------------------------

INSTANT_TAGS: "OrderedDict[str, List[str]]" = OrderedDict(
    {
        "Assets": [
            "Assets",
        ],
        "Liabilities": [
            "Liabilities",
            "LiabilitiesAndStockholdersEquity",
        ],
        "AssetsCurrent": [
            "AssetsCurrent",
        ],
        "LiabilitiesCurrent": [
            "LiabilitiesCurrent",
        ],
        "Cash": [
            "CashAndCashEquivalentsAtCarryingValue",
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
            "CashAndDueFromBanks",
        ],
        "AccountsReceivable": [
            "AccountsReceivableNetCurrent",
            "AccountsReceivableNet",
            "AccountsNotesAndLoansReceivableNetCurrent",
        ],
        "Inventory": [
            "InventoryNet",
            "InventoryFinishedGoodsNetOfAllowancesCustomerAdvancesAndProgressBillings",
        ],
        "PropertyPlantEquipmentNet": [
            "PropertyPlantAndEquipmentNet",
            "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",
        ],
        "StockholdersEquity": [
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
            "PartnersCapital",
        ],
    }
)

FLOW_TAGS: "OrderedDict[str, List[str]]" = OrderedDict(
    {
        "Revenue": [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
            "SalesRevenueGoodsNet",
            "SalesRevenueServicesNet",
            "RegulatedAndUnregulatedOperatingRevenue",
        ],
        "GrossProfit": [
            "GrossProfit",
        ],
        "OperatingIncomeLoss": [
            "OperatingIncomeLoss",
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        ],
        "NetIncomeLoss": [
            "NetIncomeLoss",
            "ProfitLoss",
            "NetIncomeLossAvailableToCommonStockholdersBasic",
        ],
        "OperatingCashFlow": [
            "NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        ],
        "SGA": [
            "SellingGeneralAndAdministrativeExpense",
            "SellingAndMarketingExpense",
            "GeneralAndAdministrativeExpense",
        ],
        "RND": [
            "ResearchAndDevelopmentExpense",
        ],
    }
)

ALL_CANONICAL = list(INSTANT_TAGS) + list(FLOW_TAGS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SEC/AAER accounting-risk dataset from extracted SEC quarterly files."
    )
    parser.add_argument(
        "--extracted-root",
        required=True,
        help=r'Folder containing quarterly directories, e.g. "C:\Users\maha moussa\sec_financial_statements\extracted"',
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Root of the sec-aaer-accounting-risk-screening repository.",
    )
    parser.add_argument("--start-year", type=int, default=2009)
    parser.add_argument("--raw-end-year", type=int, default=2026)
    parser.add_argument("--analysis-end-year", type=int, default=2023)
    parser.add_argument("--label-window-years", type=int, default=3)
    parser.add_argument("--chunksize", type=int, default=750_000)
    parser.add_argument(
        "--reviewed-labels",
        default=None,
        help="Optional path to the adjudicated AAER label CSV. Defaults to data/labels/aaer_labels_reviewed.csv.",
    )
    parser.add_argument(
        "--aaer-index",
        default=None,
        help="Optional path to sec_aaer_index.csv. Defaults to the repository root.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Re-read all quarterly SEC files even if the feature CSV already exists.",
    )
    return parser.parse_args()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def normalize_cik(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"\D", "", regex=True)
        .str.lstrip("0")
        .replace("", np.nan)
    )


def safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    num_arr = pd.to_numeric(num, errors="coerce").astype(float)
    den_arr = pd.to_numeric(den, errors="coerce").astype(float)
    out = np.full(len(num_arr), np.nan, dtype=float)
    valid = np.isfinite(num_arr) & np.isfinite(den_arr) & (np.abs(den_arr) > 1e-12)
    out[valid] = num_arr[valid] / den_arr[valid]
    return pd.Series(out, index=num.index)


def quarter_dirs(root: Path, start_year: int, end_year: int) -> List[Path]:
    pattern = re.compile(r"^(\d{4})q([1-4])$", re.I)
    found: List[Tuple[int, int, Path]] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = pattern.match(path.name)
        if not match:
            continue
        year, quarter = int(match.group(1)), int(match.group(2))
        if start_year <= year <= end_year:
            found.append((year, quarter, path))
    found.sort(key=lambda x: (x[0], x[1]))
    return [p for _, _, p in found]


def build_tag_lookup() -> Dict[str, Tuple[str, str, int]]:
    lookup: Dict[str, Tuple[str, str, int]] = {}
    for canonical, tags in INSTANT_TAGS.items():
        for priority, tag in enumerate(tags):
            lookup[tag] = (canonical, "instant", priority)
    for canonical, tags in FLOW_TAGS.items():
        for priority, tag in enumerate(tags):
            lookup[tag] = (canonical, "flow", priority)
    return lookup


TAG_LOOKUP = build_tag_lookup()
SELECTED_TAGS = set(TAG_LOOKUP)


def read_submissions(sub_path: Path) -> pd.DataFrame:
    sub = pd.read_csv(sub_path, sep="\t", dtype=str, low_memory=False)
    sub = normalize_columns(sub)

    wanted = [
        "adsh", "cik", "name", "sic", "countryba", "stprba", "form",
        "period", "fy", "fp", "filed", "accepted",
    ]
    missing = [c for c in ["adsh", "cik", "form", "period"] if c not in sub.columns]
    if missing:
        raise ValueError(f"{sub_path} is missing required columns: {missing}")

    keep = [c for c in wanted if c in sub.columns]
    sub = sub[keep].copy()
    for c in wanted:
        if c not in sub.columns:
            sub[c] = np.nan

    sub["form"] = sub["form"].astype(str).str.upper().str.strip()
    sub = sub[sub["form"].isin(["10-K", "10-Q"])].copy()
    sub["cik"] = normalize_cik(sub["cik"])
    sub["period_num"] = pd.to_numeric(sub["period"], errors="coerce")
    sub["fy_num"] = pd.to_numeric(sub["fy"], errors="coerce")
    period_year = pd.to_numeric(sub["period"].astype(str).str[:4], errors="coerce")
    sub["fy_num"] = sub["fy_num"].fillna(period_year)
    sub = sub.dropna(subset=["adsh", "cik", "period_num", "fy_num"]).copy()
    sub["fy_num"] = sub["fy_num"].astype(int)
    sub["period_num"] = sub["period_num"].astype(int)
    sub = sub.drop_duplicates(subset=["adsh"], keep="last")
    return sub


def read_selected_facts(
    num_path: Path,
    sub: pd.DataFrame,
    chunksize: int,
) -> pd.DataFrame:
    # Inspect header first because SEC schemas can vary slightly by year.
    header = pd.read_csv(num_path, sep="\t", nrows=0)
    columns = [str(c).strip().lower() for c in header.columns]
    required = {"adsh", "tag", "ddate", "qtrs", "uom", "value"}
    missing = required - set(columns)
    if missing:
        raise ValueError(f"{num_path} is missing required columns: {sorted(missing)}")

    requested = [c for c in ["adsh", "tag", "version", "ddate", "qtrs", "uom", "coreg", "value"] if c in columns]
    sub_keys = sub[["adsh", "period_num", "form"]].copy()
    selected_adsh = set(sub_keys["adsh"].astype(str))
    pieces: List[pd.DataFrame] = []

    reader = pd.read_csv(
        num_path,
        sep="\t",
        usecols=requested,
        dtype={"adsh": str, "tag": str, "version": str, "uom": str, "coreg": str},
        chunksize=chunksize,
        low_memory=False,
    )

    for chunk in reader:
        chunk = normalize_columns(chunk)
        chunk = chunk[
            chunk["adsh"].astype(str).isin(selected_adsh)
            & chunk["tag"].astype(str).isin(SELECTED_TAGS)
        ].copy()
        if chunk.empty:
            continue

        if "coreg" in chunk.columns:
            coreg = chunk["coreg"].fillna("").astype(str).str.strip()
            chunk = chunk[coreg.eq("")].copy()
        if chunk.empty:
            continue

        # All mapped variables are monetary statement variables.
        chunk["uom"] = chunk["uom"].fillna("").astype(str).str.upper().str.strip()
        chunk = chunk[chunk["uom"].eq("USD")].copy()
        if chunk.empty:
            continue

        chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")
        chunk["ddate"] = pd.to_numeric(chunk["ddate"], errors="coerce")
        chunk["qtrs"] = pd.to_numeric(chunk["qtrs"], errors="coerce")
        chunk = chunk.dropna(subset=["value", "ddate", "qtrs"]).copy()
        if chunk.empty:
            continue

        chunk = chunk.merge(sub_keys, on="adsh", how="inner", validate="many_to_one")
        chunk = chunk[chunk["ddate"].astype(int).eq(chunk["period_num"].astype(int))].copy()
        if chunk.empty:
            continue

        mapped = chunk["tag"].map(TAG_LOOKUP)
        chunk["canonical"] = mapped.map(lambda x: x[0] if isinstance(x, tuple) else np.nan)
        chunk["kind"] = mapped.map(lambda x: x[1] if isinstance(x, tuple) else np.nan)
        chunk["tag_priority"] = mapped.map(lambda x: x[2] if isinstance(x, tuple) else 999)
        chunk = chunk.dropna(subset=["canonical"]).copy()

        preferred = np.where(
            chunk["kind"].eq("instant"),
            0,
            np.where(chunk["form"].eq("10-K"), 4, 1),
        )
        chunk["qtrs_penalty"] = np.abs(chunk["qtrs"].astype(float) - preferred.astype(float))

        # Instant values must be instant facts. Flow values can use a fallback
        # duration only if the preferred duration is unavailable.
        chunk = chunk[
            (chunk["kind"].eq("instant") & chunk["qtrs"].eq(0))
            | (chunk["kind"].eq("flow") & chunk["qtrs"].between(1, 4))
        ].copy()
        if chunk.empty:
            continue

        chunk["selection_score"] = (
            chunk["tag_priority"].astype(float) * 100.0
            + chunk["qtrs_penalty"].astype(float) * 10.0
        )
        pieces.append(chunk[["adsh", "canonical", "value", "selection_score"]])

    if not pieces:
        return pd.DataFrame(index=sub["adsh"].astype(str).unique())

    long = pd.concat(pieces, ignore_index=True)
    long = (
        long.sort_values(["adsh", "canonical", "selection_score"])
        .drop_duplicates(["adsh", "canonical"], keep="first")
    )
    wide = long.pivot(index="adsh", columns="canonical", values="value")
    wide.columns.name = None
    return wide


def build_raw_feature_table(
    extracted_root: Path,
    start_year: int,
    raw_end_year: int,
    chunksize: int,
) -> pd.DataFrame:
    qdirs = quarter_dirs(extracted_root, start_year, raw_end_year)
    if not qdirs:
        raise FileNotFoundError(
            f"No quarterly folders such as 2009q1 were found in {extracted_root}"
        )

    print(f"Found {len(qdirs)} quarterly folders.")
    all_quarters: List[pd.DataFrame] = []

    for i, qdir in enumerate(qdirs, start=1):
        sub_path = qdir / "sub.txt"
        num_path = qdir / "num.txt"
        if not sub_path.exists() or not num_path.exists():
            print(f"[{i}/{len(qdirs)}] SKIP {qdir.name}: sub.txt or num.txt is missing")
            continue

        print(f"[{i}/{len(qdirs)}] Reading {qdir.name}")
        sub = read_submissions(sub_path)
        if sub.empty:
            print("  no 10-K/10-Q submissions")
            continue

        facts = read_selected_facts(num_path, sub, chunksize)
        quarter = sub.merge(facts, left_on="adsh", right_index=True, how="left")
        quarter["quarter_folder"] = qdir.name
        all_quarters.append(quarter)
        print(f"  filings retained: {len(quarter):,}")

    if not all_quarters:
        raise RuntimeError("No filing rows were created from the extracted SEC data.")

    df = pd.concat(all_quarters, ignore_index=True)
    df = df.drop_duplicates(subset=["adsh"], keep="last").copy()

    for col in ALL_CANONICAL:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return engineer_features(df)


def exact_year_lag(df: pd.DataFrame, col: str) -> pd.Series:
    grouped = df.groupby(["cik", "fp_key"], sort=False, dropna=False)
    lag_val = grouped[col].shift(1)
    lag_year = grouped["fy_num"].shift(1)
    return lag_val.where((df["fy_num"] - lag_year).eq(1))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cik"] = normalize_cik(df["cik"])
    df["fy_num"] = pd.to_numeric(df["fy_num"], errors="coerce")
    df = df.dropna(subset=["cik", "fy_num"]).copy()
    df["fy_num"] = df["fy_num"].astype(int)

    fp = df["fp"].fillna("").astype(str).str.upper().str.strip()
    df["fp_key"] = np.where(df["form"].eq("10-K"), "FY", fp)
    df["fp_num"] = df["fp_key"].map({"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "FY": 4})
    df["fp_num"] = pd.to_numeric(df["fp_num"], errors="coerce").fillna(0).astype(int)
    df["is_10k"] = df["form"].eq("10-K").astype(int)

    filed_num = pd.to_numeric(df["filed"], errors="coerce").fillna(0)
    df = df.assign(_filed_num=filed_num).sort_values(
        ["cik", "fp_key", "fy_num", "_filed_num", "adsh"]
    ).reset_index(drop=True)

    # Core ratios.
    abs_assets = df["Assets"].abs()
    df["log_assets"] = np.log1p(abs_assets)
    df["leverage"] = safe_divide(df["Liabilities"], df["Assets"])
    df["current_ratio"] = safe_divide(df["AssetsCurrent"], df["LiabilitiesCurrent"])
    df["roa"] = safe_divide(df["NetIncomeLoss"], df["Assets"])
    df["profit_margin"] = safe_divide(df["NetIncomeLoss"], df["Revenue"])
    df["operating_margin"] = safe_divide(df["OperatingIncomeLoss"], df["Revenue"])
    df["cash_to_assets"] = safe_divide(df["Cash"], df["Assets"])
    df["receivables_to_revenue"] = safe_divide(df["AccountsReceivable"], df["Revenue"])
    df["inventory_to_assets"] = safe_divide(df["Inventory"], df["Assets"])
    df["accruals_to_assets"] = safe_divide(
        df["NetIncomeLoss"] - df["OperatingCashFlow"], df["Assets"]
    )
    df["loss_indicator"] = (df["NetIncomeLoss"] < 0).astype(float)

    # Exact prior-year, same-fiscal-period lags.
    lag_source_cols = [
        "Assets", "Liabilities", "AssetsCurrent", "LiabilitiesCurrent",
        "Revenue", "GrossProfit", "NetIncomeLoss", "OperatingCashFlow",
        "Cash", "AccountsReceivable", "Inventory", "SGA",
        "PropertyPlantEquipmentNet",
    ]
    for col in lag_source_cols:
        df[f"{col}_lag"] = exact_year_lag(df, col)

    # Year-over-year changes.
    df["rev_growth_yoy"] = safe_divide(df["Revenue"], df["Revenue_lag"]) - 1.0
    df["assets_growth_yoy"] = safe_divide(df["Assets"], df["Assets_lag"]) - 1.0
    df["liabilities_growth_yoy"] = safe_divide(df["Liabilities"], df["Liabilities_lag"]) - 1.0
    df["income_growth_yoy"] = safe_divide(df["NetIncomeLoss"], df["NetIncomeLoss_lag"]) - 1.0
    df["ocf_growth_yoy"] = safe_divide(df["OperatingCashFlow"], df["OperatingCashFlow_lag"]) - 1.0
    df["ar_growth_yoy"] = safe_divide(df["AccountsReceivable"], df["AccountsReceivable_lag"]) - 1.0
    df["inventory_growth_yoy"] = safe_divide(df["Inventory"], df["Inventory_lag"]) - 1.0
    df["cash_growth_yoy"] = safe_divide(df["Cash"], df["Cash_lag"]) - 1.0

    df["revenue_growth"] = df["rev_growth_yoy"]
    df["asset_growth"] = df["assets_growth_yoy"]

    # Beneish-style and related red-flag variables.
    df["gross_margin"] = safe_divide(df["GrossProfit"], df["Revenue"])
    df["gross_margin_lag"] = safe_divide(df["GrossProfit_lag"], df["Revenue_lag"])
    df["gross_margin_change"] = df["gross_margin"] - df["gross_margin_lag"]

    df["sga_to_sales"] = safe_divide(df["SGA"], df["Revenue"])
    df["sga_to_sales_lag"] = safe_divide(df["SGA_lag"], df["Revenue_lag"])
    df["sga_to_sales_change"] = df["sga_to_sales"] - df["sga_to_sales_lag"]

    ar_sales = safe_divide(df["AccountsReceivable"], df["Revenue"])
    ar_sales_lag = safe_divide(df["AccountsReceivable_lag"], df["Revenue_lag"])
    df["dsri"] = safe_divide(ar_sales, ar_sales_lag)
    df["gmi"] = safe_divide(df["gross_margin_lag"], df["gross_margin"])

    asset_quality = 1.0 - safe_divide(
        df["AssetsCurrent"].fillna(0) + df["PropertyPlantEquipmentNet"].fillna(0),
        df["Assets"],
    )
    asset_quality_lag = 1.0 - safe_divide(
        df["AssetsCurrent_lag"].fillna(0) + df["PropertyPlantEquipmentNet_lag"].fillna(0),
        df["Assets_lag"],
    )
    df["aqi"] = safe_divide(asset_quality, asset_quality_lag)
    df["sgi"] = safe_divide(df["Revenue"], df["Revenue_lag"])
    df["sgai"] = safe_divide(df["sga_to_sales"], df["sga_to_sales_lag"])

    leverage_lag = safe_divide(df["Liabilities_lag"], df["Assets_lag"])
    df["lvgi"] = safe_divide(df["leverage"], leverage_lag)
    df["tata"] = df["accruals_to_assets"]

    working_capital = (
        df["AssetsCurrent"].fillna(0)
        - df["LiabilitiesCurrent"].fillna(0)
        - df["Cash"].fillna(0)
    )
    working_capital_lag = (
        df["AssetsCurrent_lag"].fillna(0)
        - df["LiabilitiesCurrent_lag"].fillna(0)
        - df["Cash_lag"].fillna(0)
    )
    df["wc_accruals_to_assets"] = safe_divide(
        working_capital - working_capital_lag, df["Assets"]
    )
    df["cashflow_to_income"] = safe_divide(
        df["OperatingCashFlow"], df["NetIncomeLoss"]
    )
    df["revenue_to_assets"] = safe_divide(df["Revenue"], df["Assets"])
    df["ar_to_assets"] = safe_divide(df["AccountsReceivable"], df["Assets"])
    df["inventory_to_revenue"] = safe_divide(df["Inventory"], df["Revenue"])

    df = df.drop(columns=["_filed_num"], errors="ignore")
    return df


def normalize_company_name(value: object) -> str:
    text = str(value or "").upper()
    text = text.replace("&", " AND ")
    text = re.sub(r"[^A-Z0-9 ]+", " ", text)
    text = re.sub(r"\bTHE\b", " ", text)
    suffixes = (
        r"\b(INCORPORATED|INC|CORPORATION|CORP|COMPANY|CO|LIMITED|LTD|"
        r"PLC|LLC|LP|L P|HOLDINGS?|GROUP|SA|NV|AG|SE)\b"
    )
    text = re.sub(suffixes, " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_review_template(
    features: pd.DataFrame,
    aaer_index_path: Path,
    out_path: Path,
) -> None:
    if fuzz is None or process is None:
        raise ImportError(
            "rapidfuzz is required to create the review template. "
            "Install requirements.txt and rerun."
        )

    aaer = pd.read_csv(aaer_index_path, low_memory=False)
    if "details" not in aaer.columns:
        raise ValueError(f"{aaer_index_path} does not contain a 'details' column.")

    date_col = "date" if "date" in aaer.columns else None
    if date_col:
        parsed_dates = pd.to_datetime(aaer[date_col], errors="coerce")
        aaer["aaer_year"] = parsed_dates.dt.year
    else:
        aaer["aaer_year"] = pd.to_numeric(
            aaer["details"].astype(str).str.extract(r"\b(20\d{2})\b")[0],
            errors="coerce",
        )

    names = (
        features[["cik", "name"]]
        .dropna()
        .drop_duplicates()
        .assign(name_norm=lambda x: x["name"].map(normalize_company_name))
    )
    names = names[names["name_norm"].str.len().ge(5)].copy()

    # Keep the most frequent/recent displayed name for each normalized CIK/name.
    choices = names["name_norm"].drop_duplicates().tolist()
    norm_to_rows: Dict[str, pd.DataFrame] = {
        key: grp for key, grp in names.groupby("name_norm", sort=False)
    }

    rows: List[dict] = []
    print(f"Creating AAER review candidates from {len(aaer):,} AAER rows...")

    for idx, rec in aaer.iterrows():
        details = str(rec.get("details", ""))
        details_norm = normalize_company_name(details)
        if len(details_norm) < 5:
            continue

        candidates: Dict[str, float] = {}

        # Strong exact containment candidates.
        for name_norm in choices:
            if len(name_norm) >= 6 and re.search(
                rf"(?<![A-Z0-9]){re.escape(name_norm)}(?![A-Z0-9])",
                details_norm,
            ):
                candidates[name_norm] = 100.0

        # Add a small number of high fuzzy-score candidates.
        fuzzy_hits = process.extract(
            details_norm,
            choices,
            scorer=fuzz.partial_ratio,
            limit=5,
            score_cutoff=88,
        )
        for name_norm, score, _ in fuzzy_hits:
            candidates[name_norm] = max(candidates.get(name_norm, 0.0), float(score))

        for name_norm, score in sorted(
            candidates.items(), key=lambda kv: kv[1], reverse=True
        ):
            for _, company in norm_to_rows[name_norm].head(3).iterrows():
                rows.append(
                    {
                        "keep_label": "",
                        "review_status": (
                            "LIKELY COMPANY MATCH" if score >= 99 else "REVIEW"
                        ),
                        "review_notes": "",
                        "cik": company["cik"],
                        "company_name": company["name"],
                        "aaer_year": rec.get("aaer_year"),
                        "match_score": round(score, 2),
                        "aaer_date": rec.get("date", ""),
                        "aaer_numbers": rec.get("aaer_numbers", ""),
                        "aaer_details": details,
                        "aaer_links": rec.get("links", ""),
                        "label_source": "SEC AAER index + company-name candidate matching",
                    }
                )

        if (idx + 1) % 250 == 0:
            print(f"  processed {idx + 1:,}/{len(aaer):,}")

    review = pd.DataFrame(rows)
    if review.empty:
        raise RuntimeError("No AAER/company review candidates were generated.")

    review["aaer_year"] = pd.to_numeric(review["aaer_year"], errors="coerce")
    review = review.dropna(subset=["cik", "aaer_year"]).copy()
    review["aaer_year"] = review["aaer_year"].astype(int)
    review = (
        review.sort_values(
            ["review_status", "match_score", "company_name", "aaer_year"],
            ascending=[True, False, True, True],
        )
        .drop_duplicates(["cik", "aaer_year", "aaer_details"])
    )
    review.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved manual review template: {out_path}")
    print(f"Rows requiring review: {len(review):,}")


def apply_reviewed_labels(
    features: pd.DataFrame,
    reviewed_path: Path,
    label_window_years: int,
) -> pd.DataFrame:
    labels = pd.read_csv(reviewed_path, dtype=str, low_memory=False)
    required = {"cik", "aaer_year"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"{reviewed_path} is missing columns: {sorted(missing)}")

    labels["cik"] = normalize_cik(labels["cik"])
    labels["aaer_year"] = pd.to_numeric(labels["aaer_year"], errors="coerce")

    if "keep_label" in labels.columns:
        keep = pd.to_numeric(labels["keep_label"], errors="coerce").eq(1)
        labels = labels[keep].copy()

    labels = labels.dropna(subset=["cik", "aaer_year"]).copy()
    labels["aaer_year"] = labels["aaer_year"].astype(int)
    labels = labels.drop_duplicates(["cik", "aaer_year"])

    year_map: Dict[str, np.ndarray] = {
        str(cik): np.sort(grp["aaer_year"].unique())
        for cik, grp in labels.groupby("cik")
    }

    matched_year: List[float] = []
    outcome: List[int] = []
    for cik, fy in zip(features["cik"].astype(str), features["fy_num"].astype(int)):
        years = year_map.get(cik)
        if years is None or len(years) == 0:
            matched_year.append(np.nan)
            outcome.append(0)
            continue
        valid = years[(years >= fy) & (years <= fy + label_window_years)]
        if len(valid):
            matched_year.append(float(valid[0]))
            outcome.append(1)
        else:
            matched_year.append(np.nan)
            outcome.append(0)

    out = features.copy()
    out["aaer_label"] = np.asarray(outcome, dtype=int)
    out["matched_future_aaer_year"] = matched_year
    return out


MODEL_NUMERIC_FEATURES = [
    "log_assets", "leverage", "current_ratio", "roa", "profit_margin",
    "operating_margin", "cash_to_assets", "receivables_to_revenue",
    "inventory_to_assets", "accruals_to_assets", "loss_indicator",
    "revenue_growth", "asset_growth", "Assets", "Liabilities",
    "AssetsCurrent", "LiabilitiesCurrent", "Revenue", "GrossProfit",
    "OperatingIncomeLoss", "NetIncomeLoss", "OperatingCashFlow", "Cash",
    "AccountsReceivable", "Inventory", "SGA", "RND", "fp_num", "is_10k",
    "rev_growth_yoy", "assets_growth_yoy", "liabilities_growth_yoy",
    "income_growth_yoy", "ocf_growth_yoy", "dsri", "gmi", "aqi", "sgi",
    "sgai", "lvgi", "tata", "wc_accruals_to_assets",
    "cashflow_to_income", "ar_growth_yoy", "inventory_growth_yoy",
    "cash_growth_yoy", "gross_margin", "gross_margin_lag",
    "gross_margin_change", "sga_to_sales", "sga_to_sales_lag",
    "sga_to_sales_change", "revenue_to_assets", "ar_to_assets",
    "inventory_to_revenue",
]


def add_split_and_training_winsorization(
    labeled: pd.DataFrame,
    analysis_end_year: int,
) -> pd.DataFrame:
    df = labeled[labeled["fy_num"].between(2009, analysis_end_year)].copy()
    df["split"] = np.select(
        [
            df["fy_num"].between(2009, 2018),
            df["fy_num"].between(2019, 2020),
            df["fy_num"].between(2021, analysis_end_year),
        ],
        ["train", "val", "test"],
        default="exclude",
    )
    df = df[df["split"].ne("exclude")].copy()

    # Two-digit SIC categories learned from the training period.
    sic = (
        df["sic"]
        .astype(str)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(4)
        .str[:2]
    )
    sic = sic.where(sic.str.fullmatch(r"\d{2}"), "OTHER")
    train_sic = sorted(set(sic[df["split"].eq("train")]) - {"OTHER"})
    sic = sic.where(sic.isin(train_sic), "OTHER")
    dummies = pd.get_dummies(sic, prefix="sic2", dtype=int)
    df = pd.concat([df.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)

    # Winsorization bounds are learned only on training observations.
    skip = {"loss_indicator", "fp_num", "is_10k"}
    bounds: Dict[str, Tuple[float, float]] = {}
    for col in MODEL_NUMERIC_FEATURES:
        if col not in df.columns or col in skip:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        train_values = df.loc[df["split"].eq("train"), col].replace(
            [np.inf, -np.inf], np.nan
        )
        finite = train_values.dropna()
        if len(finite) < 100:
            continue
        lo, hi = finite.quantile([0.005, 0.995]).tolist()
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            bounds[col] = (float(lo), float(hi))
            df[col] = df[col].clip(lo, hi)

    df.attrs["winsorization_bounds"] = bounds
    return df


def main() -> int:
    args = parse_args()
    started = time.time()

    extracted_root = Path(args.extracted_root).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    repo_root.mkdir(parents=True, exist_ok=True)

    if not extracted_root.exists():
        raise FileNotFoundError(f"Extracted SEC folder does not exist: {extracted_root}")

    feature_csv = repo_root / f"sec_financial_features_{args.start_year}_{args.raw_end_year}.csv"

    if feature_csv.exists() and not args.force_rebuild:
        print(f"Using existing feature table: {feature_csv}")
        features = pd.read_csv(feature_csv, low_memory=False)
    else:
        features = build_raw_feature_table(
            extracted_root=extracted_root,
            start_year=args.start_year,
            raw_end_year=args.raw_end_year,
            chunksize=args.chunksize,
        )
        features.to_csv(feature_csv, index=False, encoding="utf-8-sig")
        print(f"Saved feature table: {feature_csv}")
        print(f"Rows: {len(features):,}; unique firms: {features['cik'].nunique():,}")

    if args.reviewed_labels:
        reviewed_path = Path(args.reviewed_labels).expanduser().resolve()
    else:
        reviewed_candidates = [
            repo_root / "data" / "labels" / "aaer_labels_reviewed.csv",
            repo_root / "aaer_labels_reviewed.csv",
        ]
        reviewed_path = next(
            (candidate for candidate in reviewed_candidates if candidate.exists()),
            reviewed_candidates[0],
        )

    if args.aaer_index:
        aaer_index_path = Path(args.aaer_index).expanduser().resolve()
    else:
        aaer_index_path = repo_root / "sec_aaer_index.csv"

    review_template_path = repo_root / "aaer_label_review_template.csv"

    if not reviewed_path.exists():
        if aaer_index_path.exists():
            make_review_template(features, aaer_index_path, review_template_path)
            print("\nMANUAL REVIEW REQUIRED")
            print("----------------------")
            print(f"Open: {review_template_path}")
            print("Set keep_label=1 for true public-company issuer matches.")
            print("Set keep_label=0 for false, individual-only, auditor-only, or law-firm-only matches.")
            print(f"Save the reviewed file as: {reviewed_path}")
            print("Then rerun this same command; the feature CSV will be reused.")
            return 2

        print("\nAAER files are missing.")
        print("Run scripts/01_download_sec_aaer_index.py first, then rerun this builder.")
        print(f"Expected: {aaer_index_path}")
        return 2

    labeled = apply_reviewed_labels(
        features=features,
        reviewed_path=reviewed_path,
        label_window_years=args.label_window_years,
    )
    analysis = add_split_and_training_winsorization(
        labeled=labeled,
        analysis_end_year=args.analysis_end_year,
    )

    output_dir = repo_root / "outputs" / "aaer_enhanced_accounting_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "enhanced_10k_10q_analysis_dataset.csv"
    analysis.to_csv(dataset_path, index=False, encoding="utf-8-sig")

    summary = {
        "extracted_root": str(extracted_root),
        "feature_csv": str(feature_csv),
        "reviewed_labels": str(reviewed_path),
        "analysis_dataset": str(dataset_path),
        "label_window_years": args.label_window_years,
        "analysis_end_year": args.analysis_end_year,
        "rows": int(len(analysis)),
        "unique_firms": int(analysis["cik"].nunique()),
        "positive_rows": int(analysis["aaer_label"].sum()),
        "split_summary": (
            analysis.groupby("split")["aaer_label"]
            .agg(rows="size", positives="sum", prevalence="mean")
            .reset_index()
            .to_dict(orient="records")
        ),
        "winsorization_bounds": {
            key: [value[0], value[1]]
            for key, value in analysis.attrs.get("winsorization_bounds", {}).items()
        },
    }
    summary_path = output_dir / "dataset_build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nDATASET BUILD COMPLETE")
    print("----------------------")
    print(f"Saved: {dataset_path}")
    print(f"Rows: {len(analysis):,}")
    print(f"Unique firms: {analysis['cik'].nunique():,}")
    print(f"AAER-positive rows: {int(analysis['aaer_label'].sum()):,}")
    print("\nOutcome by split:")
    print(
        analysis.groupby("split")["aaer_label"]
        .agg(rows="size", positives="sum", prevalence="mean")
        .to_string()
    )
    print(f"\nElapsed: {(time.time() - started) / 60:.1f} minutes")
    print("\nNext run:")
    print("python scripts/analysis/run_main_benchmark.py")
    print("python scripts/analysis/make_main_figures.py")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped by user.", file=sys.stderr)
        raise SystemExit(130)
