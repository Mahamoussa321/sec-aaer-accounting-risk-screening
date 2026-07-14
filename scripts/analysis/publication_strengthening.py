#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Publication-strengthening analyses for the SEC-AAER screening study.

This script is intentionally self-contained and resumable. It adds:
  1. firm-clustered bootstrap confidence intervals and paired model differences;
  2. raw and validation-calibrated probability diagnostics;
  3. 10-K/10-Q and year-specific robustness;
  4. firm-disjoint temporal cross-fitting (cold-start firm validation);
  5. DeepCut repeated-seed stability and structural ablations;
  6. 10-K-only retraining;
  7. AAER label-window sensitivity (0, 1, 2, and 3 years);
  8. publication tables, figures, and an automatically generated report.

Run from the repository root, or pass --project-root.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RANDOM_STATE = 20260531
TOP_FRACTIONS = (0.005, 0.01, 0.02, 0.05, 0.10)

NUMERIC_FEATURES = [
    "log_assets", "leverage", "current_ratio", "roa", "profit_margin",
    "operating_margin", "cash_to_assets", "receivables_to_revenue",
    "inventory_to_assets", "accruals_to_assets", "loss_indicator",
    "revenue_growth", "asset_growth", "Assets", "Liabilities",
    "AssetsCurrent", "LiabilitiesCurrent", "Revenue", "GrossProfit",
    "OperatingIncomeLoss", "NetIncomeLoss", "OperatingCashFlow", "Cash",
    "AccountsReceivable", "Inventory", "SGA", "RND", "fp_num", "is_10k",
    "rev_growth_yoy", "assets_growth_yoy", "liabilities_growth_yoy",
    "income_growth_yoy", "ocf_growth_yoy", "dsri", "gmi", "aqi", "sgi",
    "sgai", "lvgi", "tata", "wc_accruals_to_assets", "cashflow_to_income",
    "ar_growth_yoy", "inventory_growth_yoy", "cash_growth_yoy",
    "gross_margin", "gross_margin_lag", "gross_margin_change", "sga_to_sales",
    "sga_to_sales_lag", "sga_to_sales_change", "revenue_to_assets",
    "ar_to_assets", "inventory_to_revenue",
]

DEEPCUT_THRESHOLD_FEATURES = [
    "tata", "accruals_to_assets", "wc_accruals_to_assets", "dsri", "gmi",
    "aqi", "sgi", "sgai", "lvgi", "leverage", "roa", "current_ratio",
    "gross_margin_change", "rev_growth_yoy", "assets_growth_yoy",
    "ar_growth_yoy", "inventory_growth_yoy", "cash_to_assets",
    "receivables_to_revenue", "inventory_to_revenue", "loss_indicator",
]

DEEPCUT_CONTEXT_BASE = [
    "log_assets", "fp_num", "is_10k", "leverage", "roa", "cash_to_assets",
    "loss_indicator",
]

MODEL_DISPLAY = {
    "prob_Logistic_regression": "Logistic regression",
    "prob_Random_forest": "Random forest",
    "prob_Extra_trees": "Extra trees",
    "prob_Gradient_boosting": "Gradient boosting",
    "prob_HistGradientBoosting": "HistGradientBoosting",
    "prob_DeepCut_inspired_neural_threshold": "DeepCut-inspired neural threshold",
}


@dataclass
class Paths:
    root: Path
    data_file: Path
    predictions_file: Path
    reviewed_labels: Path
    output_dir: Path
    cache_dir: Path
    figures_dir: Path
    tables_dir: Path


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def find_first(candidates: Iterable[Path], description: str) -> Path:
    for path in candidates:
        if path.exists():
            return path.resolve()
    joined = "\n - ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find {description}. Checked:\n - {joined}")


def resolve_paths(root: Path) -> Paths:
    root = root.resolve()
    data_file = find_first(
        [
            root / "outputs" / "aaer_enhanced_accounting_benchmark" / "enhanced_10k_10q_analysis_dataset.csv",
            root / "aaer_enhanced_accounting_benchmark" / "enhanced_10k_10q_analysis_dataset.csv",
        ],
        "the enhanced analysis dataset",
    )
    predictions_file = find_first(
        [
            root / "outputs" / "accounting_deepcut_ml_benchmark" / "test_predictions.csv",
            root / "accounting_deepcut_ml_benchmark" / "test_predictions.csv",
        ],
        "the main test predictions",
    )
    reviewed_labels = find_first(
        [
            root / "aaer_labels_reviewed.csv",
            root / "manual_review" / "aaer_labels_reviewed.csv",
            root.parent.parent / "manual_review" / "aaer_labels_reviewed.csv",
        ],
        "the reviewed AAER labels",
    )
    output_dir = root / "outputs" / "publication_strengthening"
    cache_dir = output_dir / "cache"
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    for path in (output_dir, cache_dir, figures_dir, tables_dir):
        path.mkdir(parents=True, exist_ok=True)
    return Paths(root, data_file, predictions_file, reviewed_labels, output_dir, cache_dir, figures_dir, tables_dir)


def parse_seed_list(value: str) -> list[int]:
    seeds = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def safe_roc_auc(y: np.ndarray, p: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    y = np.asarray(y, dtype=int)
    if sample_weight is None:
        active = np.ones(len(y), dtype=bool)
    else:
        active = np.asarray(sample_weight) > 0
    if len(np.unique(y[active])) < 2:
        return float("nan")
    return float(roc_auc_score(y, p, sample_weight=sample_weight))


def safe_pr_auc(y: np.ndarray, p: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    y = np.asarray(y, dtype=int)
    if sample_weight is None:
        active = np.ones(len(y), dtype=bool)
    else:
        active = np.asarray(sample_weight) > 0
    if len(np.unique(y[active])) < 2:
        return float("nan")
    return float(average_precision_score(y, p, sample_weight=sample_weight))


def expected_calibration_error(
    y: np.ndarray,
    p: np.ndarray,
    sample_weight: np.ndarray | None = None,
    n_bins: int = 10,
) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    w = np.ones(len(y), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if w.sum() <= 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    value = 0.0
    for i in range(n_bins):
        mask = (p >= edges[i]) & ((p < edges[i + 1]) if i < n_bins - 1 else (p <= edges[i + 1])) & (w > 0)
        if not np.any(mask):
            continue
        bin_weight = float(w[mask].sum())
        observed = float(np.average(y[mask], weights=w[mask]))
        predicted = float(np.average(p[mask], weights=w[mask]))
        value += (bin_weight / float(w.sum())) * abs(observed - predicted)
    return float(value)


def calibration_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"y": np.asarray(y, int), "p": np.asarray(p, float)})
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), q=min(n_bins, len(frame)), duplicates="drop")
    out = frame.groupby("bin", observed=True).agg(
        n=("y", "size"),
        observed_rate=("y", "mean"),
        mean_probability=("p", "mean"),
        min_probability=("p", "min"),
        max_probability=("p", "max"),
    ).reset_index(drop=True)
    return out


def pick_threshold(y: np.ndarray, p: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for threshold in np.linspace(0.01, 0.99, 99):
        pred = (p >= threshold).astype(int)
        score = f1_score(y, pred, zero_division=0)
        if score > best_f1:
            best_t, best_f1 = float(threshold), float(score)
    return best_t


def weighted_topk(
    y: np.ndarray,
    p: np.ndarray,
    fraction: float,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    w = np.ones(len(y), dtype=float) if sample_weight is None else np.asarray(sample_weight, float)
    total_weight = float(w.sum())
    total_positive = float(np.sum(w * y))
    if total_weight <= 0 or total_positive <= 0:
        return {"top_k": float("nan"), "true_positives": float("nan"), "precision": float("nan"), "recall": float("nan"), "lift": float("nan")}
    budget = max(1.0, fraction * total_weight)
    order = np.argsort(-p, kind="mergesort")
    ww = w[order]
    yy = y[order]
    cumulative = np.cumsum(ww)
    last = min(int(np.searchsorted(cumulative, budget, side="left")), len(order) - 1)
    selected_w = ww[: last + 1].copy()
    selected_y = yy[: last + 1]
    excess = float(selected_w.sum() - budget)
    if excess > 0:
        selected_w[-1] = max(0.0, selected_w[-1] - excess)
    true_positives = float(np.sum(selected_w * selected_y))
    precision = true_positives / budget
    recall = true_positives / total_positive
    prevalence = total_positive / total_weight
    return {
        "top_k": budget,
        "true_positives": true_positives,
        "precision": precision,
        "recall": recall,
        "lift": precision / prevalence if prevalence > 0 else float("nan"),
    }


def metric_row(
    model: str,
    y: np.ndarray,
    p: np.ndarray,
    threshold: float | None = None,
    subgroup: str = "Temporal test",
) -> dict[str, float | int | str]:
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), 0.0, 1.0)
    if threshold is None:
        threshold = 0.5
    pred = (p >= threshold).astype(int)
    row: dict[str, float | int | str] = {
        "subgroup": subgroup,
        "model": model,
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()) if len(y) else float("nan"),
        "threshold": float(threshold),
        "roc_auc": safe_roc_auc(y, p),
        "pr_auc": safe_pr_auc(y, p),
        "brier": float(brier_score_loss(y, p)) if len(y) else float("nan"),
        "log_loss": float(log_loss(y, np.clip(p, 1e-8, 1 - 1e-8), labels=[0, 1])) if len(y) else float("nan"),
        "ece_10": expected_calibration_error(y, p),
        "mean_probability": float(np.mean(p)) if len(y) else float("nan"),
        "observed_to_expected": float(y.sum() / p.sum()) if p.sum() > 0 else float("nan"),
        "accuracy": float(accuracy_score(y, pred)) if len(y) else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) == 2 else float("nan"),
        "precision": float(precision_score(y, pred, zero_division=0)) if len(y) else float("nan"),
        "recall": float(recall_score(y, pred, zero_division=0)) if len(y) else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)) if len(y) else float("nan"),
    }
    for fraction in TOP_FRACTIONS:
        values = weighted_topk(y, p, fraction)
        suffix = str(fraction).replace(".", "p")
        row[f"precision_top_{suffix}"] = values["precision"]
        row[f"recall_top_{suffix}"] = values["recall"]
        row[f"lift_top_{suffix}"] = values["lift"]
        row[f"positives_top_{suffix}"] = values["true_positives"]
    return row


def sanitize_probability_column(name: str) -> str:
    return "prob_" + re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def load_data(paths: Paths) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    log(f"Reading dataset: {paths.data_file}")
    header = pd.read_csv(paths.data_file, nrows=0).columns.tolist()
    sic_features = [c for c in header if str(c).startswith("sic2_")]
    numeric_features = [c for c in NUMERIC_FEATURES if c in header]
    all_features = numeric_features + sic_features
    threshold_features = [c for c in DEEPCUT_THRESHOLD_FEATURES if c in header]
    context_features = [c for c in DEEPCUT_CONTEXT_BASE if c in header] + sic_features
    metadata = [c for c in ["cik", "name", "fy_num", "fp", "form", "split", "aaer_label", "matched_future_aaer_year"] if c in header]
    usecols = list(dict.fromkeys(metadata + all_features + threshold_features + context_features))
    df = pd.read_csv(paths.data_file, usecols=usecols, low_memory=False)
    df["cik"] = df["cik"].astype(str)
    df["aaer_label"] = pd.to_numeric(df["aaer_label"], errors="raise").astype("int8")
    df["fy_num"] = pd.to_numeric(df["fy_num"], errors="coerce")
    for col in all_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if col.startswith("sic2_") or col in {"fp_num", "is_10k", "loss_indicator"}:
            df[col] = df[col].astype("float32")
    log(f"Loaded {len(df):,} rows, {df['cik'].nunique():,} firms, {int(df['aaer_label'].sum()):,} positive rows.")
    return df, all_features, threshold_features, context_features


def existing_prediction_diagnostics(paths: Paths, bootstrap_reps: int, seed: int) -> None:
    marker = paths.cache_dir / "existing_diagnostics.done"
    if marker.exists():
        log("Skipping existing-prediction diagnostics (already completed).")
        return
    log("Stage 1/7: existing-prediction diagnostics and firm-clustered bootstrap CIs.")
    pred = pd.read_csv(paths.predictions_file, low_memory=False)
    pred["cik"] = pred["cik"].astype(str)
    train_ciks = set(
        pd.read_csv(paths.data_file, usecols=["cik", "split"], low_memory=False)
        .query("split == 'train'")["cik"].astype(str)
    )
    pred["firm_seen_in_train"] = pred["cik"].isin(train_ciks)
    probability_columns = [c for c in pred.columns if c.startswith("prob_")]
    display = {c: MODEL_DISPLAY.get(c, c.replace("prob_", "").replace("_", " ")) for c in probability_columns}

    subgroups = {
        "All temporal test": pred,
        "10-K only (model trained on 10-K and 10-Q)": pred[pred["form"].eq("10-K")],
        "10-Q only (model trained on 10-K and 10-Q)": pred[pred["form"].eq("10-Q")],
        "Firms represented in the training period": pred[pred["firm_seen_in_train"]],
        "Firms absent from the training period": pred[~pred["firm_seen_in_train"]],
    }
    subgroup_rows: list[dict] = []
    for label, frame in subgroups.items():
        y = frame["aaer_label"].to_numpy(int)
        for col in probability_columns:
            row = metric_row(display[col], y, frame[col].to_numpy(float), subgroup=label)
            row["firms"] = int(frame["cik"].nunique())
            subgroup_rows.append(row)
    subgroup_table = pd.DataFrame(subgroup_rows)
    subgroup_table.to_csv(paths.tables_dir / "subgroup_robustness_metrics.csv", index=False)

    annual_rows = []
    for year, frame in pred.groupby("fy_num"):
        y = frame["aaer_label"].to_numpy(int)
        for col in probability_columns:
            annual_rows.append(metric_row(display[col], y, frame[col].to_numpy(float), subgroup=f"FY {int(year)}"))
    pd.DataFrame(annual_rows).to_csv(paths.tables_dir / "year_specific_metrics.csv", index=False)

    calibration_rows = []
    for col in probability_columns:
        bins = calibration_bins(pred["aaer_label"].to_numpy(int), pred[col].to_numpy(float), n_bins=10)
        bins.insert(0, "model", display[col])
        calibration_rows.append(bins)
    pd.concat(calibration_rows, ignore_index=True).to_csv(paths.tables_dir / "raw_calibration_bins.csv", index=False)

    y = pred["aaer_label"].to_numpy(int)
    firm_codes, unique_firms = pd.factorize(pred["cik"], sort=True)
    n_firms = len(unique_firms)
    rng = np.random.default_rng(seed)
    arrays = {col: pred[col].to_numpy(float) for col in probability_columns}
    metrics = ["roc_auc", "pr_auc", "brier", "recall_top_0.02", "recall_top_0.05", "recall_top_0.1"]
    point: dict[str, dict[str, float]] = {}
    for col, p in arrays.items():
        point[col] = {
            "roc_auc": safe_roc_auc(y, p),
            "pr_auc": safe_pr_auc(y, p),
            "brier": float(np.mean((y - p) ** 2)),
        }
        for fraction in (0.02, 0.05, 0.10):
            point[col][f"recall_top_{fraction}"] = weighted_topk(y, p, fraction)["recall"]
    bootstrap: dict[str, dict[str, list[float]]] = {col: {m: [] for m in metrics} for col in probability_columns}
    deep_col = "prob_DeepCut_inspired_neural_threshold"
    paired: dict[tuple[str, str], list[float]] = {}
    for rep in range(bootstrap_reps):
        firm_counts = rng.multinomial(n_firms, np.full(n_firms, 1.0 / n_firms))
        weights = firm_counts[firm_codes].astype(float)
        if np.sum(weights * y) <= 0 or np.sum(weights * (1 - y)) <= 0:
            continue
        current: dict[str, dict[str, float]] = {}
        for col, p in arrays.items():
            values = {
                "roc_auc": safe_roc_auc(y, p, weights),
                "pr_auc": safe_pr_auc(y, p, weights),
                "brier": float(np.average((y - p) ** 2, weights=weights)),
            }
            for fraction in (0.02, 0.05, 0.10):
                values[f"recall_top_{fraction}"] = weighted_topk(y, p, fraction, weights)["recall"]
            current[col] = values
            for metric, value in values.items():
                bootstrap[col][metric].append(value)
        if deep_col in current:
            for other in probability_columns:
                if other == deep_col:
                    continue
                for metric in ("roc_auc", "pr_auc", "recall_top_0.05", "recall_top_0.1"):
                    paired.setdefault((other, metric), []).append(current[deep_col][metric] - current[other][metric])
        if (rep + 1) % max(1, min(100, bootstrap_reps)) == 0:
            log(f"  completed {rep + 1}/{bootstrap_reps} cluster-bootstrap replicates")

    ci_rows = []
    for col in probability_columns:
        for metric in metrics:
            values = np.asarray(bootstrap[col][metric], float)
            values = values[np.isfinite(values)]
            ci_rows.append({
                "model": display[col],
                "metric": metric,
                "estimate": point[col][metric],
                "ci_lower": float(np.quantile(values, 0.025)) if len(values) else np.nan,
                "ci_upper": float(np.quantile(values, 0.975)) if len(values) else np.nan,
                "bootstrap_reps": int(len(values)),
                "cluster_unit": "CIK firm",
            })
    ci_table = pd.DataFrame(ci_rows)
    ci_table.to_csv(paths.tables_dir / "firm_clustered_bootstrap_confidence_intervals.csv", index=False)

    paired_rows = []
    for (other, metric), values in paired.items():
        arr = np.asarray(values, float)
        arr = arr[np.isfinite(arr)]
        estimate = point[deep_col][metric] - point[other][metric]
        p_two = min(1.0, 2.0 * min(float(np.mean(arr <= 0)), float(np.mean(arr >= 0)))) if len(arr) else np.nan
        paired_rows.append({
            "comparison": f"DeepCut-inspired neural threshold minus {display[other]}",
            "metric": metric,
            "estimate_difference": estimate,
            "ci_lower": float(np.quantile(arr, 0.025)) if len(arr) else np.nan,
            "ci_upper": float(np.quantile(arr, 0.975)) if len(arr) else np.nan,
            "bootstrap_p_two_sided": p_two,
            "bootstrap_reps": int(len(arr)),
        })
    pd.DataFrame(paired_rows).to_csv(paths.tables_dir / "paired_firm_clustered_model_differences.csv", index=False)

    unseen = pred[~pred["firm_seen_in_train"]]
    note = (
        f"Test rows from firms absent in training: {len(unseen):,}\n"
        f"Unique unseen firms: {unseen['cik'].nunique():,}\n"
        f"Positive rows among unseen firms: {int(unseen['aaer_label'].sum()):,}\n\n"
    )
    if int(unseen["aaer_label"].sum()) == 0:
        note += (
            "ROC-AUC, PR-AUC, and recall cannot be estimated on this naturally occurring unseen-firm subset "
            "because it contains no positive outcomes. The firm-disjoint temporal cross-fitting stage therefore "
            "constructs a valid cold-start evaluation by holding out complete firms, including positive firms, "
            "from all training and validation periods.\n"
        )
    (paths.output_dir / "unseen_firm_diagnostic.txt").write_text(note, encoding="utf-8")

    # Forest plots for the two ranking metrics.
    for metric, title, filename in [
        ("roc_auc", "Firm-clustered 95% CIs for ROC-AUC", "figS1_clustered_roc_auc_cis.png"),
        ("pr_auc", "Firm-clustered 95% CIs for PR-AUC", "figS2_clustered_pr_auc_cis.png"),
    ]:
        plot = ci_table[ci_table["metric"].eq(metric)].sort_values("estimate")
        fig, ax = plt.subplots(figsize=(9, 5.5))
        positions = np.arange(len(plot))
        estimate = plot["estimate"].to_numpy(float)
        lower = plot["ci_lower"].to_numpy(float)
        upper = plot["ci_upper"].to_numpy(float)
        ax.errorbar(estimate, positions, xerr=np.vstack([estimate - lower, upper - estimate]), fmt="o", capsize=4)
        ax.set_yticks(positions, plot["model"])
        ax.set_xlabel(metric.replace("_", " ").upper())
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(paths.figures_dir / filename, dpi=300, bbox_inches="tight")
        fig.savefig((paths.figures_dir / filename).with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)

    marker.write_text(now(), encoding="utf-8")
    log("Existing-prediction diagnostics completed.")


def make_popular_models(seed: int) -> dict[str, object]:
    return {
        "Logistic regression": LogisticRegression(max_iter=4000, class_weight="balanced", random_state=seed),
        "Random forest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=8, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed,
        ),
        "Extra trees": ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=8, max_features="sqrt",
            class_weight="balanced", n_jobs=-1, random_state=seed,
        ),
        "Gradient boosting": GradientBoostingClassifier(
            n_estimators=350, learning_rate=0.025, max_depth=2, subsample=0.8, random_state=seed,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=350, learning_rate=0.025, max_leaf_nodes=31,
            l2_regularization=0.01, random_state=seed,
        ),
    }


def fit_popular_model(
    model_name: str,
    model: object,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    y_train: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if y_train is None:
        y_train = train["aaer_label"].to_numpy(int)
    if y_val is None:
        y_val = val["aaer_label"].to_numpy(int)
    if y_test is None:
        y_test = test["aaer_label"].to_numpy(int)
    pre = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    pipe = Pipeline([("pre", pre), ("clf", model)])
    if model_name in {"Gradient boosting", "HistGradientBoosting"}:
        positives = max(1, int(y_train.sum()))
        negatives = max(1, int(len(y_train) - y_train.sum()))
        weight_pos = len(y_train) / (2 * positives)
        weight_neg = len(y_train) / (2 * negatives)
        sample_weight = np.where(y_train == 1, weight_pos, weight_neg)
        pipe.fit(train[list(features)], y_train, clf__sample_weight=sample_weight)
    else:
        pipe.fit(train[list(features)], y_train)
    return pipe.predict_proba(val[list(features)])[:, 1], pipe.predict_proba(test[list(features)])[:, 1]


def fit_probability_calibrator(
    y_val: np.ndarray,
    p_val: np.ndarray,
    p_test: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    eps = 1e-6
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {"raw": (p_val, p_test)}
    logit_val = np.log(np.clip(p_val, eps, 1 - eps) / np.clip(1 - p_val, eps, 1 - eps)).reshape(-1, 1)
    logit_test = np.log(np.clip(p_test, eps, 1 - eps) / np.clip(1 - p_test, eps, 1 - eps)).reshape(-1, 1)
    platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    platt.fit(logit_val, y_val)
    result["platt"] = (platt.predict_proba(logit_val)[:, 1], platt.predict_proba(logit_test)[:, 1])
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(p_val, y_val)
    result["isotonic"] = (isotonic.predict(p_val), isotonic.predict(p_test))
    return result


class DeepCutTrainer:
    def __init__(
        self,
        threshold_features: Sequence[str],
        context_features: Sequence[str],
        max_epochs: int = 120,
        patience: int = 18,
        batch_size: int = 4096,
        device: str = "auto",
    ) -> None:
        self.threshold_features = list(threshold_features)
        self.context_features = list(context_features)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.batch_size = int(batch_size)
        self.device_requested = device

    def prepare(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
    ) -> dict[str, object]:
        x_imp = SimpleImputer(strategy="median")
        x_scaler = StandardScaler()
        z_imp = SimpleImputer(strategy="median")
        z_scaler = StandardScaler(with_mean=False)
        x_train_raw = x_imp.fit_transform(train[self.threshold_features])
        x_val_raw = x_imp.transform(val[self.threshold_features])
        x_test_raw = x_imp.transform(test[self.threshold_features])
        x_train = x_scaler.fit_transform(x_train_raw).astype("float32")
        x_val = x_scaler.transform(x_val_raw).astype("float32")
        x_test = x_scaler.transform(x_test_raw).astype("float32")
        z_train = z_scaler.fit_transform(z_imp.fit_transform(train[self.context_features])).astype("float32")
        z_val = z_scaler.transform(z_imp.transform(val[self.context_features])).astype("float32")
        z_test = z_scaler.transform(z_imp.transform(test[self.context_features])).astype("float32")
        return {
            "x_train": x_train, "x_val": x_val, "x_test": x_test,
            "z_train": z_train, "z_val": z_val, "z_test": z_test,
            "x_scaler": x_scaler,
        }

    def fit_arrays(
        self,
        arrays: dict[str, object],
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        seed: int,
        variant: str = "full_adaptive_two_sided",
    ) -> dict[str, object]:
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:
            raise ImportError("PyTorch is required for the DeepCut analyses.") from exc

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if self.device_requested == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.device_requested)
        log(f"DeepCut variant={variant}, seed={seed}, device={device}")

        x_train = np.asarray(arrays["x_train"], dtype="float32")
        x_val = np.asarray(arrays["x_val"], dtype="float32")
        x_test = np.asarray(arrays["x_test"], dtype="float32")
        z_train = np.asarray(arrays["z_train"], dtype="float32")
        z_val = np.asarray(arrays["z_val"], dtype="float32")
        z_test = np.asarray(arrays["z_test"], dtype="float32")
        y_train = np.asarray(y_train, dtype="float32")
        y_val = np.asarray(y_val, dtype="float32")
        y_test = np.asarray(y_test, dtype="float32")

        class DeepCutAccounting(nn.Module):
            def __init__(self, p: int, q: int, chosen_variant: str, hidden: int = 64, dropout: float = 0.15):
                super().__init__()
                self.variant = chosen_variant
                self.p = p
                self.context_net = nn.Sequential(
                    nn.Linear(q, hidden), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                )
                self.base = nn.Linear(hidden, 1)
                if chosen_variant == "context_only_mlp":
                    self.context_head = nn.Linear(hidden, 1)
                elif chosen_variant == "fixed_global_two_sided":
                    self.global_mid = nn.Parameter(torch.zeros(p))
                    self.global_log_width = nn.Parameter(torch.zeros(p))
                    self.global_w_low = nn.Parameter(torch.zeros(p))
                    self.global_w_high = nn.Parameter(torch.zeros(p))
                elif chosen_variant == "adaptive_high_only":
                    self.high_mid = nn.Linear(hidden, p)
                    self.high_log_width = nn.Linear(hidden, p)
                    self.high_weight = nn.Linear(hidden, p)
                else:
                    self.mid = nn.Linear(hidden, p)
                    self.log_width = nn.Linear(hidden, p)
                    self.w_low = nn.Linear(hidden, p)
                    self.w_high = nn.Linear(hidden, p)

            def forward(self, x, z):
                h = self.context_net(z)
                if self.variant == "context_only_mlp":
                    logit = self.context_head(h).squeeze(1)
                    empty = torch.zeros_like(x)
                    return logit, empty, empty, empty, empty
                base_logit = self.base(h).squeeze(1)
                if self.variant == "fixed_global_two_sided":
                    mid = self.global_mid.unsqueeze(0).expand_as(x)
                    width = F.softplus(self.global_log_width).unsqueeze(0).expand_as(x) + 0.05
                    c_low, c_high = mid - width, mid + width
                    w_low = F.softplus(self.global_w_low).unsqueeze(0).expand_as(x)
                    w_high = F.softplus(self.global_w_high).unsqueeze(0).expand_as(x)
                    logit = base_logit + torch.sum(w_low * F.relu(c_low - x) + w_high * F.relu(x - c_high), dim=1)
                    return logit, c_low, c_high, w_low, w_high
                if self.variant == "adaptive_high_only":
                    mid = self.high_mid(h)
                    width = F.softplus(self.high_log_width(h)) + 0.05
                    c_high = mid + width
                    w_high = F.softplus(self.high_weight(h))
                    c_low = torch.zeros_like(c_high)
                    w_low = torch.zeros_like(w_high)
                    logit = base_logit + torch.sum(w_high * F.relu(x - c_high), dim=1)
                    return logit, c_low, c_high, w_low, w_high
                mid = self.mid(h)
                width = F.softplus(self.log_width(h)) + 0.05
                c_low, c_high = mid - width, mid + width
                w_low = F.softplus(self.w_low(h))
                w_high = F.softplus(self.w_high(h))
                logit = base_logit + torch.sum(w_low * F.relu(c_low - x) + w_high * F.relu(x - c_high), dim=1)
                return logit, c_low, c_high, w_low, w_high

        model = DeepCutAccounting(x_train.shape[1], z_train.shape[1], variant).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0015, weight_decay=1e-4)
        train_dataset = TensorDataset(
            torch.tensor(x_train), torch.tensor(z_train), torch.tensor(y_train),
        )
        generator = torch.Generator()
        generator.manual_seed(seed)
        loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        x_val_tensor = torch.tensor(x_val, device=device)
        z_val_tensor = torch.tensor(z_val, device=device)
        y_val_tensor = torch.tensor(y_val, device=device)
        x_test_tensor = torch.tensor(x_test, device=device)
        z_test_tensor = torch.tensor(z_test, device=device)
        positives = max(1.0, float(y_train.sum()))
        negatives = max(1.0, float(len(y_train) - y_train.sum()))
        pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
        best_state = None
        best_val_ap = -np.inf
        best_val_loss = np.inf
        patience_left = self.patience
        best_epoch = 0
        for epoch in range(1, self.max_epochs + 1):
            model.train()
            losses = []
            for x_batch, z_batch, y_batch in loader:
                x_batch = x_batch.to(device, non_blocking=True)
                z_batch = z_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits, c_low, c_high, w_low, w_high = model(x_batch, z_batch)
                bce = F.binary_cross_entropy_with_logits(logits, y_batch, pos_weight=pos_weight)
                if variant == "context_only_mlp":
                    regularization = torch.tensor(0.0, device=device)
                else:
                    width_penalty = torch.mean((c_high - c_low) ** 2)
                    weight_penalty = torch.mean(w_low ** 2 + w_high ** 2)
                    regularization = 0.001 * width_penalty + 0.0005 * weight_penalty
                loss = bce + regularization
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            model.eval()
            with torch.no_grad():
                val_logits, _, _, _, _ = model(x_val_tensor, z_val_tensor)
                val_prob = torch.sigmoid(val_logits).detach().cpu().numpy()
                val_loss = float(F.binary_cross_entropy_with_logits(val_logits, y_val_tensor, pos_weight=pos_weight).detach().cpu())
            val_ap = safe_pr_auc(y_val.astype(int), val_prob)
            val_auc = safe_roc_auc(y_val.astype(int), val_prob)
            if epoch == 1 or epoch % 10 == 0:
                log(f"  epoch={epoch:03d} train_loss={np.mean(losses):.4f} val_loss={val_loss:.4f} val_auc={val_auc:.4f} val_pr_auc={val_ap:.4f}")
            improved = (val_ap > best_val_ap + 1e-6) or (
                abs(val_ap - best_val_ap) <= 1e-6 and val_loss < best_val_loss
            )
            if improved:
                best_val_ap, best_val_loss, best_epoch = val_ap, val_loss, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_left = self.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    log(f"  early stopping at epoch {epoch}")
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            val_logits, _, _, _, _ = model(x_val_tensor, z_val_tensor)
            test_logits, c_low, c_high, w_low, w_high = model(x_test_tensor, z_test_tensor)
            p_val = torch.sigmoid(val_logits).detach().cpu().numpy()
            p_test = torch.sigmoid(test_logits).detach().cpu().numpy()
            c_low_np = c_low.detach().cpu().numpy()
            c_high_np = c_high.detach().cpu().numpy()
            w_low_np = w_low.detach().cpu().numpy()
            w_high_np = w_high.detach().cpu().numpy()
        threshold_summary = None
        if variant != "context_only_mlp":
            scaler = arrays["x_scaler"]
            means = np.asarray(scaler.mean_, float)
            scales = np.asarray(scaler.scale_, float)
            threshold_summary = pd.DataFrame({
                "feature": self.threshold_features,
                "mean_low_cut_standardized": np.mean(c_low_np, axis=0),
                "mean_high_cut_standardized": np.mean(c_high_np, axis=0),
                "mean_low_cut_original": np.mean(c_low_np, axis=0) * scales + means,
                "mean_high_cut_original": np.mean(c_high_np, axis=0) * scales + means,
                "mean_low_exceedance_weight": np.mean(w_low_np, axis=0),
                "mean_high_exceedance_weight": np.mean(w_high_np, axis=0),
            })
        del model, train_dataset, loader, x_val_tensor, z_val_tensor, y_val_tensor, x_test_tensor, z_test_tensor
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return {
            "p_val": p_val,
            "p_test": p_test,
            "best_epoch": best_epoch,
            "best_val_pr_auc": best_val_ap,
            "threshold_summary": threshold_summary,
        }


def fit_or_load_deepcut(
    trainer: DeepCutTrainer,
    arrays: dict[str, object],
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    variant: str,
    cache_path: Path,
) -> dict[str, object]:
    threshold_path = cache_path.with_suffix(".thresholds.csv")
    metadata_path = cache_path.with_suffix(".json")
    if cache_path.exists() and metadata_path.exists():
        saved = np.load(cache_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        thresholds = pd.read_csv(threshold_path) if threshold_path.exists() else None
        log(f"Loaded cached DeepCut run: {cache_path.name}")
        return {"p_val": saved["p_val"], "p_test": saved["p_test"], "threshold_summary": thresholds, **metadata}
    result = trainer.fit_arrays(arrays, y_train, y_val, y_test, seed=seed, variant=variant)
    np.savez_compressed(cache_path, p_val=result["p_val"], p_test=result["p_test"])
    metadata = {"best_epoch": int(result["best_epoch"]), "best_val_pr_auc": float(result["best_val_pr_auc"]), "seed": seed, "variant": variant}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if result["threshold_summary"] is not None:
        result["threshold_summary"].to_csv(threshold_path, index=False)
    return result


def calibration_stage(
    paths: Paths,
    df: pd.DataFrame,
    all_features: Sequence[str],
    threshold_features: Sequence[str],
    context_features: Sequence[str],
    max_epochs: int,
    patience: int,
    device: str,
) -> None:
    marker = paths.cache_dir / "calibration.done"
    if marker.exists():
        log("Skipping validation-based calibration (already completed).")
        return
    log("Stage 2/7: validation-based probability calibration for all benchmark models.")
    train = df[df["split"].eq("train")].copy()
    val = df[df["split"].eq("val")].copy()
    test = df[df["split"].eq("test")].copy()
    y_train = train["aaer_label"].to_numpy(int)
    y_val = val["aaer_label"].to_numpy(int)
    y_test = test["aaer_label"].to_numpy(int)
    prediction_frame = test[["cik", "name", "fy_num", "fp", "form", "aaer_label"]].reset_index(drop=True).copy()
    calibration_rows = []
    bin_tables = []

    popular_cache = paths.cache_dir / "popular_validation_test_predictions.csv"
    if popular_cache.exists():
        popular_predictions = pd.read_csv(popular_cache)
        log("Loaded cached popular-model validation/test predictions.")
    else:
        records = {"row": np.arange(max(len(val), len(test)))}
        popular_predictions = pd.DataFrame(records)
        for model_name, model in make_popular_models(RANDOM_STATE).items():
            log(f"Fitting {model_name} for calibration.")
            p_val, p_test = fit_popular_model(model_name, model, train, val, test, all_features)
            popular_predictions.loc[: len(val) - 1, f"val__{sanitize_probability_column(model_name)}"] = p_val
            popular_predictions.loc[: len(test) - 1, f"test__{sanitize_probability_column(model_name)}"] = p_test
        popular_predictions.to_csv(popular_cache, index=False)

    trainer = DeepCutTrainer(threshold_features, context_features, max_epochs=max_epochs, patience=patience, device=device)
    arrays = trainer.prepare(train, val, test)
    deep_cache = paths.cache_dir / f"deepcut_full_seed_{RANDOM_STATE}.npz"
    deep_result = fit_or_load_deepcut(trainer, arrays, y_train, y_val, y_test, RANDOM_STATE, "full_adaptive_two_sided", deep_cache)

    raw_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model_name in make_popular_models(RANDOM_STATE):
        col = sanitize_probability_column(model_name)
        raw_predictions[model_name] = (
            popular_predictions[f"val__{col}"].dropna().to_numpy(float)[: len(val)],
            popular_predictions[f"test__{col}"].dropna().to_numpy(float)[: len(test)],
        )
    raw_predictions["DeepCut-inspired neural threshold"] = (deep_result["p_val"], deep_result["p_test"])

    for model_name, (p_val_raw, p_test_raw) in raw_predictions.items():
        calibrations = fit_probability_calibrator(y_val, p_val_raw, p_test_raw)
        validation_brier = {method: brier_score_loss(y_val, np.clip(values[0], 0, 1)) for method, values in calibrations.items()}
        selected_method = min(validation_brier, key=validation_brier.get)
        for method, (p_val, p_test) in calibrations.items():
            threshold = pick_threshold(y_val, p_val)
            row = metric_row(model_name, y_test, p_test, threshold=threshold, subgroup=f"Calibration={method}")
            row.update({
                "calibration_method": method,
                "selected_by_validation_brier": int(method == selected_method),
                "validation_brier": float(validation_brier[method]),
            })
            calibration_rows.append(row)
            bins = calibration_bins(y_test, p_test, n_bins=10)
            bins.insert(0, "model", model_name)
            bins.insert(1, "calibration_method", method)
            bin_tables.append(bins)
            prediction_frame[f"{sanitize_probability_column(model_name)}__{method}"] = p_test

    pd.DataFrame(calibration_rows).to_csv(paths.tables_dir / "calibrated_model_metrics.csv", index=False)
    pd.concat(bin_tables, ignore_index=True).to_csv(paths.tables_dir / "calibrated_probability_bins.csv", index=False)
    prediction_frame.to_csv(paths.output_dir / "calibrated_test_predictions.csv", index=False)

    # Calibration plots for the primary two models.
    bins_all = pd.concat(bin_tables, ignore_index=True)
    primary = ["Logistic regression", "DeepCut-inspired neural threshold"]
    for model_name in primary:
        fig, ax = plt.subplots(figsize=(6.5, 6))
        model_bins = bins_all[bins_all["model"].eq(model_name)]
        for method in ["raw", "platt", "isotonic"]:
            values = model_bins[model_bins["calibration_method"].eq(method)]
            ax.plot(values["mean_probability"], values["observed_rate"], marker="o", label=method)
        upper = max(0.02, float(model_bins[["mean_probability", "observed_rate"]].to_numpy().max()) * 1.05)
        ax.plot([0, upper], [0, upper], linestyle="--", linewidth=1, label="perfect calibration")
        ax.set_xlim(0, upper)
        ax.set_ylim(0, upper)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed AAER-associated rate")
        ax.set_title(f"Calibration: {model_name}")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        filename = "figS3_calibration_logistic" if model_name.startswith("Logistic") else "figS4_calibration_deepcut"
        fig.savefig(paths.figures_dir / f"{filename}.png", dpi=300, bbox_inches="tight")
        fig.savefig(paths.figures_dir / f"{filename}.pdf", bbox_inches="tight")
        plt.close(fig)

    marker.write_text(now(), encoding="utf-8")
    log("Calibration stage completed.")


def cold_start_stage(
    paths: Paths,
    df: pd.DataFrame,
    all_features: Sequence[str],
    threshold_features: Sequence[str],
    context_features: Sequence[str],
    n_folds: int,
    max_epochs: int,
    patience: int,
    device: str,
    bootstrap_reps: int,
) -> None:
    marker = paths.cache_dir / "cold_start.done"
    if marker.exists():
        log("Skipping firm-disjoint temporal cross-fitting (already completed).")
        return
    log("Stage 3/7: firm-disjoint temporal cross-fitting for cold-start generalization.")
    test_all = df[df["split"].eq("test")].copy()
    firm_outcomes = test_all.groupby("cik")["aaer_label"].max().astype(int)
    firms = firm_outcomes.index.to_numpy()
    stratify = firm_outcomes.to_numpy()
    if int(stratify.sum()) < n_folds:
        n_folds = max(2, int(stratify.sum()))
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    fold_predictions = []
    fold_metrics = []
    for fold, (_, heldout_indices) in enumerate(splitter.split(firms, stratify), start=1):
        heldout_firms = set(firms[heldout_indices])
        train = df[df["split"].eq("train") & ~df["cik"].isin(heldout_firms)].copy()
        val = df[df["split"].eq("val") & ~df["cik"].isin(heldout_firms)].copy()
        test = df[df["split"].eq("test") & df["cik"].isin(heldout_firms)].copy()
        y_train = train["aaer_label"].to_numpy(int)
        y_val = val["aaer_label"].to_numpy(int)
        y_test = test["aaer_label"].to_numpy(int)
        log(
            f"Cold-start fold {fold}/{n_folds}: held-out firms={len(heldout_firms):,}, "
            f"test rows={len(test):,}, test positives={int(y_test.sum()):,}"
        )
        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2 or len(np.unique(y_test)) < 2:
            raise RuntimeError(f"Cold-start fold {fold} lacks both outcome classes. Reduce --cold-start-folds.")
        fold_frame = test[["cik", "name", "fy_num", "fp", "form", "aaer_label"]].reset_index(drop=True).copy()
        p_val_log, p_test_log = fit_popular_model(
            "Logistic regression", make_popular_models(RANDOM_STATE + fold)["Logistic regression"],
            train, val, test, all_features,
        )
        fold_frame["prob_Logistic_regression"] = p_test_log
        fold_metrics.append(metric_row(
            "Logistic regression", y_test, p_test_log,
            threshold=pick_threshold(y_val, p_val_log), subgroup=f"Cold-start fold {fold}",
        ))
        trainer = DeepCutTrainer(threshold_features, context_features, max_epochs=max_epochs, patience=patience, device=device)
        arrays = trainer.prepare(train, val, test)
        cache = paths.cache_dir / f"cold_start_fold_{fold}_deepcut.npz"
        result = fit_or_load_deepcut(
            trainer, arrays, y_train, y_val, y_test,
            seed=RANDOM_STATE + fold, variant="full_adaptive_two_sided", cache_path=cache,
        )
        fold_frame["prob_DeepCut_inspired_neural_threshold"] = result["p_test"]
        fold_metrics.append(metric_row(
            "DeepCut-inspired neural threshold", y_test, result["p_test"],
            threshold=pick_threshold(y_val, result["p_val"]), subgroup=f"Cold-start fold {fold}",
        ))
        fold_frame["cold_start_fold"] = fold
        fold_predictions.append(fold_frame)
        del train, val, test, arrays
        gc.collect()
    pooled = pd.concat(fold_predictions, ignore_index=True)
    pooled.to_csv(paths.output_dir / "cold_start_firm_disjoint_test_predictions.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(paths.tables_dir / "cold_start_fold_metrics.csv", index=False)
    pooled_rows = []
    y = pooled["aaer_label"].to_numpy(int)
    for col, model_name in [
        ("prob_Logistic_regression", "Logistic regression"),
        ("prob_DeepCut_inspired_neural_threshold", "DeepCut-inspired neural threshold"),
    ]:
        pooled_rows.append(metric_row(model_name, y, pooled[col].to_numpy(float), subgroup="Pooled firm-disjoint temporal test"))
    pd.DataFrame(pooled_rows).to_csv(paths.tables_dir / "cold_start_pooled_metrics.csv", index=False)

    # Cluster bootstrap on pooled cold-start predictions.
    firm_codes, unique_firms = pd.factorize(pooled["cik"], sort=True)
    rng = np.random.default_rng(RANDOM_STATE + 99)
    cols = ["prob_Logistic_regression", "prob_DeepCut_inspired_neural_threshold"]
    boot = {col: {m: [] for m in ["roc_auc", "pr_auc", "recall_top_0.05", "recall_top_0.1"]} for col in cols}
    for _ in range(bootstrap_reps):
        counts = rng.multinomial(len(unique_firms), np.full(len(unique_firms), 1 / len(unique_firms)))
        weights = counts[firm_codes].astype(float)
        if np.sum(weights * y) <= 0 or np.sum(weights * (1 - y)) <= 0:
            continue
        for col in cols:
            p = pooled[col].to_numpy(float)
            boot[col]["roc_auc"].append(safe_roc_auc(y, p, weights))
            boot[col]["pr_auc"].append(safe_pr_auc(y, p, weights))
            boot[col]["recall_top_0.05"].append(weighted_topk(y, p, 0.05, weights)["recall"])
            boot[col]["recall_top_0.1"].append(weighted_topk(y, p, 0.10, weights)["recall"])
    ci_rows = []
    for col in cols:
        model_name = MODEL_DISPLAY[col]
        p = pooled[col].to_numpy(float)
        points = {
            "roc_auc": safe_roc_auc(y, p), "pr_auc": safe_pr_auc(y, p),
            "recall_top_0.05": weighted_topk(y, p, 0.05)["recall"],
            "recall_top_0.1": weighted_topk(y, p, 0.10)["recall"],
        }
        for metric, values in boot[col].items():
            arr = np.asarray(values, float)
            ci_rows.append({
                "model": model_name, "metric": metric, "estimate": points[metric],
                "ci_lower": float(np.quantile(arr, 0.025)), "ci_upper": float(np.quantile(arr, 0.975)),
                "bootstrap_reps": len(arr),
            })
    pd.DataFrame(ci_rows).to_csv(paths.tables_dir / "cold_start_clustered_confidence_intervals.csv", index=False)
    marker.write_text(now(), encoding="utf-8")
    log("Firm-disjoint temporal cross-fitting completed.")


def repeated_seed_and_ablation_stage(
    paths: Paths,
    df: pd.DataFrame,
    threshold_features: Sequence[str],
    context_features: Sequence[str],
    full_seeds: Sequence[int],
    ablation_seeds: Sequence[int],
    max_epochs: int,
    patience: int,
    device: str,
) -> None:
    marker = paths.cache_dir / "seeds_ablations.done"
    if marker.exists():
        log("Skipping repeated-seed and ablation analysis (already completed).")
        return
    log("Stage 4/7: repeated-seed DeepCut stability and structural ablations.")
    train = df[df["split"].eq("train")].copy()
    val = df[df["split"].eq("val")].copy()
    test = df[df["split"].eq("test")].copy()
    y_train = train["aaer_label"].to_numpy(int)
    y_val = val["aaer_label"].to_numpy(int)
    y_test = test["aaer_label"].to_numpy(int)
    trainer = DeepCutTrainer(threshold_features, context_features, max_epochs=max_epochs, patience=patience, device=device)
    arrays = trainer.prepare(train, val, test)
    plan = [("full_adaptive_two_sided", seed) for seed in full_seeds]
    for variant in ["fixed_global_two_sided", "adaptive_high_only", "context_only_mlp"]:
        plan.extend((variant, seed) for seed in ablation_seeds)
    rows = []
    threshold_tables = []
    for variant, seed in plan:
        cache = paths.cache_dir / f"deepcut_{variant}_seed_{seed}.npz"
        if variant == "full_adaptive_two_sided" and seed == RANDOM_STATE:
            primary_cache = paths.cache_dir / f"deepcut_full_seed_{RANDOM_STATE}.npz"
            if primary_cache.exists() and not cache.exists():
                cache = primary_cache
        result = fit_or_load_deepcut(trainer, arrays, y_train, y_val, y_test, seed, variant, cache)
        threshold = pick_threshold(y_val, result["p_val"])
        row = metric_row(variant, y_test, result["p_test"], threshold=threshold, subgroup="Temporal test")
        row.update({"variant": variant, "seed": seed, "best_epoch": result.get("best_epoch"), "best_val_pr_auc": result.get("best_val_pr_auc")})
        rows.append(row)
        if variant == "full_adaptive_two_sided" and result.get("threshold_summary") is not None:
            table = result["threshold_summary"].copy()
            table["seed"] = seed
            threshold_tables.append(table)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(paths.tables_dir / "deepcut_seed_and_ablation_metrics.csv", index=False)
    summary = metrics.groupby("variant").agg(
        seeds=("seed", "nunique"),
        roc_auc_mean=("roc_auc", "mean"), roc_auc_sd=("roc_auc", "std"),
        pr_auc_mean=("pr_auc", "mean"), pr_auc_sd=("pr_auc", "std"),
        recall_top_0p05_mean=("recall_top_0p05", "mean"), recall_top_0p05_sd=("recall_top_0p05", "std"),
        recall_top_0p1_mean=("recall_top_0p1", "mean"), recall_top_0p1_sd=("recall_top_0p1", "std"),
    ).reset_index()
    summary.to_csv(paths.tables_dir / "deepcut_seed_and_ablation_summary.csv", index=False)
    if threshold_tables:
        threshold_all = pd.concat(threshold_tables, ignore_index=True)
        threshold_all.to_csv(paths.tables_dir / "deepcut_thresholds_across_seeds.csv", index=False)
        threshold_stability = threshold_all.groupby("feature").agg(
            seeds=("seed", "nunique"),
            low_cut_mean=("mean_low_cut_original", "mean"), low_cut_sd=("mean_low_cut_original", "std"),
            high_cut_mean=("mean_high_cut_original", "mean"), high_cut_sd=("mean_high_cut_original", "std"),
            low_weight_mean=("mean_low_exceedance_weight", "mean"), low_weight_sd=("mean_low_exceedance_weight", "std"),
            high_weight_mean=("mean_high_exceedance_weight", "mean"), high_weight_sd=("mean_high_exceedance_weight", "std"),
        ).reset_index()
        threshold_stability.to_csv(paths.tables_dir / "deepcut_threshold_stability_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    order = summary.sort_values("pr_auc_mean")
    positions = np.arange(len(order))
    ax.errorbar(order["pr_auc_mean"], positions, xerr=order["pr_auc_sd"].fillna(0), fmt="o", capsize=4)
    ax.set_yticks(positions, order["variant"].str.replace("_", " "))
    ax.set_xlabel("Test PR-AUC (mean ± SD across seeds)")
    ax.set_title("DeepCut repeated-seed stability and structural ablations")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "figS5_deepcut_ablation_seed_stability.png", dpi=300, bbox_inches="tight")
    fig.savefig(paths.figures_dir / "figS5_deepcut_ablation_seed_stability.pdf", bbox_inches="tight")
    plt.close(fig)
    marker.write_text(now(), encoding="utf-8")
    log("Repeated-seed and ablation analysis completed.")


def ten_k_only_stage(
    paths: Paths,
    df: pd.DataFrame,
    all_features: Sequence[str],
    threshold_features: Sequence[str],
    context_features: Sequence[str],
    max_epochs: int,
    patience: int,
    device: str,
) -> None:
    marker = paths.cache_dir / "ten_k_only.done"
    if marker.exists():
        log("Skipping 10-K-only retraining (already completed).")
        return
    log("Stage 5/7: 10-K-only retraining and temporal test robustness.")
    tenk = df[df["form"].eq("10-K")].copy()
    train = tenk[tenk["split"].eq("train")].copy()
    val = tenk[tenk["split"].eq("val")].copy()
    test = tenk[tenk["split"].eq("test")].copy()
    y_train, y_val, y_test = (x["aaer_label"].to_numpy(int) for x in (train, val, test))
    p_val_log, p_test_log = fit_popular_model(
        "Logistic regression", make_popular_models(RANDOM_STATE)["Logistic regression"],
        train, val, test, all_features,
    )
    rows = [metric_row(
        "Logistic regression", y_test, p_test_log, pick_threshold(y_val, p_val_log), subgroup="10-K-only training and test",
    )]
    trainer = DeepCutTrainer(threshold_features, context_features, max_epochs=max_epochs, patience=patience, device=device)
    arrays = trainer.prepare(train, val, test)
    result = fit_or_load_deepcut(
        trainer, arrays, y_train, y_val, y_test, RANDOM_STATE,
        "full_adaptive_two_sided", paths.cache_dir / "ten_k_only_deepcut.npz",
    )
    rows.append(metric_row(
        "DeepCut-inspired neural threshold", y_test, result["p_test"],
        pick_threshold(y_val, result["p_val"]), subgroup="10-K-only training and test",
    ))
    pd.DataFrame(rows).to_csv(paths.tables_dir / "ten_k_only_retrained_metrics.csv", index=False)
    prediction = test[["cik", "name", "fy_num", "fp", "form", "aaer_label"]].reset_index(drop=True).copy()
    prediction["prob_Logistic_regression"] = p_test_log
    prediction["prob_DeepCut_inspired_neural_threshold"] = result["p_test"]
    prediction.to_csv(paths.output_dir / "ten_k_only_test_predictions.csv", index=False)
    marker.write_text(now(), encoding="utf-8")
    log("10-K-only retraining completed.")


def labels_for_window(df: pd.DataFrame, reviewed_labels: pd.DataFrame, window: int) -> np.ndarray:
    accepted = reviewed_labels[pd.to_numeric(reviewed_labels["keep_label"], errors="coerce").eq(1)].copy()
    accepted["cik"] = accepted["cik"].astype(str)
    accepted["aaer_year"] = pd.to_numeric(accepted["aaer_year"], errors="coerce").astype("Int64")
    years_by_cik = accepted.dropna(subset=["aaer_year"]).groupby("cik")["aaer_year"].apply(lambda x: tuple(int(v) for v in x)).to_dict()
    out = np.zeros(len(df), dtype="int8")
    for i, (cik, fiscal_year) in enumerate(zip(df["cik"].astype(str), df["fy_num"])):
        if pd.isna(fiscal_year):
            continue
        fiscal_year = int(fiscal_year)
        out[i] = int(any(fiscal_year <= year <= fiscal_year + window for year in years_by_cik.get(cik, ())))
    return out


def label_window_stage(
    paths: Paths,
    df: pd.DataFrame,
    all_features: Sequence[str],
    threshold_features: Sequence[str],
    context_features: Sequence[str],
    max_epochs: int,
    patience: int,
    device: str,
) -> None:
    marker = paths.cache_dir / "label_windows.done"
    if marker.exists():
        log("Skipping AAER label-window sensitivity (already completed).")
        return
    log("Stage 6/7: AAER label-window sensitivity (0, 1, 2, and 3 years).")
    reviewed = pd.read_csv(paths.reviewed_labels, low_memory=False)
    train_mask = df["split"].eq("train").to_numpy()
    val_mask = df["split"].eq("val").to_numpy()
    test_mask = df["split"].eq("test").to_numpy()
    train, val, test = df[train_mask].copy(), df[val_mask].copy(), df[test_mask].copy()
    trainer = DeepCutTrainer(threshold_features, context_features, max_epochs=max_epochs, patience=patience, device=device)
    arrays = trainer.prepare(train, val, test)
    rows = []
    prevalence_rows = []
    for window in (0, 1, 2, 3):
        labels = labels_for_window(df, reviewed, window)
        y_train, y_val, y_test = labels[train_mask], labels[val_mask], labels[test_mask]
        prevalence_rows.extend([
            {"label_window_years": window, "split": "train", "rows": len(y_train), "positives": int(y_train.sum()), "prevalence": float(y_train.mean())},
            {"label_window_years": window, "split": "val", "rows": len(y_val), "positives": int(y_val.sum()), "prevalence": float(y_val.mean())},
            {"label_window_years": window, "split": "test", "rows": len(y_test), "positives": int(y_test.sum()), "prevalence": float(y_test.mean())},
        ])
        log(f"  label window={window}: train/val/test positives={int(y_train.sum())}/{int(y_val.sum())}/{int(y_test.sum())}")
        if any(len(np.unique(y)) < 2 for y in (y_train, y_val, y_test)):
            log(f"  skipping model fitting for window={window}; one split lacks both classes")
            continue
        p_val_log, p_test_log = fit_popular_model(
            "Logistic regression", make_popular_models(RANDOM_STATE + window)["Logistic regression"],
            train, val, test, all_features, y_train=y_train, y_val=y_val, y_test=y_test,
        )
        row = metric_row("Logistic regression", y_test, p_test_log, pick_threshold(y_val, p_val_log), subgroup=f"AAER window={window}")
        row["label_window_years"] = window
        rows.append(row)
        if window == 3:
            cache = paths.cache_dir / f"deepcut_full_seed_{RANDOM_STATE}.npz"
        else:
            cache = paths.cache_dir / f"label_window_{window}_deepcut.npz"
        result = fit_or_load_deepcut(
            trainer, arrays, y_train, y_val, y_test, RANDOM_STATE + window,
            "full_adaptive_two_sided", cache,
        )
        row = metric_row(
            "DeepCut-inspired neural threshold", y_test, result["p_test"],
            pick_threshold(y_val, result["p_val"]), subgroup=f"AAER window={window}",
        )
        row["label_window_years"] = window
        rows.append(row)
    pd.DataFrame(prevalence_rows).to_csv(paths.tables_dir / "label_window_split_prevalence.csv", index=False)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(paths.tables_dir / "label_window_sensitivity_metrics.csv", index=False)
    if not metrics.empty:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for model, group in metrics.groupby("model"):
            ax.plot(group["label_window_years"], group["pr_auc"], marker="o", label=model)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xlabel("Forward AAER label window (years)")
        ax.set_ylabel("Temporal-test PR-AUC")
        ax.set_title("Sensitivity to AAER labeling window")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(paths.figures_dir / "figS6_label_window_sensitivity.png", dpi=300, bbox_inches="tight")
        fig.savefig(paths.figures_dir / "figS6_label_window_sensitivity.pdf", bbox_inches="tight")
        plt.close(fig)
    marker.write_text(now(), encoding="utf-8")
    log("AAER label-window sensitivity completed.")


def generate_report(paths: Paths) -> None:
    log("Stage 7/7: generating publication-strengthening report and output manifest.")
    tables = {path.stem: pd.read_csv(path) for path in paths.tables_dir.glob("*.csv")}
    lines = [
        "# SEC–AAER Publication-Strengthening Analysis",
        "",
        f"Generated: {now()}",
        "",
        "## Analyses completed",
        "",
        "- Firm-clustered bootstrap confidence intervals and paired model differences.",
        "- Raw and validation-based Platt/isotonic calibration diagnostics.",
        "- 10-K, 10-Q, year-specific, and naturally unseen-firm subgroup diagnostics.",
        "- Firm-disjoint temporal cross-fitting for cold-start generalization.",
        "- DeepCut repeated-seed stability and structural ablations.",
        "- 10-K-only model retraining.",
        "- AAER label-window sensitivity for 0–3 future years.",
        "",
    ]
    subgroup = tables.get("subgroup_robustness_metrics")
    if subgroup is not None:
        unseen = subgroup[subgroup["subgroup"].eq("Firms absent from the training period")]
        if not unseen.empty and int(unseen.iloc[0]["positives"]) == 0:
            lines.extend([
                "## Important unseen-firm finding",
                "",
                "The naturally occurring test subset of firms absent from the training period contains no positive AAER-associated observations. Standard ROC-AUC and PR-AUC are therefore undefined for that subset. The firm-disjoint temporal cross-fitting analysis is the valid cold-start generalization result and should be reported instead.",
                "",
            ])
    calibrated = tables.get("calibrated_model_metrics")
    if calibrated is not None:
        selected = calibrated[calibrated["selected_by_validation_brier"].eq(1)].copy()
        lines.extend(["## Validation-selected calibration", ""])
        for _, row in selected.sort_values("model").iterrows():
            lines.append(
                f"- **{row['model']}**: {row['calibration_method']} calibration; test ROC-AUC {row['roc_auc']:.4f}, PR-AUC {row['pr_auc']:.5f}, Brier {row['brier']:.6f}, ECE {row['ece_10']:.6f}."
            )
        lines.append("")
    cold = tables.get("cold_start_pooled_metrics")
    if cold is not None:
        lines.extend(["## Firm-disjoint temporal cross-fitting", ""])
        for _, row in cold.iterrows():
            lines.append(
                f"- **{row['model']}**: ROC-AUC {row['roc_auc']:.4f}, PR-AUC {row['pr_auc']:.5f}, top-5% recall {row['recall_top_0p05']:.4f}."
            )
        lines.append("")
    ablation = tables.get("deepcut_seed_and_ablation_summary")
    if ablation is not None:
        lines.extend(["## DeepCut stability and ablations", ""])
        for _, row in ablation.iterrows():
            lines.append(
                f"- **{str(row['variant']).replace('_', ' ')}** ({int(row['seeds'])} seeds): ROC-AUC {row['roc_auc_mean']:.4f} ± {row['roc_auc_sd'] if pd.notna(row['roc_auc_sd']) else 0:.4f}; PR-AUC {row['pr_auc_mean']:.5f} ± {row['pr_auc_sd'] if pd.notna(row['pr_auc_sd']) else 0:.5f}."
            )
        lines.append("")
    lines.extend([
        "## Submission caution",
        "",
        "These analyses strengthen statistical validation, but the manuscript must be revised to use the corrected 904-positive dataset and the new uncertainty, calibration, cold-start, ablation, and sensitivity results. The original manuscript numbers should not be retained.",
        "",
    ])
    (paths.output_dir / "PUBLICATION_STRENGTHENING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = []
    for path in sorted(paths.output_dir.rglob("*")):
        if path.is_file():
            manifest.append({"relative_path": str(path.relative_to(paths.output_dir)), "bytes": path.stat().st_size})
    pd.DataFrame(manifest).to_csv(paths.output_dir / "output_manifest.csv", index=False)
    log("Report generated.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--cold-start-folds", type=int, default=5)
    parser.add_argument("--full-seeds", default="20260531,20260532,20260533,20260534,20260535")
    parser.add_argument("--ablation-seeds", default="20260531,20260532,20260533")
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument("--fast", action="store_true", help="Small test run; not for manuscript reporting.")
    args = parser.parse_args()
    if args.fast:
        args.bootstrap_reps = min(args.bootstrap_reps, 20)
        args.cold_start_folds = 2
        args.full_seeds = "20260531"
        args.ablation_seeds = "20260531"
        args.max_epochs = min(args.max_epochs, 3)
        args.patience = min(args.patience, 2)
    paths = resolve_paths(args.project_root)
    config = vars(args).copy()
    config["project_root"] = str(args.project_root)
    (paths.output_dir / "run_configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    log(f"Project root: {paths.root}")
    log(f"Output directory: {paths.output_dir}")
    existing_prediction_diagnostics(paths, args.bootstrap_reps, RANDOM_STATE)
    if args.diagnostics_only:
        generate_report(paths)
        return 0
    df, all_features, threshold_features, context_features = load_data(paths)
    calibration_stage(paths, df, all_features, threshold_features, context_features, args.max_epochs, args.patience, args.device)
    cold_start_stage(paths, df, all_features, threshold_features, context_features, args.cold_start_folds, args.max_epochs, args.patience, args.device, max(200, args.bootstrap_reps // 2))
    repeated_seed_and_ablation_stage(paths, df, threshold_features, context_features, parse_seed_list(args.full_seeds), parse_seed_list(args.ablation_seeds), args.max_epochs, args.patience, args.device)
    ten_k_only_stage(paths, df, all_features, threshold_features, context_features, args.max_epochs, args.patience, args.device)
    label_window_stage(paths, df, all_features, threshold_features, context_features, args.max_epochs, args.patience, args.device)
    generate_report(paths)
    log("All publication-strengthening analyses completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("Interrupted by user. Completed stages are cached and the run can be restarted.")
        raise
    except Exception:
        traceback.print_exc()
        raise
