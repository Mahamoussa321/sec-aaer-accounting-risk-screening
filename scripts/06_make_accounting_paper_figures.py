#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_accounting_paper_figures.py

Creates publication-ready figures for the accounting-journal version:
SEC 10-K/10-Q + AAER enforcement-risk screening with popular ML models
and a DeepCut-inspired neural threshold model.

Expected project folder:
    C:\\Users\\maham

Expected inputs:
    outputs\\accounting_deepcut_ml_benchmark\\model_comparison_summary.csv
    outputs\\accounting_deepcut_ml_benchmark\\topk_screening_metrics.csv
    outputs\\accounting_deepcut_ml_benchmark\\test_predictions.csv
    outputs\\accounting_deepcut_ml_benchmark\\deepcut_threshold_summary.csv  optional

Outputs:
    outputs\\accounting_paper_figures\\*.png
    outputs\\accounting_paper_figures\\*.pdf
    outputs\\accounting_paper_figures\\figure_captions.txt

Run:
    cd C:\\Users\\maham
    python .\\make_accounting_paper_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, precision_recall_curve, auc


# =============================================================================
# Paths
# =============================================================================

ROOT = Path(".")
RESULT_DIR = ROOT / "outputs" / "accounting_deepcut_ml_benchmark"
ALT_DATASET = ROOT / "outputs" / "aaer_enhanced_accounting_benchmark" / "enhanced_10k_10q_analysis_dataset.csv"
FIG_DIR = ROOT / "outputs" / "accounting_paper_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SUMMARY_FILE = RESULT_DIR / "model_comparison_summary.csv"
TOPK_FILE = RESULT_DIR / "topk_screening_metrics.csv"
PRED_FILE = RESULT_DIR / "test_predictions.csv"
THRESHOLD_FILE = RESULT_DIR / "deepcut_threshold_summary.csv"
DESIGN_FILE = RESULT_DIR / "analysis_design.json"


# =============================================================================
# Style
# =============================================================================

COLORS = {
    "navy": "#0B3C88",
    "blue": "#1557B0",
    "teal": "#007A78",
    "teal_light": "#E9F6F4",
    "blue_light": "#EDF4FF",
    "orange": "#E65A0C",
    "orange_light": "#FFF3E8",
    "gray": "#4B5563",
    "light_gray": "#E5E7EB",
    "dark": "#111827",
    "white": "#FFFFFF",
}

MODEL_COLORS = {
    "Logistic regression": COLORS["blue"],
    "Random forest": "#5177B8",
    "Extra trees": "#8A9CCB",
    "Gradient boosting": COLORS["teal"],
    "HistGradientBoosting": "#5AA6A4",
    "DeepCut-inspired neural threshold": COLORS["orange"],
}

MODEL_ORDER = [
    "Logistic regression",
    "Random forest",
    "Extra trees",
    "Gradient boosting",
    "HistGradientBoosting",
    "DeepCut-inspired neural threshold",
]

PRED_COLS = {
    "Logistic regression": "prob_Logistic_regression",
    "Random forest": "prob_Random_forest",
    "Extra trees": "prob_Extra_trees",
    "Gradient boosting": "prob_Gradient_boosting",
    "HistGradientBoosting": "prob_HistGradientBoosting",
    "DeepCut-inspired neural threshold": "prob_DeepCut_inspired_neural_threshold",
}


def set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": COLORS["dark"],
        "axes.linewidth": 0.8,
    })


def save_fig(fig, name: str):
    png = FIG_DIR / f"{name}.png"
    pdf = FIG_DIR / f"{name}.pdf"
    fig.savefig(png, bbox_inches="tight", dpi=300)
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def add_box(ax, xy, wh, text, fc="white", ec=None, lw=1.2, fontsize=10,
            color=None, ha="center", va="center", weight="normal", radius=0.02):
    if ec is None:
        ec = COLORS["blue"]
    if color is None:
        color = COLORS["dark"]
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.018,rounding_size={radius}",
        fc=fc, ec=ec, lw=lw
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha=ha, va=va, fontsize=fontsize,
            color=color, weight=weight, wrap=True)
    return patch


def arrow(ax, start, end, color=None, lw=1.4, mutation_scale=12):
    if color is None:
        color = COLORS["blue"]
    arr = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=mutation_scale,
        lw=lw, color=color, shrinkA=3, shrinkB=3
    )
    ax.add_patch(arr)
    return arr


# =============================================================================
# Load data
# =============================================================================

def load_inputs():
    if not MODEL_SUMMARY_FILE.exists():
        raise FileNotFoundError(f"Missing {MODEL_SUMMARY_FILE}")
    if not TOPK_FILE.exists():
        raise FileNotFoundError(f"Missing {TOPK_FILE}")
    if not PRED_FILE.exists():
        raise FileNotFoundError(f"Missing {PRED_FILE}")

    summary = pd.read_csv(MODEL_SUMMARY_FILE)
    topk = pd.read_csv(TOPK_FILE)
    pred = pd.read_csv(PRED_FILE)

    # Keep desired model order when present
    summary["model"] = pd.Categorical(summary["model"], categories=MODEL_ORDER, ordered=True)
    summary = summary.sort_values("model").reset_index(drop=True)

    return summary, topk, pred


def get_split_counts():
    """
    Reads the saved analysis dataset if available. Otherwise uses the final run counts.
    """
    candidates = [
        RESULT_DIR / "accounting_deepcut_analysis_dataset.csv",
        ALT_DATASET,
    ]
    for p in candidates:
        if p.exists():
            try:
                df = pd.read_csv(p, usecols=lambda c: c in ["split", "aaer_label", "cik", "form"], low_memory=False)
                tab = pd.crosstab(df["split"], df["aaer_label"])
                out = {}
                for split in ["train", "val", "test"]:
                    if split in tab.index:
                        n0 = int(tab.loc[split].get(0, 0))
                        n1 = int(tab.loc[split].get(1, 0))
                        out[split] = {"rows": n0 + n1, "positives": n1}
                out["all"] = {"rows": int(len(df)), "positives": int(df["aaer_label"].sum())}
                out["firms"] = int(df["cik"].nunique()) if "cik" in df.columns else 13879
                return out
            except Exception:
                pass

    # Fallback from your final run
    return {
        "train": {"rows": 205029, "positives": 1826},
        "val": {"rows": 44636, "positives": 479},
        "test": {"rows": 74362, "positives": 584},
        "all": {"rows": 324027, "positives": 2889},
        "firms": 13879,
    }


# =============================================================================
# Figure 1: workflow schematic
# =============================================================================

def make_workflow_figure():
    counts = get_split_counts()

    fig, ax = plt.subplots(figsize=(13.5, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Input blocks
    add_box(ax, (0.02, 0.62), (0.16, 0.23),
            "SEC Financial\nStatement Data Sets\n\n10-K and 10-Q filings",
            fc="white", ec=COLORS["navy"], color=COLORS["navy"], fontsize=10, weight="bold")
    add_box(ax, (0.02, 0.30), (0.16, 0.23),
            "SEC AAER Releases\n\nAccounting and Auditing\nEnforcement Releases",
            fc="white", ec=COLORS["navy"], color=COLORS["navy"], fontsize=10, weight="bold")

    # Dataset block
    dataset_text = (
        "Matched filing-period dataset\n\n"
        f"✓ {counts['all']['rows']:,} observations\n"
        f"✓ {counts['firms']:,} firms\n"
        f"✓ {counts['all']['positives']:,} AAER-positive observations"
    )
    add_box(ax, (0.25, 0.43), (0.16, 0.28), dataset_text,
            fc=COLORS["teal_light"], ec=COLORS["teal"], color=COLORS["dark"], fontsize=9.5, weight="normal")

    # Feature block
    add_box(ax, (0.46, 0.43), (0.14, 0.28),
            "Feature engineering\n\nfinancial ratios\naccruals\ngrowth\nBeneish-style indices\nindustry controls",
            fc="white", ec=COLORS["blue"], color=COLORS["dark"], fontsize=9.2)

    # Split block
    split_text = (
        "Temporal split\n\n"
        f"Train 2009–2018\n{counts['train']['rows']:,} rows, {counts['train']['positives']:,} positives\n\n"
        f"Validation 2019–2020\n{counts['val']['rows']:,} rows, {counts['val']['positives']:,} positives\n\n"
        f"Test 2021–2023\n{counts['test']['rows']:,} rows, {counts['test']['positives']:,} positives"
    )
    add_box(ax, (0.65, 0.37), (0.17, 0.40), split_text,
            fc=COLORS["blue_light"], ec=COLORS["blue"], color=COLORS["dark"], fontsize=8.6)

    # Model block
    model_text = (
        "Model comparison\n\n"
        "Logistic regression\n"
        "Random forest\n"
        "Extra trees\n"
        "Gradient boosting\n"
        "HistGradientBoosting\n"
        "DeepCut-inspired neural thresholding"
    )
    add_box(ax, (0.84, 0.36), (0.13, 0.42), model_text,
            fc="white", ec=COLORS["navy"], color=COLORS["dark"], fontsize=8.2)

    # Orange DeepCut emphasis inside model block
    add_box(ax, (0.855, 0.375), (0.10, 0.055), "DeepCut-inspired\nthresholding",
            fc=COLORS["orange_light"], ec=COLORS["orange"], color=COLORS["orange"], fontsize=7.4, lw=1.0)

    # Evaluation block at bottom right
    add_box(ax, (0.76, 0.11), (0.20, 0.13),
            "Evaluation: ROC-AUC, PR-AUC, F1-score,\nTop-k screening lift",
            fc=COLORS["teal_light"], ec=COLORS["teal"], color=COLORS["dark"], fontsize=9.5)

    # Rare-event objective callout
    add_box(ax, (0.38, 0.12), (0.30, 0.10),
            "Rare-event screening objective:\nprioritize high-risk filings for review",
            fc=COLORS["orange_light"], ec=COLORS["orange"], color=COLORS["dark"], fontsize=10)

    # Connectors
    arrow(ax, (0.18, 0.73), (0.25, 0.58), color=COLORS["blue"])
    arrow(ax, (0.18, 0.41), (0.25, 0.55), color=COLORS["blue"])
    arrow(ax, (0.41, 0.57), (0.46, 0.57), color=COLORS["blue"])
    arrow(ax, (0.60, 0.57), (0.65, 0.57), color=COLORS["blue"])
    arrow(ax, (0.82, 0.57), (0.84, 0.57), color=COLORS["blue"])
    arrow(ax, (0.90, 0.36), (0.86, 0.24), color=COLORS["teal"])

    save_fig(fig, "fig1_workflow_sec_aaer_screening")


# =============================================================================
# Figure 2: temporal split
# =============================================================================

def make_temporal_split_figure():
    counts = get_split_counts()
    test_prev = counts["test"]["positives"] / counts["test"]["rows"] * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(2008.5, 2023.9)
    ax.set_ylim(-0.35, 1.25)
    ax.axis("off")

    y = 0.5
    # Main timeline
    ax.plot([2009, 2018], [y, y], lw=5, color=COLORS["blue"], solid_capstyle="round")
    ax.plot([2018, 2020], [y, y], lw=5, color=COLORS["teal"], solid_capstyle="round")
    ax.plot([2020, 2023], [y, y], lw=5, color=COLORS["blue"], solid_capstyle="round")
    arrow(ax, (2023.1, y), (2023.7, y), color=COLORS["dark"], lw=1.0, mutation_scale=14)

    for yr in range(2009, 2024):
        ax.plot([yr, yr], [y - 0.04, y + 0.04], color=COLORS["dark"], lw=0.7)
        ax.text(yr, y - 0.10, str(yr), ha="center", va="top", fontsize=9)

    # Dots and callouts
    callouts = [
        (2014, COLORS["blue"], f"{counts['train']['rows']:,} rows\n{counts['train']['positives']:,} positives"),
        (2019, COLORS["teal"], f"{counts['val']['rows']:,} rows\n{counts['val']['positives']:,} positives"),
        (2022, COLORS["blue"], f"{counts['test']['rows']:,} rows\n{counts['test']['positives']:,} positives"),
    ]
    for x, col, txt in callouts:
        ax.add_patch(Circle((x, y), 0.10, fc=col, ec="white", lw=1.2, zorder=5))
        ax.plot([x, x], [y, 0.95], color=col, lw=1)
        add_box(ax, (x - 1.25, 0.95), (2.5, 0.18), txt, fc="white", ec=col, color=COLORS["dark"], fontsize=10)

    # Period labels
    add_box(ax, (2010.5, 0.08), (6.0, 0.16), "Train: 2009–2018",
            fc="white", ec=COLORS["blue"], color=COLORS["blue"], fontsize=11, weight="bold")
    add_box(ax, (2017.9, 0.08), (2.8, 0.16), "Validation: 2019–2020",
            fc="white", ec=COLORS["teal"], color=COLORS["teal"], fontsize=11, weight="bold")
    add_box(ax, (2020.7, 0.08), (2.8, 0.16), "Test: 2021–2023",
            fc="white", ec=COLORS["blue"], color=COLORS["blue"], fontsize=11, weight="bold")

    # Summary box
    summary = (
        f"{counts['all']['rows']:,} filing-period observations\n"
        f"{counts['firms']:,} firms\n"
        f"{counts['all']['positives']:,} AAER-positive observations"
    )
    add_box(ax, (2009.0, 1.05), (3.0, 0.16), summary,
            fc=COLORS["teal_light"], ec=COLORS["teal"], color=COLORS["dark"], fontsize=9.8)

    add_box(ax, (2019.0, -0.22), (4.3, 0.15),
            f"Test prevalence = {test_prev:.3f}%",
            fc=COLORS["orange_light"], ec=COLORS["orange"], color=COLORS["dark"], fontsize=11)

    add_box(ax, (2009.0, -0.22), (4.4, 0.15),
            "Rare-event temporal evaluation",
            fc=COLORS["orange_light"], ec=COLORS["orange"], color=COLORS["orange"], fontsize=11, weight="bold")

    save_fig(fig, "fig2_temporal_split_dataset_summary")


# =============================================================================
# Model metric bars
# =============================================================================

def make_metric_bar(summary: pd.DataFrame, metric: str, filename: str, xlabel: str, xmax: float | None = None):
    df = summary.copy()
    df = df[df["model"].isin(MODEL_ORDER)].copy()
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER[::-1], ordered=True)
    df = df.sort_values("model")

    values = df[metric].astype(float).values
    models = df["model"].astype(str).values
    colors = [MODEL_COLORS.get(m, COLORS["blue"]) for m in models]
    best_idx = int(np.nanargmax(values))

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    bars = ax.barh(models, values, color=colors, edgecolor=COLORS["dark"], linewidth=0.6)

    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + (0.01 * (xmax if xmax else max(values))), bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=10, color=COLORS["dark"])
        if i == best_idx:
            ax.text(val + (0.075 * (xmax if xmax else max(values))), bar.get_y() + bar.get_height() / 2,
                    "best", va="center", ha="left", fontsize=9, color="white",
                    bbox=dict(boxstyle="round,pad=0.18", fc=COLORS["navy"], ec=COLORS["navy"], lw=0.5))

    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.18)
    if xmax is not None:
        ax.set_xlim(0, xmax)
    ax.tick_params(axis="y", length=0)

    # No title; captions will be in manuscript.
    save_fig(fig, filename)


def make_model_metric_figures(summary: pd.DataFrame):
    make_metric_bar(summary, "roc_auc", "fig3a_model_roc_auc", "ROC-AUC", xmax=1.0)
    make_metric_bar(summary, "pr_auc", "fig3b_model_pr_auc", "PR-AUC", xmax=max(0.04, float(summary["pr_auc"].max()) * 1.6))
    make_metric_bar(summary, "f1", "fig3c_model_f1_score", "F1-score", xmax=max(0.10, float(summary["f1"].max()) * 1.45))


# =============================================================================
# Top-k screening figures
# =============================================================================

def make_topk_recall_figure(topk: pd.DataFrame):
    models = ["Logistic regression", "Gradient boosting", "DeepCut-inspired neural threshold"]
    fig, ax = plt.subplots(figsize=(9.4, 5.8))

    for m in models:
        d = topk[topk["model"] == m].sort_values("top_fraction")
        x = d["top_fraction"].values * 100
        y = d["recall_at_k"].values
        ax.plot(x, y, marker="o", lw=2.3, color=MODEL_COLORS[m], label=m)
        for xi, yi in zip(x, y):
            ax.text(xi, yi + 0.008, f"{yi:.3f}", ha="center", va="bottom", fontsize=8.5, color=MODEL_COLORS[m])

    ax.set_xlabel("Screening budget (% of filings reviewed)")
    ax.set_ylabel("Recall of AAER-positive filings")
    ax.set_xticks([0.5, 1, 2, 5, 10])
    ax.set_xticklabels(["0.5%", "1%", "2%", "5%", "10%"])
    ax.set_ylim(0, max(0.33, topk["recall_at_k"].max() * 1.20))
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="upper left", frameon=False, ncol=1)

    # Callout
    ax.text(0.02, -0.22,
            "DeepCut-inspired neural thresholding captures the most positives at 2%, 5%, and 10% review budgets.",
            transform=ax.transAxes, fontsize=10.5, color=COLORS["dark"],
            bbox=dict(boxstyle="round,pad=0.45", fc=COLORS["orange_light"], ec=COLORS["orange"], lw=1.0))

    save_fig(fig, "fig4_topk_recall_screening_budget")


def make_topk_positives_figure(topk: pd.DataFrame):
    budgets = [0.005, 0.01, 0.02, 0.05, 0.10]
    models = ["Logistic regression", "Gradient boosting", "DeepCut-inspired neural threshold"]
    rows = []
    for m in models:
        d = topk[topk["model"] == m]
        for b in budgets:
            hit = d[np.isclose(d["top_fraction"], b)]
            if not hit.empty:
                rows.append({
                    "model": m,
                    "budget": b * 100,
                    "positives": int(hit["true_positives_captured"].iloc[0]),
                })
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9.6, 5.5))
    x = np.arange(len(budgets))
    width = 0.24

    for i, m in enumerate(models):
        d = df[df["model"] == m].set_index("budget").reindex([b * 100 for b in budgets])
        vals = d["positives"].values
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width=width, color=MODEL_COLORS[m], label=m,
                      edgecolor=COLORS["dark"], linewidth=0.4)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    str(int(val)), ha="center", va="bottom", fontsize=8.5)

    ax.set_xlabel("Screening budget (% of filings reviewed)")
    ax.set_ylabel("AAER-positive filings captured")
    ax.set_xticks(x)
    ax.set_xticklabels(["0.5%", "1%", "2%", "5%", "10%"])
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="upper left", frameon=False)

    save_fig(fig, "fig5_topk_positives_captured")


# =============================================================================
# ROC and PR curves from test predictions
# =============================================================================

def make_roc_curves(pred: pd.DataFrame):
    y = pred["aaer_label"].astype(int).values

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for m in MODEL_ORDER:
        col = PRED_COLS.get(m)
        if col not in pred.columns:
            continue
        p = pred[col].astype(float).values
        fpr, tpr, _ = roc_curve(y, p)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2.0, color=MODEL_COLORS[m], label=f"{m} ({roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], ls="--", lw=1.0, color=COLORS["gray"])
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.20)
    ax.legend(loc="lower right", frameon=False, fontsize=8.2)
    save_fig(fig, "fig6_roc_curves")


def make_precision_recall_curves(pred: pd.DataFrame):
    y = pred["aaer_label"].astype(int).values
    prevalence = y.mean()

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for m in MODEL_ORDER:
        col = PRED_COLS.get(m)
        if col not in pred.columns:
            continue
        p = pred[col].astype(float).values
        precision, recall, _ = precision_recall_curve(y, p)
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, lw=2.0, color=MODEL_COLORS[m], label=f"{m} ({pr_auc:.3f})")

    ax.axhline(prevalence, ls="--", lw=1.0, color=COLORS["gray"], label=f"Prevalence ({prevalence:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.12, np.nanmax([prevalence * 8, 0.08])))
    ax.grid(alpha=0.20)
    ax.legend(loc="upper right", frameon=False, fontsize=8.2)
    save_fig(fig, "fig7_precision_recall_curves")


# =============================================================================
# DeepCut interpretation figures
# =============================================================================

def make_deepcut_method_schematic():
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Panel A
    add_box(ax, (0.03, 0.55), (0.42, 0.38), "", fc="white", ec=COLORS["blue"], lw=1.2)
    ax.text(0.05, 0.89, "A", color="white", weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc=COLORS["navy"], ec=COLORS["navy"]))
    ax.text(0.095, 0.895, "Context-adaptive thresholds", color=COLORS["navy"], weight="bold", fontsize=13)

    add_box(ax, (0.06, 0.64), (0.12, 0.20),
            "Context inputs:\nfirm size\nfiling type\nprofitability\ncash intensity\nindustry",
            fc=COLORS["teal_light"], ec=COLORS["teal"], color=COLORS["dark"], fontsize=9)
    # Neural node diagram
    for x, ys in [(0.25, [0.68, 0.74, 0.80]), (0.31, [0.70, 0.76]), (0.37, [0.71, 0.79])]:
        for yy in ys:
            ax.add_patch(Circle((x, yy), 0.009, fc=COLORS["teal"], ec=COLORS["teal"]))
    for y1 in [0.68, 0.74, 0.80]:
        for y2 in [0.70, 0.76]:
            ax.plot([0.25, 0.31], [y1, y2], color=COLORS["teal"], lw=0.8, alpha=0.6)
    for y1 in [0.70, 0.76]:
        for y2 in [0.71, 0.79]:
            ax.plot([0.31, 0.37], [y1, y2], color=COLORS["teal"], lw=0.8, alpha=0.6)
    arrow(ax, (0.18, 0.74), (0.24, 0.74), color=COLORS["teal"], lw=1.2)
    arrow(ax, (0.38, 0.79), (0.43, 0.82), color=COLORS["teal"], lw=1.2)
    arrow(ax, (0.38, 0.71), (0.43, 0.66), color=COLORS["teal"], lw=1.2)
    add_box(ax, (0.43, 0.78), (0.14, 0.07), r"$c_{high}(z)$", fc=COLORS["blue_light"], ec=COLORS["blue"], fontsize=14)
    add_box(ax, (0.43, 0.62), (0.14, 0.07), r"$c_{low}(z)$", fc=COLORS["teal_light"], ec=COLORS["teal"], fontsize=14)
    ax.text(0.49, 0.735, r"$c_{low}(z) \leq c_{high}(z)$", fontsize=12, color=COLORS["dark"])

    # Panel B
    add_box(ax, (0.52, 0.55), (0.45, 0.38), "", fc="white", ec=COLORS["blue"], lw=1.2)
    ax.text(0.54, 0.89, "B", color="white", weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc=COLORS["navy"], ec=COLORS["navy"]))
    ax.text(0.585, 0.895, "Two-sided exceedances", color=COLORS["navy"], weight="bold", fontsize=13)
    ax.plot([0.58, 0.92], [0.75, 0.75], color=COLORS["dark"], lw=1.3)
    arrow(ax, (0.92, 0.75), (0.95, 0.75), color=COLORS["dark"], lw=1.0)
    ax.plot([0.68, 0.68], [0.65, 0.84], color=COLORS["teal"], ls="--", lw=1)
    ax.plot([0.83, 0.83], [0.65, 0.84], color=COLORS["blue"], ls="--", lw=1)
    ax.add_patch(Circle((0.68, 0.75), 0.010, fc=COLORS["teal"], ec=COLORS["teal"]))
    ax.add_patch(Circle((0.83, 0.75), 0.010, fc=COLORS["blue"], ec=COLORS["blue"]))
    ax.text(0.65, 0.84, r"$c_{low}(z)$", color=COLORS["teal"], fontsize=12)
    ax.text(0.80, 0.84, r"$c_{high}(z)$", color=COLORS["blue"], fontsize=12)
    ax.text(0.60, 0.70, "Low-end\nexceedance", color=COLORS["teal"], ha="center", fontsize=10)
    ax.text(0.755, 0.70, "Low-risk\nmiddle region", color=COLORS["dark"], ha="center", fontsize=10)
    ax.text(0.885, 0.70, "High-end\nexceedance", color=COLORS["blue"], ha="center", fontsize=10)
    add_box(ax, (0.56, 0.58), (0.17, 0.055), r"$low\_exceed=\mathrm{ReLU}(c_{low}-x)$",
            fc="white", ec=COLORS["teal"], fontsize=8.5)
    add_box(ax, (0.77, 0.58), (0.18, 0.055), r"$high\_exceed=\mathrm{ReLU}(x-c_{high})$",
            fc="white", ec=COLORS["blue"], fontsize=8.5)

    # Panel C
    add_box(ax, (0.03, 0.08), (0.42, 0.38), "", fc="white", ec=COLORS["blue"], lw=1.2)
    ax.text(0.05, 0.42, "C", color="white", weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc=COLORS["navy"], ec=COLORS["navy"]))
    ax.text(0.095, 0.425, "Accounting threshold variables", color=COLORS["navy"], weight="bold", fontsize=13)
    variables = [
        "TATA", "accruals/assets", "DSRI", "GMI", "AQI", "SGI",
        "LVGI", "leverage", "ROA", "current ratio", "revenue growth", "receivables/revenue",
        "inventory/revenue"
    ]
    x0, y0 = 0.07, 0.35
    w, h = 0.11, 0.045
    for i, var in enumerate(variables):
        row, col = divmod(i, 3)
        add_box(ax, (x0 + col * 0.12, y0 - row * 0.06), (w, h), var,
                fc=COLORS["teal_light"], ec=COLORS["teal"], fontsize=8.5, color=COLORS["dark"])
    ax.text(0.24, 0.12, "21 threshold features", ha="center", color=COLORS["teal"], fontsize=11, weight="bold")

    # Panel D
    add_box(ax, (0.52, 0.08), (0.45, 0.38), "", fc="white", ec=COLORS["blue"], lw=1.2)
    ax.text(0.54, 0.42, "D", color="white", weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc=COLORS["navy"], ec=COLORS["navy"]))
    ax.text(0.585, 0.425, "Risk score and interpretation", color=COLORS["navy"], weight="bold", fontsize=13)
    add_box(ax, (0.56, 0.31), (0.37, 0.08),
            r"$logit(p)=b(z)+\sum w_{low}(z)\,low\_exceed+\sum w_{high}(z)\,high\_exceed$",
            fc=COLORS["blue_light"], ec=COLORS["blue"], fontsize=12)
    add_box(ax, (0.59, 0.20), (0.32, 0.07),
            "DeepCut identifies filing risk by learning\ncontext-dependent lower and upper accounting thresholds.",
            fc=COLORS["orange_light"], ec=COLORS["orange"], color=COLORS["dark"], fontsize=10)
    add_box(ax, (0.60, 0.12), (0.28, 0.05), "81 context features",
            fc="white", ec=COLORS["blue"], color=COLORS["blue"], fontsize=10, lw=1.0)

    save_fig(fig, "fig8_deepcut_thresholding_method_schematic")


def make_deepcut_weight_figure():
    if not THRESHOLD_FILE.exists():
        print(f"Skipping DeepCut threshold figure; missing {THRESHOLD_FILE}")
        return

    df = pd.read_csv(THRESHOLD_FILE)
    needed = {"feature", "mean_low_exceedance_weight", "mean_high_exceedance_weight"}
    if not needed.issubset(df.columns):
        print(f"Skipping DeepCut threshold figure; expected columns not found in {THRESHOLD_FILE}")
        return

    df["total_weight"] = df["mean_low_exceedance_weight"].abs() + df["mean_high_exceedance_weight"].abs()
    df = df.sort_values("total_weight", ascending=False).head(12).sort_values("total_weight")

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    y = np.arange(len(df))
    ax.barh(y - 0.18, df["mean_low_exceedance_weight"], height=0.35,
            color=COLORS["teal"], label="Low-end exceedance weight", edgecolor=COLORS["dark"], linewidth=0.4)
    ax.barh(y + 0.18, df["mean_high_exceedance_weight"], height=0.35,
            color=COLORS["orange"], label="High-end exceedance weight", edgecolor=COLORS["dark"], linewidth=0.4)

    ax.set_yticks(y)
    ax.set_yticklabels(df["feature"])
    ax.set_xlabel("Mean learned exceedance weight")
    ax.grid(axis="x", alpha=0.22)
    ax.legend(frameon=False, loc="lower right")
    save_fig(fig, "fig9_deepcut_exceedance_weight_summary")


# =============================================================================
# Captions
# =============================================================================

def write_captions():
    captions = {
        "Figure 1": "Study workflow for SEC accounting-enforcement risk screening. SEC 10-K and 10-Q financial-statement data are linked to SEC Accounting and Auditing Enforcement Releases (AAERs), transformed into accounting-risk features, split chronologically, and evaluated using discrimination, calibration, F1-score, and top-k screening metrics.",
        "Figure 2": "Temporal validation design. Models are trained on 2009–2018 filings, tuned on 2019–2020 filings, and evaluated on the held-out 2021–2023 test period. The test period contains 74,362 filing-period observations and 584 AAER-positive observations, corresponding to a prevalence of approximately 0.785%.",
        "Figure 3a": "ROC-AUC comparison across benchmark models and the DeepCut-inspired neural threshold model on the temporal test set.",
        "Figure 3b": "PR-AUC comparison across models. Because AAER-positive observations are rare, PR-AUC is reported alongside ROC-AUC.",
        "Figure 3c": "F1-score comparison across models using thresholds selected on the validation period.",
        "Figure 4": "Top-k screening recall across review budgets. The DeepCut-inspired neural threshold model captures the largest fraction of AAER-positive filings at the 2%, 5%, and 10% review-budget levels.",
        "Figure 5": "Number of AAER-positive filings captured within each screening budget for logistic regression, gradient boosting, and DeepCut-inspired neural thresholding.",
        "Figure 6": "Receiver operating characteristic curves for all evaluated models on the temporal test set.",
        "Figure 7": "Precision-recall curves for all evaluated models on the temporal test set. The dashed horizontal line indicates the test-set AAER prevalence.",
        "Figure 8": "DeepCut-inspired neural thresholding architecture. The model learns context-dependent lower and upper thresholds for accounting variables and transforms deviations beyond those thresholds into two-sided exceedance features.",
        "Figure 9": "DeepCut-inspired threshold interpretation. Mean low-end and high-end exceedance weights summarize which threshold deviations contribute most strongly to predicted AAER risk.",
    }

    out = FIG_DIR / "figure_captions.txt"
    with out.open("w", encoding="utf-8") as f:
        for k, v in captions.items():
            f.write(f"{k}. {v}\n\n")
    print(f"Saved: {out}")


# =============================================================================
# Main
# =============================================================================

def main():
    set_style()
    summary, topk, pred = load_inputs()

    make_workflow_figure()
    make_temporal_split_figure()
    make_model_metric_figures(summary)
    make_topk_recall_figure(topk)
    make_topk_positives_figure(topk)
    make_roc_curves(pred)
    make_precision_recall_curves(pred)
    make_deepcut_method_schematic()
    make_deepcut_weight_figure()
    write_captions()

    print("\nDONE.")
    print(f"All figures saved in: {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()
