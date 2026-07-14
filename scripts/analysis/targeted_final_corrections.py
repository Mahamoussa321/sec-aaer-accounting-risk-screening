#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Final targeted corrections for the SEC-AAER manuscript.

This script does only the two analyses still required before manuscript revision:

1. Re-evaluates the temporal test on the mature FY2021-FY2022 cohort and
   reports FY2023 separately because a three-year forward AAER window is not
   fully observable through the available 2026Q1 enforcement data.
2. Repeats the 0-, 1-, 2-, and 3-year AAER label-window sensitivity with the
   same five DeepCut random seeds for every window, evaluated on the same
   mature FY2021-FY2022 test cohort. Logistic regression is deterministic under
   the specified solver, so one fit per window is reused across the five seed
   rows and explicitly identified as such.

The run is resumable. Existing compatible DeepCut caches from the completed
publication-strengthening analysis are reused when available.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The full strengthening module is shipped beside this script.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import publication_strengthening as ps  # noqa: E402


DEFAULT_SEEDS = (20260531, 20260532, 20260533, 20260534, 20260535)
PRIMARY_DEEPCUT = "DeepCut-inspired neural threshold"
LOGISTIC = "Logistic regression"


@dataclass
class FinalPaths:
    root: Path
    data_file: Path
    raw_predictions: Path
    main_summary: Path
    reviewed_labels: Path
    strengthening_dir: Path
    strengthening_cache: Path
    calibrated_predictions: Path
    calibrated_metrics: Path
    output_dir: Path
    cache_dir: Path
    tables_dir: Path
    figures_dir: Path


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def first_existing(candidates: Iterable[Path], description: str) -> Path:
    for path in candidates:
        if path.exists():
            return path.resolve()
    listing = "\n - ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not locate {description}. Checked:\n - {listing}")


def resolve_paths(root: Path) -> FinalPaths:
    root = root.resolve()
    data_file = first_existing(
        [
            root / "outputs" / "aaer_enhanced_accounting_benchmark" / "enhanced_10k_10q_analysis_dataset.csv",
            root / "aaer_enhanced_accounting_benchmark" / "enhanced_10k_10q_analysis_dataset.csv",
        ],
        "the enhanced SEC-AAER dataset",
    )
    raw_predictions = first_existing(
        [
            root / "outputs" / "accounting_deepcut_ml_benchmark" / "test_predictions.csv",
            root / "accounting_deepcut_ml_benchmark" / "test_predictions.csv",
        ],
        "the original temporal-test predictions",
    )
    main_summary = first_existing(
        [
            root / "outputs" / "accounting_deepcut_ml_benchmark" / "model_comparison_summary.csv",
            root / "accounting_deepcut_ml_benchmark" / "model_comparison_summary.csv",
        ],
        "the main model comparison summary",
    )
    reviewed_labels = first_existing(
        [
            root / "aaer_labels_reviewed.csv",
            root / "manual_review" / "aaer_labels_reviewed.csv",
            root.parent.parent / "manual_review" / "aaer_labels_reviewed.csv",
        ],
        "the reviewed AAER label file",
    )
    strengthening_dir = first_existing(
        [
            root / "outputs" / "publication_strengthening",
            root / "publication_strengthening",
        ],
        "the completed publication-strengthening output folder",
    )
    strengthening_cache = strengthening_dir / "cache"
    calibrated_predictions = first_existing(
        [strengthening_dir / "calibrated_test_predictions.csv"],
        "the calibrated temporal-test predictions",
    )
    calibrated_metrics = first_existing(
        [strengthening_dir / "tables" / "calibrated_model_metrics.csv"],
        "the validation calibration summary",
    )
    output_dir = root / "outputs" / "final_targeted_corrections"
    cache_dir = output_dir / "cache"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    for folder in (output_dir, cache_dir, tables_dir, figures_dir):
        folder.mkdir(parents=True, exist_ok=True)
    return FinalPaths(
        root=root,
        data_file=data_file,
        raw_predictions=raw_predictions,
        main_summary=main_summary,
        reviewed_labels=reviewed_labels,
        strengthening_dir=strengthening_dir,
        strengthening_cache=strengthening_cache,
        calibrated_predictions=calibrated_predictions,
        calibrated_metrics=calibrated_metrics,
        output_dir=output_dir,
        cache_dir=cache_dir,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
    )


def parse_seeds(text: str) -> list[int]:
    seeds = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seed values must be unique.")
    return seeds


def model_from_probability_column(column: str) -> str:
    probability_key = column.split("__", 1)[0]
    if probability_key in ps.MODEL_DISPLAY:
        return ps.MODEL_DISPLAY[probability_key]
    base = probability_key[5:] if probability_key.startswith("prob_") else probability_key
    return base.replace("_", " ")


def raw_probability_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("prob_") and c.endswith("__raw")]


def selected_calibration_map(calibration_metrics: pd.DataFrame) -> dict[str, tuple[str, float]]:
    selected = calibration_metrics[pd.to_numeric(calibration_metrics["selected_by_validation_brier"], errors="coerce").eq(1)].copy()
    result: dict[str, tuple[str, float]] = {}
    for _, row in selected.iterrows():
        result[str(row["model"])] = (str(row["calibration_method"]), float(row["threshold"]))
    return result


def raw_threshold_map(calibration_metrics: pd.DataFrame) -> dict[str, float]:
    raw = calibration_metrics[calibration_metrics["calibration_method"].eq("raw")]
    return {str(r["model"]): float(r["threshold"]) for _, r in raw.iterrows()}


def metric_rows_for_cohort(
    frame: pd.DataFrame,
    cohort_name: str,
    calibration_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = frame["aaer_label"].to_numpy(int)
    raw_thresholds = raw_threshold_map(calibration_metrics)
    selected_map = selected_calibration_map(calibration_metrics)
    raw_rows: list[dict] = []
    selected_rows: list[dict] = []
    for raw_col in raw_probability_columns(frame):
        model = model_from_probability_column(raw_col)
        p = frame[raw_col].to_numpy(float)
        row = ps.metric_row(model, y, p, raw_thresholds.get(model, 0.5), subgroup=cohort_name)
        row["calibration_method"] = "raw"
        raw_rows.append(row)
        method, threshold = selected_map.get(model, ("raw", raw_thresholds.get(model, 0.5)))
        selected_col = raw_col.replace("__raw", f"__{method}")
        if selected_col not in frame.columns:
            selected_col = raw_col
            method = "raw"
        selected_row = ps.metric_row(
            model,
            y,
            frame[selected_col].to_numpy(float),
            threshold,
            subgroup=cohort_name,
        )
        selected_row["calibration_method"] = method
        selected_rows.append(selected_row)
    return pd.DataFrame(raw_rows), pd.DataFrame(selected_rows)


def clustered_bootstrap(
    frame: pd.DataFrame,
    probability_columns: Sequence[str],
    model_names: dict[str, str],
    reps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = frame["aaer_label"].to_numpy(int)
    firm_codes, firms = pd.factorize(frame["cik"].astype(str), sort=True)
    n_firms = len(firms)
    rng = np.random.default_rng(seed)
    arrays = {col: frame[col].to_numpy(float) for col in probability_columns}
    metrics = ["roc_auc", "pr_auc", "brier", "recall_top_0.005", "recall_top_0.01", "recall_top_0.02", "recall_top_0.05", "recall_top_0.1"]

    def calculate(prob: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
        values = {
            "roc_auc": ps.safe_roc_auc(y, prob, weights),
            "pr_auc": ps.safe_pr_auc(y, prob, weights),
            "brier": float(np.average((y - prob) ** 2, weights=weights)) if weights is not None else float(np.mean((y - prob) ** 2)),
        }
        for fraction in ps.TOP_FRACTIONS:
            values[f"recall_top_{fraction}"] = ps.weighted_topk(y, prob, fraction, weights)["recall"]
        return values

    point = {col: calculate(prob) for col, prob in arrays.items()}
    boot = {col: {metric: [] for metric in metrics} for col in probability_columns}
    deep_col = next((c for c in probability_columns if "DeepCut" in c), None)
    paired: dict[tuple[str, str], list[float]] = {}

    for rep in range(reps):
        counts = rng.multinomial(n_firms, np.full(n_firms, 1.0 / n_firms))
        weights = counts[firm_codes].astype(float)
        if np.sum(weights * y) <= 0 or np.sum(weights * (1 - y)) <= 0:
            continue
        current = {}
        for col, prob in arrays.items():
            values = calculate(prob, weights)
            current[col] = values
            for metric, value in values.items():
                boot[col][metric].append(value)
        if deep_col is not None:
            for other in probability_columns:
                if other == deep_col:
                    continue
                for metric in ("roc_auc", "pr_auc", "recall_top_0.02", "recall_top_0.05", "recall_top_0.1"):
                    paired.setdefault((other, metric), []).append(current[deep_col][metric] - current[other][metric])
        if (rep + 1) % max(1, min(100, reps)) == 0:
            log(f"  mature-test cluster bootstrap: {rep + 1}/{reps}")

    ci_rows = []
    for col in probability_columns:
        for metric in metrics:
            values = np.asarray(boot[col][metric], float)
            values = values[np.isfinite(values)]
            ci_rows.append({
                "model": model_names[col],
                "metric": metric,
                "estimate": point[col][metric],
                "ci_lower": float(np.quantile(values, 0.025)) if len(values) else np.nan,
                "ci_upper": float(np.quantile(values, 0.975)) if len(values) else np.nan,
                "bootstrap_reps": int(len(values)),
                "cluster_unit": "CIK firm",
            })
    paired_rows = []
    if deep_col is not None:
        for (other, metric), values in paired.items():
            arr = np.asarray(values, float)
            arr = arr[np.isfinite(arr)]
            estimate = point[deep_col][metric] - point[other][metric]
            p_two = min(1.0, 2.0 * min(float(np.mean(arr <= 0)), float(np.mean(arr >= 0)))) if len(arr) else np.nan
            paired_rows.append({
                "comparison": f"{model_names[deep_col]} minus {model_names[other]}",
                "metric": metric,
                "estimate_difference": estimate,
                "ci_lower": float(np.quantile(arr, 0.025)) if len(arr) else np.nan,
                "ci_upper": float(np.quantile(arr, 0.975)) if len(arr) else np.nan,
                "bootstrap_p_two_sided": p_two,
                "bootstrap_reps": int(len(arr)),
            })
    return pd.DataFrame(ci_rows), pd.DataFrame(paired_rows)


def find_full_deepcut_cache(cache_dir: Path, seed: int) -> Path | None:
    candidates = [
        cache_dir / f"deepcut_full_seed_{seed}.npz",
        cache_dir / f"deepcut_full_adaptive_two_sided_seed_{seed}.npz",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def mature_deepcut_seed_metrics(paths: FinalPaths, mature_mask: np.ndarray, y_mature: np.ndarray, seeds: Sequence[int]) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        cache = find_full_deepcut_cache(paths.strengthening_cache, seed)
        if cache is None:
            log(f"  no completed full DeepCut cache found for seed {seed}; skipping mature seed summary")
            continue
        saved = np.load(cache)
        p = np.asarray(saved["p_test"], float)[mature_mask]
        row = ps.metric_row(PRIMARY_DEEPCUT, y_mature, p, subgroup="Mature FY2021-FY2022 temporal test")
        row["seed"] = int(seed)
        row["cache_file"] = cache.name
        metadata = cache.with_suffix(".json")
        if metadata.exists():
            info = json.loads(metadata.read_text(encoding="utf-8"))
            row["best_epoch"] = info.get("best_epoch")
            row["best_validation_pr_auc"] = info.get("best_val_pr_auc")
        rows.append(row)
    return pd.DataFrame(rows)


def save_ci_figure(ci: pd.DataFrame, metric: str, title: str, base: Path) -> None:
    plot = ci[ci["metric"].eq(metric)].sort_values("estimate")
    if plot.empty:
        return
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
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
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_topk_figure(raw_metrics: pd.DataFrame, base: Path) -> None:
    chosen = raw_metrics[raw_metrics["model"].isin([PRIMARY_DEEPCUT, LOGISTIC])].copy()
    if chosen.empty:
        return
    fractions = list(ps.TOP_FRACTIONS)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for _, row in chosen.iterrows():
        recalls = [row[f"recall_top_{str(f).replace('.', 'p')}"] for f in fractions]
        ax.plot(np.asarray(fractions) * 100, recalls, marker="o", label=row["model"])
    ax.set_xlabel("Percentage of mature test filings reviewed")
    ax.set_ylabel("Recall of AAER-positive filings")
    ax.set_title("Mature FY2021-FY2022 top-k screening recall")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_calibration_figure(frame: pd.DataFrame, selected_map: dict[str, tuple[str, float]], model: str, base: Path) -> None:
    method, _ = selected_map.get(model, ("raw", 0.5))
    prefix = ps.sanitize_probability_column(model)
    column = f"{prefix}__{method}"
    if column not in frame.columns:
        return
    bins = ps.calibration_bins(frame["aaer_label"].to_numpy(int), frame[column].to_numpy(float), n_bins=10)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(bins["mean_probability"], bins["observed_rate"], marker="o", label=f"{model}: {method}")
    maximum = max(float(bins["mean_probability"].max()), float(bins["observed_rate"].max()), 0.001)
    ax.plot([0, maximum], [0, maximum], linestyle="--", label="Ideal calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed AAER rate")
    ax.set_title(f"Mature-test calibration: {model}")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def mature_test_stage(paths: FinalPaths, mature_max_fy: int, bootstrap_reps: int, seeds: Sequence[int]) -> None:
    marker = paths.cache_dir / "mature_test.done"
    if marker.exists():
        log("Skipping mature temporal-test analysis (already completed).")
        return
    log("Stage 1/3: mature FY2021-FY2022 test analysis and separate FY2023 reporting.")
    calibrated = pd.read_csv(paths.calibrated_predictions, low_memory=False)
    calibrated["cik"] = calibrated["cik"].astype(str)
    calibrated["fy_num"] = pd.to_numeric(calibrated["fy_num"], errors="coerce")
    calibrated["aaer_label"] = pd.to_numeric(calibrated["aaer_label"], errors="raise").astype(int)
    calibration_metrics = pd.read_csv(paths.calibrated_metrics, low_memory=False)

    mature = calibrated[calibrated["fy_num"].le(mature_max_fy)].reset_index(drop=True)
    incomplete = calibrated[calibrated["fy_num"].gt(mature_max_fy)].reset_index(drop=True)
    if mature.empty:
        raise RuntimeError("The mature test cohort is empty.")
    cohort_summary = pd.DataFrame([
        {
            "cohort": f"Mature temporal test through FY{mature_max_fy}",
            "rows": len(mature),
            "firms": mature["cik"].nunique(),
            "positives": int(mature["aaer_label"].sum()),
            "prevalence": float(mature["aaer_label"].mean()),
            "outcome_status": "Primary; complete three-year outcome maturation under available AAER data",
        },
        {
            "cohort": f"FY{mature_max_fy + 1} incompletely matured cohort",
            "rows": len(incomplete),
            "firms": incomplete["cik"].nunique(),
            "positives": int(incomplete["aaer_label"].sum()),
            "prevalence": float(incomplete["aaer_label"].mean()) if len(incomplete) else np.nan,
            "outcome_status": "Descriptive only; three-year forward AAER window is administratively incomplete",
        },
    ])
    cohort_summary.to_csv(paths.tables_dir / "outcome_maturity_cohort_summary.csv", index=False)

    mature_raw, mature_selected = metric_rows_for_cohort(
        mature, f"Mature temporal test through FY{mature_max_fy}", calibration_metrics
    )
    incomplete_raw, incomplete_selected = metric_rows_for_cohort(
        incomplete, f"FY{mature_max_fy + 1} incomplete follow-up", calibration_metrics
    ) if not incomplete.empty else (pd.DataFrame(), pd.DataFrame())
    mature_raw.to_csv(paths.tables_dir / "mature_temporal_test_raw_metrics.csv", index=False)
    mature_selected.to_csv(paths.tables_dir / "mature_temporal_test_selected_calibration_metrics.csv", index=False)
    incomplete_raw.to_csv(paths.tables_dir / "fy2023_incomplete_followup_raw_metrics.csv", index=False)
    incomplete_selected.to_csv(paths.tables_dir / "fy2023_incomplete_followup_selected_calibration_metrics.csv", index=False)

    probability_columns = raw_probability_columns(mature)
    names = {column: model_from_probability_column(column) for column in probability_columns}
    ci, paired = clustered_bootstrap(mature, probability_columns, names, bootstrap_reps, ps.RANDOM_STATE)
    ci.to_csv(paths.tables_dir / "mature_temporal_test_firm_clustered_confidence_intervals.csv", index=False)
    paired.to_csv(paths.tables_dir / "mature_temporal_test_paired_model_differences.csv", index=False)

    full_cal = pd.read_csv(paths.calibrated_predictions, usecols=["fy_num", "aaer_label"], low_memory=False)
    mature_mask = pd.to_numeric(full_cal["fy_num"], errors="coerce").le(mature_max_fy).to_numpy()
    seed_metrics = mature_deepcut_seed_metrics(
        paths,
        mature_mask,
        full_cal.loc[mature_mask, "aaer_label"].to_numpy(int),
        seeds,
    )
    seed_metrics.to_csv(paths.tables_dir / "mature_deepcut_five_seed_metrics.csv", index=False)
    if not seed_metrics.empty:
        summary_rows = []
        for metric in ("roc_auc", "pr_auc", "brier", "recall_top_0p02", "recall_top_0p05", "recall_top_0p1"):
            values = pd.to_numeric(seed_metrics[metric], errors="coerce")
            summary_rows.append({
                "metric": metric,
                "seeds": int(values.notna().sum()),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0,
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            })
        pd.DataFrame(summary_rows).to_csv(paths.tables_dir / "mature_deepcut_five_seed_summary.csv", index=False)

    # Selected calibration bins for manuscript/supplement.
    selected = selected_calibration_map(calibration_metrics)
    bin_tables = []
    for model, (method, _) in selected.items():
        column = ps.sanitize_probability_column(model) + f"__{method}"
        if column not in mature.columns:
            continue
        bins = ps.calibration_bins(mature["aaer_label"].to_numpy(int), mature[column].to_numpy(float), n_bins=10)
        bins.insert(0, "calibration_method", method)
        bins.insert(0, "model", model)
        bin_tables.append(bins)
    if bin_tables:
        pd.concat(bin_tables, ignore_index=True).to_csv(paths.tables_dir / "mature_selected_calibration_bins.csv", index=False)

    save_ci_figure(ci, "roc_auc", "Mature FY2021-FY2022 firm-clustered ROC-AUC intervals", paths.figures_dir / "figF1_mature_test_roc_auc")
    save_ci_figure(ci, "pr_auc", "Mature FY2021-FY2022 firm-clustered PR-AUC intervals", paths.figures_dir / "figF2_mature_test_pr_auc")
    save_topk_figure(mature_raw, paths.figures_dir / "figF3_mature_test_topk_recall")
    save_calibration_figure(mature, selected, LOGISTIC, paths.figures_dir / "figF4_mature_calibration_logistic")
    save_calibration_figure(mature, selected, PRIMARY_DEEPCUT, paths.figures_dir / "figF5_mature_calibration_deepcut")

    marker.write_text(now(), encoding="utf-8")
    log(f"Mature cohort: {len(mature):,} rows, {int(mature['aaer_label'].sum()):,} positives.")
    log("Mature temporal-test analysis completed.")


def compatible_old_cache(paths: FinalPaths, window: int, seed: int) -> Path | None:
    candidates: list[Path] = []
    if window == 0 and seed == 20260531:
        candidates.append(paths.strengthening_cache / "label_window_0_deepcut.npz")
    if window == 1 and seed == 20260532:
        candidates.append(paths.strengthening_cache / "label_window_1_deepcut.npz")
    if window == 2 and seed == 20260533:
        candidates.append(paths.strengthening_cache / "label_window_2_deepcut.npz")
    if window == 3:
        full = find_full_deepcut_cache(paths.strengthening_cache, seed)
        if full is not None:
            candidates.append(full)
    for candidate in candidates:
        if candidate.exists() and candidate.with_suffix(".json").exists():
            return candidate
    return None


def copy_cache_triplet(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    for suffix in (".json", ".thresholds.csv"):
        source_side = source.with_suffix(suffix)
        destination_side = destination.with_suffix(suffix)
        if source_side.exists():
            shutil.copy2(source_side, destination_side)


def save_window_figure(summary: pd.DataFrame, metric: str, ylabel: str, title: str, base: Path) -> None:
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for model, group in summary.groupby("model", sort=False):
        group = group.sort_values("label_window_years")
        ax.errorbar(
            group["label_window_years"],
            group[f"{metric}_mean"],
            yerr=group[f"{metric}_sd"].fillna(0.0),
            marker="o",
            capsize=4,
            label=model,
        )
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("Forward AAER label window (years)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def consistent_label_window_stage(
    paths: FinalPaths,
    mature_max_fy: int,
    seeds: Sequence[int],
    max_epochs: int,
    patience: int,
    device: str,
) -> None:
    marker = paths.cache_dir / "consistent_label_windows.done"
    if marker.exists():
        log("Skipping consistent-seed label-window sensitivity (already completed).")
        return
    log("Stage 2/3: consistent five-seed AAER label-window sensitivity.")
    df, all_features, threshold_features, context_features = ps.load_data(
        ps.Paths(
            root=paths.root,
            data_file=paths.data_file,
            predictions_file=paths.raw_predictions,
            reviewed_labels=paths.reviewed_labels,
            output_dir=paths.output_dir,
            cache_dir=paths.cache_dir,
            figures_dir=paths.figures_dir,
            tables_dir=paths.tables_dir,
        )
    )
    reviewed = pd.read_csv(paths.reviewed_labels, low_memory=False)
    train_mask = df["split"].eq("train").to_numpy()
    val_mask = df["split"].eq("val").to_numpy()
    test_mask = df["split"].eq("test").to_numpy()
    train = df.loc[train_mask].copy()
    val = df.loc[val_mask].copy()
    test = df.loc[test_mask].copy()
    mature_test_mask = pd.to_numeric(test["fy_num"], errors="coerce").le(mature_max_fy).to_numpy()
    incomplete_test_mask = ~mature_test_mask

    trainer = ps.DeepCutTrainer(
        threshold_features,
        context_features,
        max_epochs=max_epochs,
        patience=patience,
        device=device,
    )
    arrays = trainer.prepare(train, val, test)
    metric_rows: list[dict] = []
    prevalence_rows: list[dict] = []

    for window in (0, 1, 2, 3):
        labels = ps.labels_for_window(df, reviewed, window)
        y_train = labels[train_mask]
        y_val = labels[val_mask]
        y_test = labels[test_mask]
        y_mature = y_test[mature_test_mask]
        y_incomplete = y_test[incomplete_test_mask]
        prevalence_rows.extend([
            {"label_window_years": window, "split_or_cohort": "train", "rows": len(y_train), "positives": int(y_train.sum()), "prevalence": float(y_train.mean())},
            {"label_window_years": window, "split_or_cohort": "validation", "rows": len(y_val), "positives": int(y_val.sum()), "prevalence": float(y_val.mean())},
            {"label_window_years": window, "split_or_cohort": f"mature test through FY{mature_max_fy}", "rows": len(y_mature), "positives": int(y_mature.sum()), "prevalence": float(y_mature.mean())},
            {"label_window_years": window, "split_or_cohort": f"FY{mature_max_fy + 1} incomplete follow-up", "rows": len(y_incomplete), "positives": int(y_incomplete.sum()), "prevalence": float(y_incomplete.mean()) if len(y_incomplete) else np.nan},
        ])
        log(f"  window={window}: train/val/mature positives={int(y_train.sum())}/{int(y_val.sum())}/{int(y_mature.sum())}")
        if any(len(np.unique(y)) < 2 for y in (y_train, y_val, y_mature)):
            log(f"  skipping window={window}; a required split/cohort lacks both outcome classes")
            continue

        # Logistic regression is deterministic with the specified lbfgs formulation.
        logistic_cache = paths.cache_dir / f"label_window_{window}_logistic_predictions.npz"
        if logistic_cache.exists():
            saved = np.load(logistic_cache)
            p_val_log = saved["p_val"]
            p_test_log = saved["p_test"]
            log(f"  loaded cached logistic predictions for window={window}")
        else:
            p_val_log, p_test_log = ps.fit_popular_model(
                LOGISTIC,
                ps.make_popular_models(ps.RANDOM_STATE)[LOGISTIC],
                train,
                val,
                test,
                all_features,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
            )
            np.savez_compressed(logistic_cache, p_val=p_val_log, p_test=p_test_log)
        logistic_threshold = ps.pick_threshold(y_val, p_val_log)
        logistic_primary = ps.metric_row(
            LOGISTIC,
            y_mature,
            p_test_log[mature_test_mask],
            logistic_threshold,
            subgroup=f"Mature test, AAER window={window}",
        )
        logistic_primary.update({
            "label_window_years": window,
            "seed": "deterministic_reuse",
            "deterministic_comparator_reused": 1,
            "mature_max_fy": mature_max_fy,
        })
        metric_rows.append(logistic_primary)

        for seed in seeds:
            destination = paths.cache_dir / f"label_window_{window}_deepcut_seed_{seed}.npz"
            reused = 0
            if not destination.exists():
                source = compatible_old_cache(paths, window, seed)
                if source is not None:
                    copy_cache_triplet(source, destination)
                    reused = 1
                    log(f"  reused compatible cache for window={window}, seed={seed}: {source.name}")
            result = ps.fit_or_load_deepcut(
                trainer,
                arrays,
                y_train,
                y_val,
                y_test,
                seed,
                "full_adaptive_two_sided",
                destination,
            )
            deep_threshold = ps.pick_threshold(y_val, result["p_val"])
            row = ps.metric_row(
                PRIMARY_DEEPCUT,
                y_mature,
                np.asarray(result["p_test"])[mature_test_mask],
                deep_threshold,
                subgroup=f"Mature test, AAER window={window}",
            )
            row.update({
                "label_window_years": window,
                "seed": int(seed),
                "deterministic_comparator_reused": 0,
                "mature_max_fy": mature_max_fy,
                "best_epoch": result.get("best_epoch"),
                "best_validation_pr_auc": result.get("best_val_pr_auc"),
                "compatible_prior_cache_reused": reused,
            })
            metric_rows.append(row)

    prevalence = pd.DataFrame(prevalence_rows)
    prevalence.to_csv(paths.tables_dir / "consistent_seed_label_window_prevalence.csv", index=False)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(paths.tables_dir / "consistent_seed_label_window_metrics.csv", index=False)

    summary_rows = []
    metric_names = ["roc_auc", "pr_auc", "brier", "recall_top_0p02", "recall_top_0p05", "recall_top_0p1"]
    for (model, window), group in metrics.groupby(["model", "label_window_years"], sort=True):
        for metric in metric_names:
            values = pd.to_numeric(group[metric], errors="coerce")
            summary_rows.append({
                "model": model,
                "label_window_years": int(window),
                "metric": metric,
                "runs": int(values.notna().sum()),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0,
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "mature_test_max_fy": mature_max_fy,
            })
    long_summary = pd.DataFrame(summary_rows)
    long_summary.to_csv(paths.tables_dir / "consistent_seed_label_window_summary_long.csv", index=False)
    wide_parts = []
    for metric in metric_names:
        part = long_summary[long_summary["metric"].eq(metric)][["model", "label_window_years", "mean", "sd", "minimum", "maximum", "runs"]].copy()
        part = part.rename(columns={
            "mean": f"{metric}_mean",
            "sd": f"{metric}_sd",
            "minimum": f"{metric}_minimum",
            "maximum": f"{metric}_maximum",
            "runs": f"{metric}_runs",
        })
        wide_parts.append(part)
    summary = wide_parts[0]
    for part in wide_parts[1:]:
        summary = summary.merge(part, on=["model", "label_window_years"], how="outer")
    summary.to_csv(paths.tables_dir / "consistent_seed_label_window_summary.csv", index=False)

    # Seed-matched DeepCut minus deterministic logistic differences.
    paired_rows = []
    for window in (0, 1, 2, 3):
        window_rows = metrics[metrics["label_window_years"].eq(window)]
        log_rows = window_rows[window_rows["model"].eq(LOGISTIC)]
        deep_rows = window_rows[window_rows["model"].eq(PRIMARY_DEEPCUT)]
        if log_rows.empty:
            continue
        comparator = log_rows.iloc[0]
        for _, deep in deep_rows.iterrows():
            for metric in ("roc_auc", "pr_auc", "recall_top_0p02", "recall_top_0p05", "recall_top_0p1"):
                paired_rows.append({
                    "label_window_years": window,
                    "seed": deep["seed"],
                    "metric": metric,
                    "deepcut_minus_logistic": float(deep[metric] - comparator[metric]),
                })
    paired_seed = pd.DataFrame(paired_rows)
    paired_seed.to_csv(paths.tables_dir / "consistent_seed_label_window_deepcut_minus_logistic.csv", index=False)

    save_window_figure(summary, "roc_auc", "ROC-AUC", "Mature-test ROC-AUC by AAER label window", paths.figures_dir / "figF6_label_window_roc_auc_consistent_seeds")
    save_window_figure(summary, "pr_auc", "PR-AUC", "Mature-test PR-AUC by AAER label window", paths.figures_dir / "figF7_label_window_pr_auc_consistent_seeds")
    save_window_figure(summary, "recall_top_0p05", "Top-5% recall", "Mature-test top-5% recall by AAER label window", paths.figures_dir / "figF8_label_window_top5_recall_consistent_seeds")

    marker.write_text(now(), encoding="utf-8")
    del df, train, val, test, arrays
    gc.collect()
    log("Consistent-seed label-window sensitivity completed.")


def safe_value(table: pd.DataFrame, model: str, column: str) -> float:
    row = table[table["model"].eq(model)]
    return float(row.iloc[0][column]) if not row.empty else float("nan")


def generate_report(paths: FinalPaths, mature_max_fy: int, seeds: Sequence[int]) -> None:
    log("Stage 3/3: generating final targeted-corrections report.")
    cohort = pd.read_csv(paths.tables_dir / "outcome_maturity_cohort_summary.csv")
    mature_raw = pd.read_csv(paths.tables_dir / "mature_temporal_test_raw_metrics.csv")
    mature_selected = pd.read_csv(paths.tables_dir / "mature_temporal_test_selected_calibration_metrics.csv")
    paired = pd.read_csv(paths.tables_dir / "mature_temporal_test_paired_model_differences.csv")
    window_summary = pd.read_csv(paths.tables_dir / "consistent_seed_label_window_summary.csv")
    seed_summary_path = paths.tables_dir / "mature_deepcut_five_seed_summary.csv"
    seed_summary = pd.read_csv(seed_summary_path) if seed_summary_path.exists() else pd.DataFrame()

    mature_cohort = cohort.iloc[0]
    incomplete_cohort = cohort.iloc[1]
    deep_roc = safe_value(mature_raw, PRIMARY_DEEPCUT, "roc_auc")
    deep_pr = safe_value(mature_raw, PRIMARY_DEEPCUT, "pr_auc")
    log_roc = safe_value(mature_raw, LOGISTIC, "roc_auc")
    log_pr = safe_value(mature_raw, LOGISTIC, "pr_auc")
    deep_top5 = safe_value(mature_raw, PRIMARY_DEEPCUT, "recall_top_0p05")
    log_top5 = safe_value(mature_raw, LOGISTIC, "recall_top_0p05")
    deep_cal = mature_selected[mature_selected["model"].eq(PRIMARY_DEEPCUT)]
    log_cal = mature_selected[mature_selected["model"].eq(LOGISTIC)]

    comparison = paired[(paired["comparison"].str.contains("Logistic", na=False)) & paired["metric"].eq("roc_auc")]
    if not comparison.empty:
        comp = comparison.iloc[0]
        comparison_sentence = (
            f"The mature-cohort DeepCut-minus-logistic ROC-AUC difference was {comp['estimate_difference']:.4f} "
            f"(firm-clustered 95% CI {comp['ci_lower']:.4f} to {comp['ci_upper']:.4f}; "
            f"bootstrap p={comp['bootstrap_p_two_sided']:.3f})."
        )
    else:
        comparison_sentence = "The paired DeepCut-versus-logistic mature-cohort comparison was unavailable."

    lines = [
        "# Final Targeted Corrections for the SEC-AAER Manuscript",
        "",
        f"Generated: {now()}",
        "",
        "## What this run corrected",
        "",
        f"1. The primary temporal test is now limited to FY2021-FY{mature_max_fy}, for which the three-year forward AAER outcome window is mature under the available enforcement data.",
        f"2. FY{mature_max_fy + 1} is reported separately as an administratively incomplete follow-up cohort and is not used for the primary three-year inference.",
        f"3. The 0-, 1-, 2-, and 3-year label-window sensitivity uses the identical DeepCut seeds {', '.join(str(s) for s in seeds)} for every horizon and the identical mature test cohort.",
        "4. Logistic regression is deterministic under the specified solver; one fit per label horizon is reused across seed rows and marked in the output table.",
        "",
        "## Outcome-mature temporal test",
        "",
        f"The primary mature test contains {int(mature_cohort['rows']):,} filing-period observations from {int(mature_cohort['firms']):,} firms, including {int(mature_cohort['positives']):,} AAER-positive observations (prevalence {float(mature_cohort['prevalence']):.4%}).",
        f"The separately reported FY{mature_max_fy + 1} cohort contains {int(incomplete_cohort['rows']):,} observations and {int(incomplete_cohort['positives']):,} observed positives, but later AAERs can still change its three-year labels.",
        "",
        f"- DeepCut raw ranking: ROC-AUC {deep_roc:.4f}, PR-AUC {deep_pr:.5f}, top-5% recall {deep_top5:.4f}.",
        f"- Logistic raw ranking: ROC-AUC {log_roc:.4f}, PR-AUC {log_pr:.5f}, top-5% recall {log_top5:.4f}.",
        f"- {comparison_sentence}",
        "",
    ]
    if not deep_cal.empty:
        row = deep_cal.iloc[0]
        lines.append(f"Validation-selected calibration for DeepCut was {row['calibration_method']}; mature-cohort Brier score {row['brier']:.6f} and ECE {row['ece_10']:.6f}.")
    if not log_cal.empty:
        row = log_cal.iloc[0]
        lines.append(f"Validation-selected calibration for logistic regression was {row['calibration_method']}; mature-cohort Brier score {row['brier']:.6f} and ECE {row['ece_10']:.6f}.")
    lines.extend(["", "## Five-seed mature-cohort stability", ""])
    if not seed_summary.empty:
        for _, row in seed_summary.iterrows():
            lines.append(f"- {row['metric']}: mean {row['mean']:.6f}, SD {row['sd']:.6f}, range {row['minimum']:.6f}-{row['maximum']:.6f} across {int(row['seeds'])} seeds.")
    lines.extend(["", "## Consistent-seed label-window sensitivity", ""])
    for window in (0, 1, 2, 3):
        deep = window_summary[(window_summary["model"].eq(PRIMARY_DEEPCUT)) & window_summary["label_window_years"].eq(window)]
        logistic = window_summary[(window_summary["model"].eq(LOGISTIC)) & window_summary["label_window_years"].eq(window)]
        if not deep.empty:
            d = deep.iloc[0]
            l = logistic.iloc[0] if not logistic.empty else None
            sentence = f"- {window}-year window: DeepCut ROC-AUC {d['roc_auc_mean']:.4f} ± {d['roc_auc_sd']:.4f}, PR-AUC {d['pr_auc_mean']:.5f} ± {d['pr_auc_sd']:.5f}"
            if l is not None:
                sentence += f"; logistic ROC-AUC {l['roc_auc_mean']:.4f}, PR-AUC {l['pr_auc_mean']:.5f}."
            else:
                sentence += "."
            lines.append(sentence)
    lines.extend([
        "",
        "## Manuscript decision",
        "",
        "After these outputs are reviewed for numerical consistency, no additional large computational analysis is required for the current journal submission plan. The remaining work is manuscript revision: use the mature-cohort counts, describe FY2023 as incomplete follow-up, and state statistical uncertainty exactly as reported by the firm-clustered intervals.",
        "",
        "## Principal output tables",
        "",
        "- `tables/outcome_maturity_cohort_summary.csv`",
        "- `tables/mature_temporal_test_raw_metrics.csv`",
        "- `tables/mature_temporal_test_selected_calibration_metrics.csv`",
        "- `tables/mature_temporal_test_firm_clustered_confidence_intervals.csv`",
        "- `tables/mature_temporal_test_paired_model_differences.csv`",
        "- `tables/mature_deepcut_five_seed_summary.csv`",
        "- `tables/consistent_seed_label_window_metrics.csv`",
        "- `tables/consistent_seed_label_window_summary.csv`",
        "",
    ])
    (paths.output_dir / "FINAL_TARGETED_CORRECTIONS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    # A compact manuscript-ready value table.
    manuscript_rows = []
    for _, row in mature_raw.iterrows():
        manuscript_rows.append({
            "analysis": "Primary mature temporal test",
            "model": row["model"],
            "cohort": f"FY2021-FY{mature_max_fy}",
            "rows": row["n"],
            "positives": row["positives"],
            "prevalence": row["prevalence"],
            "roc_auc": row["roc_auc"],
            "pr_auc": row["pr_auc"],
            "brier": row["brier"],
            "top_2pct_recall": row["recall_top_0p02"],
            "top_5pct_recall": row["recall_top_0p05"],
            "top_10pct_recall": row["recall_top_0p1"],
        })
    pd.DataFrame(manuscript_rows).to_csv(paths.tables_dir / "manuscript_ready_primary_values.csv", index=False)

    manifest = []
    for path in sorted(paths.output_dir.rglob("*")):
        if path.is_file():
            manifest.append({"relative_path": str(path.relative_to(paths.output_dir)), "bytes": path.stat().st_size})
    pd.DataFrame(manifest).to_csv(paths.output_dir / "output_manifest.csv", index=False)
    log("Final targeted-corrections report generated.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--mature-max-fy", type=int, default=2022)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--mature-only", action="store_true", help="Run only the non-training mature-cohort diagnostics.")
    parser.add_argument("--fast", action="store_true", help="Small code test; not for manuscript reporting.")
    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)
    if args.fast:
        args.bootstrap_reps = min(args.bootstrap_reps, 20)
        seeds = seeds[:1]
        args.max_epochs = min(args.max_epochs, 2)
        args.patience = min(args.patience, 1)
    paths = resolve_paths(args.project_root)
    config = {
        "project_root": str(args.project_root),
        "mature_max_fy": args.mature_max_fy,
        "bootstrap_reps": args.bootstrap_reps,
        "seeds": seeds,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "device": args.device,
        "mature_only": args.mature_only,
        "fast": args.fast,
    }
    (paths.output_dir / "run_configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    log(f"Project root: {paths.root}")
    log(f"Output directory: {paths.output_dir}")
    mature_test_stage(paths, args.mature_max_fy, args.bootstrap_reps, seeds)
    if not args.mature_only:
        consistent_label_window_stage(
            paths,
            args.mature_max_fy,
            seeds,
            args.max_epochs,
            args.patience,
            args.device,
        )
        generate_report(paths, args.mature_max_fy, seeds)
    else:
        log("Mature-only mode completed; label-window retraining was not run.")
    log("Final targeted corrections completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("Interrupted by user. Individual model caches are retained; rerun to resume.")
        raise
    except Exception:
        traceback.print_exc()
        raise
