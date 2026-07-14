#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_accounting_deepcut_ml_benchmark.py

Accounting-journal version.

This script runs popular machine-learning models plus ONE DeepCut-inspired
neural thresholding model. It does NOT include SIVC and does NOT include
teacher-student distillation.

Input expected in the same folder where you run the script:
  outputs/aaer_enhanced_accounting_benchmark/enhanced_10k_10q_analysis_dataset.csv

This is the enhanced SEC 10-K + 10-Q dataset you already created.

Run:
  cd C:\\Users\\maham
  pip install pandas numpy scikit-learn torch
  python .\\run_accounting_deepcut_ml_benchmark.py

Outputs:
  outputs/accounting_deepcut_ml_benchmark/model_comparison_summary.csv
  outputs/accounting_deepcut_ml_benchmark/topk_screening_metrics.csv
  outputs/accounting_deepcut_ml_benchmark/test_predictions.csv
  outputs/accounting_deepcut_ml_benchmark/deepcut_threshold_summary.csv
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)


# =============================================================================
# Paths and settings
# =============================================================================

ROOT = Path(".")
DATA_FILE = ROOT / "outputs" / "aaer_enhanced_accounting_benchmark" / "enhanced_10k_10q_analysis_dataset.csv"
OUT_DIR = ROOT / "outputs" / "accounting_deepcut_ml_benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 20260531

# Numeric variables from the enhanced accounting benchmark.
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
    "gross_margin", "gross_margin_lag", "gross_margin_change",
    "sga_to_sales", "sga_to_sales_lag", "sga_to_sales_change",
    "revenue_to_assets", "ar_to_assets", "inventory_to_revenue",
]

# DeepCut-inspired threshold variables. Keep them focused and interpretable.
DEEPCUT_THRESHOLD_FEATURES = [
    "tata", "accruals_to_assets", "wc_accruals_to_assets", "dsri", "gmi",
    "aqi", "sgi", "sgai", "lvgi", "leverage", "roa", "current_ratio",
    "gross_margin_change", "rev_growth_yoy", "assets_growth_yoy",
    "ar_growth_yoy", "inventory_growth_yoy", "cash_to_assets",
    "receivables_to_revenue", "inventory_to_revenue", "loss_indicator",
]

# Context variables used to adapt the cutpoints.
DEEPCUT_CONTEXT_BASE = [
    "log_assets", "fp_num", "is_10k", "leverage", "roa", "cash_to_assets",
    "loss_indicator",
]


# =============================================================================
# Metrics
# =============================================================================

def pick_threshold_from_validation(y_true: np.ndarray, prob: np.ndarray) -> float:
    """Choose a threshold by maximizing validation F1."""
    best_t, best_f1 = 0.50, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        pred = (prob >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t


def metric_row(model_name: str, y_true: np.ndarray, prob: np.ndarray, threshold: float, note: str = "") -> dict:
    pred = (prob >= threshold).astype(int)
    row = {
        "model": model_name,
        "threshold": threshold,
        "n": int(len(y_true)),
        "positives": int(np.sum(y_true)),
        "prevalence": float(np.mean(y_true)),
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "brier": brier_score_loss(y_true, np.clip(prob, 0, 1)),
        "note": note,
    }
    if len(np.unique(y_true)) == 2:
        row["roc_auc"] = roc_auc_score(y_true, prob)
        row["pr_auc"] = average_precision_score(y_true, prob)
    else:
        row["roc_auc"] = np.nan
        row["pr_auc"] = np.nan
    return row


def topk_metrics(y_true: np.ndarray, prob: np.ndarray, model_name: str) -> List[dict]:
    rows = []
    n = len(y_true)
    prevalence = float(np.mean(y_true))
    total_pos = max(1, int(np.sum(y_true)))
    order = np.argsort(-prob)

    for frac in [0.005, 0.01, 0.02, 0.05, 0.10]:
        k = max(1, int(round(frac * n)))
        idx = order[:k]
        tp = int(np.sum(y_true[idx]))
        rows.append({
            "model": model_name,
            "top_fraction": frac,
            "top_k": k,
            "true_positives_captured": tp,
            "precision_at_k": tp / k,
            "recall_at_k": tp / total_pos,
            "lift_over_prevalence": (tp / k) / max(prevalence, 1e-12),
        })
    return rows


# =============================================================================
# Data loading
# =============================================================================

def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing dataset: {DATA_FILE}\n"
            "First run your enhanced accounting benchmark script to create this file."
        )

    df = pd.read_csv(DATA_FILE, low_memory=False)

    required = {"split", "aaer_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    numeric_features = [c for c in NUMERIC_FEATURES if c in df.columns]
    sic_features = [c for c in df.columns if str(c).startswith("sic2_")]
    all_ml_features = numeric_features + sic_features

    threshold_features = [c for c in DEEPCUT_THRESHOLD_FEATURES if c in df.columns]
    context_features = [c for c in DEEPCUT_CONTEXT_BASE if c in df.columns] + sic_features

    for c in numeric_features + sic_features:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    print("\nDataset loaded")
    print("--------------")
    print("Rows:", len(df))
    print("Unique firms:", df["cik"].nunique() if "cik" in df.columns else "NA")
    print("Popular ML features:", len(all_ml_features))
    print("DeepCut threshold features:", len(threshold_features))
    print("DeepCut context features:", len(context_features))
    print("\nOutcome by split:")
    print(pd.crosstab(df["split"], df["aaer_label"], margins=True))

    return df, all_ml_features, threshold_features, context_features


# =============================================================================
# Popular ML models
# =============================================================================

def fit_popular_ml(df: pd.DataFrame, features: List[str]):
    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()

    X_train, y_train = train[features], train["aaer_label"].astype(int).values
    X_val, y_val = val[features], val["aaer_label"].astype(int).values
    X_test, y_test = test[features], test["aaer_label"].astype(int).values

    pre = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])

    models = {
        "Logistic regression": LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Extra trees": ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Gradient boosting": GradientBoostingClassifier(
            n_estimators=350,
            learning_rate=0.025,
            max_depth=2,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=350,
            learning_rate=0.025,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=RANDOM_STATE,
        ),
    }

    pos = max(1, int(y_train.sum()))
    neg = max(1, int(len(y_train) - y_train.sum()))
    w_pos = len(y_train) / (2 * pos)
    w_neg = len(y_train) / (2 * neg)
    sample_weight = np.where(y_train == 1, w_pos, w_neg)

    rows, topk_rows = [], []
    pred_cols = {}

    for name, model in models.items():
        print(f"\nFitting {name}...")
        pipe = Pipeline([("pre", pre), ("clf", model)])

        if name in ["Gradient boosting", "HistGradientBoosting"]:
            pipe.fit(X_train, y_train, clf__sample_weight=sample_weight)
        else:
            pipe.fit(X_train, y_train)

        p_val = pipe.predict_proba(X_val)[:, 1]
        p_test = pipe.predict_proba(X_test)[:, 1]
        thr = pick_threshold_from_validation(y_val, p_val)

        rows.append(metric_row(name, y_test, p_test, thr, note="Popular ML benchmark; temporal test"))
        topk_rows.extend(topk_metrics(y_test, p_test, name))
        pred_cols[name] = p_test

    return pd.DataFrame(rows), pd.DataFrame(topk_rows), pred_cols


# =============================================================================
# DeepCut-inspired neural threshold model
# =============================================================================

def fit_deepcut(df: pd.DataFrame, threshold_features: List[str], context_features: List[str]):
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError as exc:
        raise ImportError("PyTorch is required. Install with: pip install torch") from exc

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()

    y_train = train["aaer_label"].astype(int).values.astype("float32")
    y_val = val["aaer_label"].astype(int).values.astype("float32")
    y_test = test["aaer_label"].astype(int).values.astype("float32")

    # Separate preprocessing for threshold variables and context variables.
    x_imp = SimpleImputer(strategy="median")
    x_scaler = StandardScaler()
    z_imp = SimpleImputer(strategy="median")
    z_scaler = StandardScaler(with_mean=False)

    X_train_raw = x_imp.fit_transform(train[threshold_features])
    X_val_raw = x_imp.transform(val[threshold_features])
    X_test_raw = x_imp.transform(test[threshold_features])

    X_train = x_scaler.fit_transform(X_train_raw).astype("float32")
    X_val = x_scaler.transform(X_val_raw).astype("float32")
    X_test = x_scaler.transform(X_test_raw).astype("float32")

    Z_train = z_scaler.fit_transform(z_imp.fit_transform(train[context_features])).astype("float32")
    Z_val = z_scaler.transform(z_imp.transform(val[context_features])).astype("float32")
    Z_test = z_scaler.transform(z_imp.transform(test[context_features])).astype("float32")

    class DeepCutAccounting(nn.Module):
        """
        DeepCut-inspired threshold/exceedance model for accounting data.

        For each accounting feature j:
          c_low_j(z) <= c_high_j(z)
          low_exceed_j  = ReLU(c_low_j(z) - x_j)
          high_exceed_j = ReLU(x_j - c_high_j(z))

        Both unusually low and unusually high values can increase the risk score.
        """
        def __init__(self, p: int, q: int, hidden: int = 64, dropout: float = 0.15):
            super().__init__()
            self.context_net = nn.Sequential(
                nn.Linear(q, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.base = nn.Linear(hidden, 1)
            self.mid = nn.Linear(hidden, p)
            self.log_width = nn.Linear(hidden, p)
            self.w_low = nn.Linear(hidden, p)
            self.w_high = nn.Linear(hidden, p)

        def forward(self, x, z):
            h = self.context_net(z)
            base_logit = self.base(h).squeeze(1)

            mid = self.mid(h)
            width = F.softplus(self.log_width(h)) + 0.05
            c_low = mid - width
            c_high = mid + width

            low_exceed = F.relu(c_low - x)
            high_exceed = F.relu(x - c_high)

            # Nonnegative weights keep the interpretation as risk-increasing exceedances.
            w_low = F.softplus(self.w_low(h))
            w_high = F.softplus(self.w_high(h))

            logit = base_logit + torch.sum(w_low * low_exceed + w_high * high_exceed, dim=1)
            return logit, c_low, c_high, w_low, w_high

    model = DeepCutAccounting(p=X_train.shape[1], q=Z_train.shape[1], hidden=64, dropout=0.15)
    opt = torch.optim.Adam(model.parameters(), lr=0.0015, weight_decay=1e-4)

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    Ztr = torch.tensor(Z_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32)
    Xva = torch.tensor(X_val, dtype=torch.float32)
    Zva = torch.tensor(Z_val, dtype=torch.float32)
    yva = torch.tensor(y_val, dtype=torch.float32)
    Xte = torch.tensor(X_test, dtype=torch.float32)
    Zte = torch.tensor(Z_test, dtype=torch.float32)

    ds = TensorDataset(Xtr, Ztr, ytr)
    loader = DataLoader(ds, batch_size=4096, shuffle=True)

    pos = max(1.0, float(y_train.sum()))
    neg = max(1.0, float(len(y_train) - y_train.sum()))
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32)

    best_state = None
    best_val_ap = -np.inf
    best_val_loss = np.inf
    patience = 18
    left = patience
    max_epochs = 120

    lambda_width = 0.001
    lambda_weight = 0.0005

    print("\nFitting DeepCut-inspired neural threshold model...")
    print("Threshold features:", ", ".join(threshold_features))

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_losses = []

        for xb, zb, yb in loader:
            opt.zero_grad()
            logits, c_low, c_high, w_low, w_high = model(xb, zb)

            bce = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
            width_penalty = torch.mean((c_high - c_low) ** 2)
            weight_penalty = torch.mean(w_low ** 2 + w_high ** 2)
            loss = bce + lambda_width * width_penalty + lambda_weight * weight_penalty

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            epoch_losses.append(float(loss.detach()))

        model.eval()
        with torch.no_grad():
            v_logits, _, _, _, _ = model(Xva, Zva)
            v_prob = torch.sigmoid(v_logits).numpy()
            v_loss = float(F.binary_cross_entropy_with_logits(v_logits, yva, pos_weight=pos_weight).detach())

        val_auc = roc_auc_score(y_val, v_prob) if len(np.unique(y_val)) == 2 else np.nan
        val_ap = average_precision_score(y_val, v_prob) if len(np.unique(y_val)) == 2 else np.nan

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"  epoch={epoch:03d} "
                f"train_loss={np.mean(epoch_losses):.4f} "
                f"val_loss={v_loss:.4f} "
                f"val_auc={val_auc:.4f} "
                f"val_pr_auc={val_ap:.4f}"
            )

        improved = (val_ap > best_val_ap + 1e-6) or (
            abs(val_ap - best_val_ap) <= 1e-6 and v_loss < best_val_loss
        )
        if improved:
            best_val_ap = val_ap
            best_val_loss = v_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            left = patience
        else:
            left -= 1
            if left <= 0:
                print(f"  early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_logits, _, _, _, _ = model(Xva, Zva)
        test_logits, c_low_test, c_high_test, w_low_test, w_high_test = model(Xte, Zte)
        p_val = torch.sigmoid(val_logits).numpy()
        p_test = torch.sigmoid(test_logits).numpy()

    thr = pick_threshold_from_validation(y_val.astype(int), p_val)
    summary = pd.DataFrame([
        metric_row(
            "DeepCut-inspired neural threshold",
            y_test.astype(int),
            p_test,
            thr,
            note="Neural accounting threshold/exceedance model; no SIVC",
        )
    ])
    topk = pd.DataFrame(topk_metrics(y_test.astype(int), p_test, "DeepCut-inspired neural threshold"))

    # Threshold summaries in original accounting-feature scale.
    c_low = c_low_test.numpy()
    c_high = c_high_test.numpy()
    w_low = w_low_test.numpy()
    w_high = w_high_test.numpy()

    means = x_scaler.mean_
    scales = x_scaler.scale_

    threshold_rows = []
    for j, feat in enumerate(threshold_features):
        low_orig = c_low[:, j] * scales[j] + means[j]
        high_orig = c_high[:, j] * scales[j] + means[j]
        threshold_rows.append({
            "feature": feat,
            "median_low_threshold_original_scale": float(np.nanmedian(low_orig)),
            "median_high_threshold_original_scale": float(np.nanmedian(high_orig)),
            "mean_low_threshold_original_scale": float(np.nanmean(low_orig)),
            "mean_high_threshold_original_scale": float(np.nanmean(high_orig)),
            "mean_low_exceedance_weight": float(np.nanmean(w_low[:, j])),
            "mean_high_exceedance_weight": float(np.nanmean(w_high[:, j])),
            "threshold_note": "Cutpoints are context-dependent; values are summarized across temporal test observations.",
        })

    pd.DataFrame(threshold_rows).to_csv(
        OUT_DIR / "deepcut_threshold_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pred = test[[c for c in ["cik", "name", "fy_num", "fp", "form", "aaer_label", "matched_future_aaer_year"] if c in test.columns]].copy()
    pred["prob_DeepCut_inspired_neural_threshold"] = p_test
    pred.to_csv(OUT_DIR / "deepcut_test_predictions.csv", index=False, encoding="utf-8-sig")

    return summary, topk, p_test


# =============================================================================
# Main
# =============================================================================

def main():
    start = time.time()

    df, all_ml_features, threshold_features, context_features = load_data()

    summary_ml, topk_ml, ml_probs = fit_popular_ml(df, all_ml_features)
    summary_deepcut, topk_deepcut, p_deepcut = fit_deepcut(df, threshold_features, context_features)

    summary = pd.concat([summary_ml, summary_deepcut], ignore_index=True)
    summary = summary.sort_values(["pr_auc", "roc_auc", "f1"], ascending=False)
    summary.to_csv(OUT_DIR / "model_comparison_summary.csv", index=False, encoding="utf-8-sig")

    topk = pd.concat([topk_ml, topk_deepcut], ignore_index=True)
    topk.to_csv(OUT_DIR / "topk_screening_metrics.csv", index=False, encoding="utf-8-sig")

    test = df[df["split"] == "test"].copy()
    pred = test[[c for c in ["cik", "name", "fy_num", "fp", "form", "aaer_label", "matched_future_aaer_year"] if c in test.columns]].copy()

    for name, p in ml_probs.items():
        pred["prob_" + re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")] = p

    if len(p_deepcut) == len(pred):
        pred["prob_DeepCut_inspired_neural_threshold"] = p_deepcut

    pred.to_csv(OUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")

    with open(OUT_DIR / "analysis_design.json", "w", encoding="utf-8") as f:
        json.dump({
            "paper_version": "Accounting journal version",
            "included_models": [
                "Logistic regression",
                "Random forest",
                "Extra trees",
                "Gradient boosting",
                "HistGradientBoosting",
                "DeepCut-inspired neural threshold",
            ],
            "excluded_models": ["SIVC", "teacher-student distillation"],
            "data_file": str(DATA_FILE),
            "outcome": "AAER enforcement-associated label from the enhanced dataset",
            "threshold_features": threshold_features,
            "context_features": context_features,
            "n_rows": int(len(df)),
            "n_test": int((df["split"] == "test").sum()),
        }, f, indent=2)

    print("\nAccounting-journal DeepCut + popular ML comparison")
    print("--------------------------------------------------")
    print(summary.to_string(index=False))

    print("\nTop-k screening metrics")
    print("-----------------------")
    print(topk.sort_values(["top_fraction", "recall_at_k"], ascending=[True, False]).to_string(index=False))

    print("\nSaved outputs:")
    print(" -", OUT_DIR / "model_comparison_summary.csv")
    print(" -", OUT_DIR / "topk_screening_metrics.csv")
    print(" -", OUT_DIR / "test_predictions.csv")
    print(" -", OUT_DIR / "deepcut_threshold_summary.csv")
    print(" -", OUT_DIR / "deepcut_test_predictions.csv")
    print(" -", OUT_DIR / "analysis_design.json")

    print(f"\nDONE in {(time.time() - start) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
