# SEC–AAER Publication-Strengthening Analysis

Generated: 2026-07-14 23:39:47

## Analyses completed

- Firm-clustered bootstrap confidence intervals and paired model differences.
- Raw and validation-based Platt/isotonic calibration diagnostics.
- 10-K, 10-Q, year-specific, and naturally unseen-firm subgroup diagnostics.
- Firm-disjoint temporal cross-fitting for cold-start generalization.
- DeepCut repeated-seed stability and structural ablations.
- 10-K-only model retraining.
- AAER label-window sensitivity for 0–3 future years.

## Important unseen-firm finding

The naturally occurring test subset of firms absent from the training period contains no positive AAER-associated observations. Standard ROC-AUC and PR-AUC are therefore undefined for that subset. The firm-disjoint temporal cross-fitting analysis is the valid cold-start generalization result and should be reported instead.

## Validation-selected calibration

- **DeepCut-inspired neural threshold**: isotonic calibration; test ROC-AUC 0.7395, PR-AUC 0.00779, Brier 0.002218, ECE 0.001162.
- **Extra trees**: isotonic calibration; test ROC-AUC 0.6508, PR-AUC 0.00459, Brier 0.002247, ECE 0.001073.
- **Gradient boosting**: isotonic calibration; test ROC-AUC 0.6173, PR-AUC 0.00298, Brier 0.002196, ECE 0.001178.
- **HistGradientBoosting**: isotonic calibration; test ROC-AUC 0.6498, PR-AUC 0.00350, Brier 0.002215, ECE 0.001221.
- **Logistic regression**: isotonic calibration; test ROC-AUC 0.6704, PR-AUC 0.00481, Brier 0.002199, ECE 0.001218.
- **Random forest**: isotonic calibration; test ROC-AUC 0.6485, PR-AUC 0.00382, Brier 0.002227, ECE 0.001244.

## Firm-disjoint temporal cross-fitting

- **Logistic regression**: ROC-AUC 0.6156, PR-AUC 0.00421, top-5% recall 0.1257.
- **DeepCut-inspired neural threshold**: ROC-AUC 0.6673, PR-AUC 0.00454, top-5% recall 0.0479.

## DeepCut stability and ablations

- **adaptive high only** (3 seeds): ROC-AUC 0.7564 ± 0.0159; PR-AUC 0.00845 ± 0.00158.
- **context only mlp** (3 seeds): ROC-AUC 0.7359 ± 0.0198; PR-AUC 0.00892 ± 0.00093.
- **fixed global two sided** (3 seeds): ROC-AUC 0.7242 ± 0.0055; PR-AUC 0.00626 ± 0.00057.
- **full adaptive two sided** (5 seeds): ROC-AUC 0.7572 ± 0.0138; PR-AUC 0.01143 ± 0.00828.

## Submission caution

These analyses strengthen statistical validation, but the manuscript must be revised to use the corrected 904-positive dataset and the new uncertainty, calibration, cold-start, ablation, and sensitivity results. The original manuscript numbers should not be retained.
