# Final Targeted Corrections for the SEC-AAER Manuscript

Generated: 2026-07-15 00:37:23

## What this run corrected

1. The primary temporal test is now limited to FY2021-FY2022, for which the three-year forward AAER outcome window is mature under the available enforcement data.
2. FY2023 is reported separately as an administratively incomplete follow-up cohort and is not used for the primary three-year inference.
3. The 0-, 1-, 2-, and 3-year label-window sensitivity uses the identical DeepCut seeds 20260531, 20260532, 20260533, 20260534, 20260535 for every horizon and the identical mature test cohort.
4. Logistic regression is deterministic under the specified solver; one fit per label horizon is reused across seed rows and marked in the output table.

## Outcome-mature temporal test

The primary mature test contains 51,627 filing-period observations from 7,326 firms, including 123 AAER-positive observations (prevalence 0.2382%).
The separately reported FY2023 cohort contains 24,296 observations and 44 observed positives, but later AAERs can still change its three-year labels.

- DeepCut raw ranking: ROC-AUC 0.7451, PR-AUC 0.00981, top-5% recall 0.2520.
- Logistic raw ranking: ROC-AUC 0.6839, PR-AUC 0.00711, top-5% recall 0.1789.
- The mature-cohort DeepCut-minus-logistic ROC-AUC difference was 0.0612 (firm-clustered 95% CI 0.0038 to 0.1188; bootstrap p=0.034).

Validation-selected calibration for DeepCut was isotonic; mature-cohort Brier score 0.002390 and ECE 0.000965.
Validation-selected calibration for logistic regression was isotonic; mature-cohort Brier score 0.002375 and ECE 0.000989.

## Five-seed mature-cohort stability

- roc_auc: mean 0.760480, SD 0.013437, range 0.745059-0.779180 across 5 seeds.
- pr_auc: mean 0.015091, SD 0.010943, range 0.008347-0.034464 across 5 seeds.
- brier: mean 0.206767, SD 0.026216, range 0.177716-0.235778 across 5 seeds.
- recall_top_0p02: mean 0.091057, SD 0.038306, range 0.040650-0.138211 across 5 seeds.
- recall_top_0p05: mean 0.229268, SD 0.030092, range 0.186992-0.260163 across 5 seeds.
- recall_top_0p1: mean 0.328455, SD 0.023423, range 0.300813-0.357724 across 5 seeds.

## Consistent-seed label-window sensitivity

- 0-year window: DeepCut ROC-AUC 0.4922 ± 0.0187, PR-AUC 0.00066 ± 0.00002; logistic ROC-AUC 0.5467, PR-AUC 0.00074.
- 1-year window: DeepCut ROC-AUC 0.6691 ± 0.0196, PR-AUC 0.00364 ± 0.00074; logistic ROC-AUC 0.6201, PR-AUC 0.00250.
- 2-year window: DeepCut ROC-AUC 0.6798 ± 0.0168, PR-AUC 0.00554 ± 0.00014; logistic ROC-AUC 0.6278, PR-AUC 0.00428.
- 3-year window: DeepCut ROC-AUC 0.7605 ± 0.0134, PR-AUC 0.01509 ± 0.01094; logistic ROC-AUC 0.6839, PR-AUC 0.00711.

## Manuscript decision

After these outputs are reviewed for numerical consistency, no additional large computational analysis is required for the current journal submission plan. The remaining work is manuscript revision: use the mature-cohort counts, describe FY2023 as incomplete follow-up, and state statistical uncertainty exactly as reported by the firm-clustered intervals.

## Principal output tables

- `tables/outcome_maturity_cohort_summary.csv`
- `tables/mature_temporal_test_raw_metrics.csv`
- `tables/mature_temporal_test_selected_calibration_metrics.csv`
- `tables/mature_temporal_test_firm_clustered_confidence_intervals.csv`
- `tables/mature_temporal_test_paired_model_differences.csv`
- `tables/mature_deepcut_five_seed_summary.csv`
- `tables/consistent_seed_label_window_metrics.csv`
- `tables/consistent_seed_label_window_summary.csv`
