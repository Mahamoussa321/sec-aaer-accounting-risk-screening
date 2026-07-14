# Final results summary

## Dataset

- 331,388 filing-period observations
- 14,047 firms
- 904 AAER-positive observations
- Mature FY2021–FY2022 test: 51,627 observations, 7,326 firms, 123 positives
- FY2023: 24,296 observations and 44 positives, reported descriptively because three-year follow-up is incomplete

## Mature temporal test

| Model | ROC-AUC | PR-AUC | Top-5% recall | Top-10% recall |
|---|---:|---:|---:|---:|
| DeepCut-inspired neural threshold | 0.7569 | 0.0124 | 0.1789 | 0.3089 |
| Logistic regression | 0.6837 | 0.0071 | 0.1789 | 0.2358 |
| Random forest | 0.6721 | 0.0047 | 0.1220 | 0.2195 |

DeepCut minus logistic ROC-AUC: 0.0733; firm-clustered 95% CI 0.0224–0.1265; bootstrap p=0.012. PR-AUC and top-k differences remain more uncertain.

Across five seeds, DeepCut ROC-AUC was 0.7585 ± 0.0146. The complete tables in `results/final/` are the authoritative source for manuscript values.
