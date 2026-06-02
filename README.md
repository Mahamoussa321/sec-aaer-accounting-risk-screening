# SEC Accounting Enforcement Risk Screening Using Public 10-K and 10-Q Financial Statements

This repository contains reproducible code for the SEC/AAER accounting-enforcement risk screening study.

The study links public SEC 10-K and 10-Q financial-statement data to SEC Accounting and Auditing Enforcement Releases (AAERs), constructs accounting-risk features, evaluates popular machine-learning models, and compares them with a DeepCut-inspired neural thresholding model.

## Study design

- Unit of analysis: SEC filing-period observation from 10-K and 10-Q filings.
- Outcome: AAER enforcement-associated label within a forward-looking fiscal-year window.
- Temporal validation: train 2009-2018, validation 2019-2020, test 2021-2023.
- Metrics: ROC-AUC, PR-AUC, F1-score, and top-k screening recall/lift.

## Models

This accounting-journal version excludes the SIVC/teacher-student model, which is reserved for a separate machine-learning methods paper.

Included models:

1. Logistic regression
2. Random forest
3. Extra trees
4. Gradient boosting
5. HistGradientBoosting
6. DeepCut-inspired neural thresholding

## Repository structure

```text
scripts/      Reproducible analysis scripts
data/         Data README only; raw SEC data are excluded
results/      Small summary result tables
figures/      Final manuscript figures
```

## Data

Raw SEC datasets and large generated feature tables are not committed to GitHub.

Expected local files for a full rerun:

```text
sec_aaer_index.csv
sec_financial_features_2009_2026.csv
aaer_labels_reviewed.csv
```

## Reproducing the main analysis

```bash
pip install -r requirements.txt
python scripts/05_run_accounting_deepcut_ml_benchmark.py
python scripts/06_make_accounting_paper_figures.py
```

## DeepCut reference

Moussa, M., Ratterman, C., Zhang, W., Zhe, S., & Sun, Y. (2026). DeepCut: Adaptive neural network thresholds for precipitation phase partitioning. *Machine Learning: Earth*, 2(1), 015009. https://doi.org/10.1088/3049-4753/ae5066
