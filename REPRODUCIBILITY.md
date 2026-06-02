# Reproducibility notes

This repository keeps public code, compact result summaries, and final figures while excluding large raw SEC data.

To rerun the main analysis, place or generate these files in the repository root:

```text
sec_aaer_index.csv
sec_financial_features_2009_2026.csv
aaer_labels_reviewed.csv
```

Then run:

```bash
python scripts/05_run_accounting_deepcut_ml_benchmark.py
python scripts/06_make_accounting_paper_figures.py
```

Large raw data and prediction-level outputs are excluded to keep the repository clean.
