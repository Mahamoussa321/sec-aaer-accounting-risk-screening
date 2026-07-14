# Interpretable neural screening of SEC accounting-enforcement risk

This repository contains the reproducible code, adjudicated issuer–AAER links, compact result tables, and figures for a study that screens public 10-K and 10-Q financial statements for subsequent SEC Accounting and Auditing Enforcement Release (AAER) association.

## Final study design

- **Raw source:** SEC Financial Statement Data Sets, 2009Q1–2026Q1.
- **Unit:** filing-period observation derived from 10-K and 10-Q submissions.
- **Labels:** 495 candidate issuer–AAER links were adjudicated; 104 were retained and 391 rejected.
- **Modeling split:** train 2009–2018, validation 2019–2020, temporal test 2021–2023.
- **Primary inference:** FY2021–FY2022 only, because the three-year forward outcome window is mature for those cohorts.
- **FY2023:** descriptive incomplete-follow-up cohort, not part of primary three-year inference.

The rebuilt dataset contains 331,388 observations, 14,047 firms, and 904 positive filing-period outcomes. The mature FY2021–FY2022 test contains 51,627 observations, 7,326 firms, and 123 positives.

## Primary result

On the mature temporal test, the DeepCut-inspired threshold model achieved ROC-AUC 0.7569 and PR-AUC 0.0124, compared with ROC-AUC 0.6837 and PR-AUC 0.0071 for logistic regression. The firm-clustered paired ROC-AUC difference was 0.0733 (95% CI 0.0224–0.1265; bootstrap p=0.012). Top-k and PR-AUC differences should be interpreted with their reported uncertainty.

## Repository map

```text
config/                 Locked analysis design
 data/labels/            Adjudicated public AAER-link decisions
 docs/                   Provenance, labeling, and repository documentation
 figures/                Manuscript-facing figures from the latest clean run
 results/                Small manuscript-facing tables and run manifest
 scripts/data/           SEC download and dataset-construction code
 scripts/analysis/       Main benchmark, figures, robustness, and final corrections
 tools/                   One-command Windows reproducibility runner
```

Raw SEC files, the 331k-row analysis dataset, predictions, model caches, and virtual environments are intentionally excluded from GitHub.

## Full clean rerun

On Windows PowerShell, use `tools/RUN_FULL_REPRODUCIBILITY.ps1`. It validates the 69 quarterly folders, rebuilds every derived dataset and model output from scratch, runs the strengthening and final analyses, validates the expected counts and results, and prepares the compact tracked release files.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for manual commands and [docs/DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md) for source details.

## Scientific cautions

The system is a **screening and prioritization tool**, not a determination of fraud or liability. AAER timing is delayed, the outcome is extremely rare, and probability calibration and firm-clustered uncertainty are required for responsible interpretation.

## License

No open-source license has yet been selected by the authors. Public availability does not by itself grant reuse rights beyond those provided by applicable law.
