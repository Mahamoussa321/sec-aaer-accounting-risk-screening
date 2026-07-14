# Reproducibility instructions

## Supported reference environment

- Windows 10/11 with PowerShell 5.1 or later
- Python 3.13.5
- Exact package versions in `requirements-lock.txt`
- CPU execution is the reference for the most reproducible seeded neural results

## Raw-data layout

The full runner expects a project folder containing:

```text
sec_financial_statements/
  data/
    2009q1/sub.txt
    2009q1/num.txt
    ...
    2026q1/sub.txt
    2026q1/num.txt
  .venv/
  work/sec-aaer-accounting-risk-screening/
```

There are 69 quarter folders from 2009Q1 through 2026Q1. Each must include at least `sub.txt` and `num.txt`.

## One-command clean rerun

From the repository root:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.	ools\RUN_FULL_REPRODUCIBILITY.ps1
```

The runner archives the previous generated outputs outside the repository, clears derived files, and executes:

1. dataset reconstruction from quarterly SEC files;
2. the six-model temporal benchmark;
3. main figure generation;
4. firm-clustered inference, calibration, cold-start validation, DeepCut seed/ablation analysis, 10-K-only robustness, and horizon sensitivity;
5. mature FY2021–FY2022 inference and consistent-five-seed horizon sensitivity;
6. compact result/figure finalization;
7. repository and numerical validation.

## Manual commands

```powershell
$Python = "<project>\.venv\Scripts\python.exe"
$Repo   = "<project>\work\sec-aaer-accounting-risk-screening"
$Raw    = "<project>\data"

& $Python "$Repo\scripts\datauild_analysis_dataset.py" `
  --extracted-root $Raw --repo-root $Repo `
  --start-year 2009 --raw-end-year 2026 `
  --analysis-end-year 2023 --label-window-years 3 --force-rebuild

Push-Location $Repo
& $Python .\scriptsnalysisun_main_benchmark.py
& $Python .\scriptsnalysis\make_main_figures.py
& $Python .\scriptsnalysis\publication_strengthening.py `
  --project-root $Repo --bootstrap-reps 1000 --cold-start-folds 5 `
  --full-seeds 20260531,20260532,20260533,20260534,20260535 `
  --ablation-seeds 20260531,20260532,20260533 `
  --max-epochs 120 --patience 18 --device auto
& $Python .\scriptsnalysis	argeted_final_corrections.py `
  --project-root $Repo --mature-max-fy 2022 --bootstrap-reps 1000 `
  --seeds 20260531,20260532,20260533,20260534,20260535 `
  --max-epochs 120 --patience 18 --device auto
& $Python .\scriptsinalize_release.py --project-root $Repo
& $Python .\scriptsalidate_reproducibility.py --project-root $Repo --strict
Pop-Location
```

## Determinism

All reported neural analyses use explicit seeds. Small floating-point differences can occur across operating systems, BLAS implementations, PyTorch versions, and CPU/GPU hardware. The validator requires exact sample counts and uses documented tolerances for neural metrics.

## Data access

The raw quarterly archives can be downloaded with `scripts/data/download_sec_financial_statement_data.py`. Set a descriptive `SEC_USER_AGENT` before accessing SEC servers. The adjudicated label file is versioned in `data/labels/`.
