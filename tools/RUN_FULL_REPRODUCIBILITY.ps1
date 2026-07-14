# RUN_FULL_REPRODUCIBILITY.ps1
# Rebuild all derived outputs from raw SEC quarters using the already-clean repo.

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Repo = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $Repo)
$RawData = Join-Path $ProjectRoot "data"
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $ProjectRoot "reproducibility_run_$Timestamp.log"
Start-Transcript -Path $LogPath -Force | Out-Null

function Invoke-Checked {
    param([string]$Description, [scriptblock]$Command)
    Write-Host ""
    Write-Host "=== $Description ===" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "$Description failed with exit code $LASTEXITCODE" }
}

try {
    Write-Host "SEC-AAER FULL REPRODUCIBILITY RUN" -ForegroundColor Cyan
    Write-Host "Repository: $Repo"
    Write-Host "Raw SEC data: $RawData"

    $Quarters = Get-ChildItem $RawData -Directory | Where-Object { $_.Name -match '^\d{4}q[1-4]$' } | Sort-Object Name
    if ($Quarters.Count -ne 69 -or $Quarters[0].Name -ne '2009q1' -or $Quarters[-1].Name -ne '2026q1') {
        throw "Expected 69 quarterly folders from 2009q1 through 2026q1."
    }
    foreach ($Folder in $Quarters) {
        if (-not (Test-Path (Join-Path $Folder.FullName 'sub.txt')) -or -not (Test-Path (Join-Path $Folder.FullName 'num.txt'))) {
            throw "Missing sub.txt or num.txt in $($Folder.FullName)"
        }
    }

    if (-not (Test-Path $Python)) {
        if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw "Python was not found." }
        & python -m venv (Join-Path $ProjectRoot '.venv')
        if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed." }
    }
    Invoke-Checked "Install locked Python environment" {
        & $Python -m pip install --upgrade pip
        & $Python -m pip install -r (Join-Path $Repo 'requirements-lock.txt')
    }

    $Index = Join-Path $Repo 'sec_aaer_index.csv'
    if (-not (Test-Path $Index)) {
        throw "Missing local AAER index snapshot: $Index. Restore it from the project archive or run the AAER downloader with SEC_USER_AGENT set."
    }
    Copy-Item (Join-Path $Repo 'data\labels\aaer_labels_reviewed.csv') (Join-Path $Repo 'aaer_labels_reviewed.csv') -Force
    Remove-Item (Join-Path $Repo 'outputs') -Recurse -Force -ErrorAction SilentlyContinue
    Get-ChildItem $Repo -Filter 'sec_financial_features_*.csv' -File -ErrorAction SilentlyContinue | Remove-Item -Force

    Push-Location $Repo
    try {
        Invoke-Checked "Rebuild feature table and analysis dataset" {
            & $Python '.\scripts\data\build_analysis_dataset.py' `
                --extracted-root $RawData --repo-root $Repo `
                --reviewed-labels '.\data\labels\aaer_labels_reviewed.csv' `
                --aaer-index '.\sec_aaer_index.csv' `
                --start-year 2009 --raw-end-year 2026 `
                --analysis-end-year 2023 --label-window-years 3 --force-rebuild
        }
        Invoke-Checked "Run main benchmark" { & $Python '.\scripts\analysis\run_main_benchmark.py' }
        Invoke-Checked "Regenerate main figures" { & $Python '.\scripts\analysis\make_main_figures.py' }
        Invoke-Checked "Run publication-strengthening analyses" {
            & $Python '.\scripts\analysis\publication_strengthening.py' `
                --project-root $Repo --bootstrap-reps 1000 --cold-start-folds 5 `
                --full-seeds '20260531,20260532,20260533,20260534,20260535' `
                --ablation-seeds '20260531,20260532,20260533' `
                --max-epochs 120 --patience 18 --device auto
        }
        Invoke-Checked "Run final targeted corrections" {
            & $Python '.\scripts\analysis\targeted_final_corrections.py' `
                --project-root $Repo --mature-max-fy 2022 --bootstrap-reps 1000 `
                --seeds '20260531,20260532,20260533,20260534,20260535' `
                --max-epochs 120 --patience 18 --device auto
        }
        Invoke-Checked "Finalize compact release outputs" { & $Python '.\scripts\finalize_release.py' --project-root $Repo }
        Invoke-Checked "Validate regenerated results" { & $Python '.\scripts\validate_reproducibility.py' --project-root $Repo --strict }
    }
    finally { Pop-Location }

    Write-Host ""
    Write-Host "FULL REPRODUCIBILITY RUN COMPLETED" -ForegroundColor Green
    Write-Host "Log: $LogPath"
}
finally {
    try { Stop-Transcript | Out-Null } catch {}
}
