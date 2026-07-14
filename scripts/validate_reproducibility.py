#!/usr/bin/env python
"""Validate repository organization and, after a full run, scientific outputs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

EXPECTED_FILES = [
    "README.md", "REPRODUCIBILITY.md", "CITATION.cff", "requirements.txt",
    "requirements-lock.txt", "environment.yml", "config/analysis_config.json",
    "data/labels/aaer_labels_reviewed.csv",
    "scripts/data/download_sec_aaer_index.py",
    "scripts/data/download_sec_financial_statement_data.py",
    "scripts/data/build_analysis_dataset.py",
    "scripts/analysis/run_main_benchmark.py",
    "scripts/analysis/make_main_figures.py",
    "scripts/analysis/publication_strengthening.py",
    "scripts/analysis/targeted_final_corrections.py",
    "scripts/finalize_release.py",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", default=".")
    p.add_argument("--repository-only", action="store_true")
    p.add_argument("--strict", action="store_true")
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def numeric(row: dict[str, str], key: str) -> float:
    return float(row[key])


def near(value: float, expected: float, tolerance: float) -> bool:
    return math.isfinite(value) and abs(value - expected) <= tolerance


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    failures: list[str] = []
    warnings: list[str] = []

    for rel in EXPECTED_FILES:
        if not (root / rel).exists():
            failures.append(f"Missing required file: {rel}")

    labels_path = root / "data/labels/aaer_labels_reviewed.csv"
    if labels_path.exists():
        labels = read_csv(labels_path)
        values = [str(r.get("keep_label", "")).strip() for r in labels]
        if len(labels) != 495:
            failures.append(f"Expected 495 adjudication rows; found {len(labels)}")
        invalid = [v for v in values if v not in {"0", "1"}]
        if invalid:
            failures.append(f"Invalid or blank keep_label values: {len(invalid)}")
        if values.count("1") != 104 or values.count("0") != 391:
            failures.append(
                f"Expected 104 accepted and 391 rejected links; found {values.count('1')} and {values.count('0')}"
            )

    ignored_runtime_roots = {".git", "outputs", "cache", "logs", ".venv", "venv", "__pycache__"}
    ignored_runtime_names = {"sec_aaer_index.csv", "aaer_labels_reviewed.csv"}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in ignored_runtime_roots for part in rel.parts):
            continue
        if p.name in ignored_runtime_names or p.name.startswith("sec_financial_features_") or "predictions" in p.name.lower():
            continue
        if p.stat().st_size > 95 * 1024 * 1024:
            failures.append(f"File exceeds 95 MB GitHub safety limit: {rel}")
        if p.suffix.lower() in {".py", ".ps1", ".md", ".yml", ".yaml", ".json", ".txt"}:
            text = p.read_text(encoding="utf-8", errors="ignore")
            secret_patterns = [
                r"ghp_[A-Za-z0-9]{30,}", r"github_pat_[A-Za-z0-9_]{30,}",
                r"AKIA[0-9A-Z]{16}", r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
            ]
            for pattern in secret_patterns:
                if re.search(pattern, text):
                    failures.append(f"Possible secret in {p.relative_to(root)}")

    downloader = root / "scripts/data/download_sec_financial_statement_data.py"
    if downloader.exists():
        text = downloader.read_text(encoding="utf-8")
        if "requests" not in text or "ZipFile" not in text:
            failures.append("SEC quarterly downloader is not functional code")

    if not args.repository_only:
        summary_path = root / "results/main/dataset_build_summary.json"
        if not summary_path.exists():
            failures.append("Missing finalized dataset summary")
        else:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            expected = {"rows": 331388, "unique_firms": 14047, "positive_rows": 904}
            for key, val in expected.items():
                if int(summary.get(key, -1)) != val:
                    failures.append(f"Dataset {key}: expected {val}, found {summary.get(key)}")

        mature_path = root / "results/final/mature_temporal_test_raw_metrics.csv"
        if mature_path.exists():
            rows = read_csv(mature_path)
            by_model = {r["model"]: r for r in rows}
            deep = by_model.get("DeepCut-inspired neural threshold")
            logit = by_model.get("Logistic regression")
            if not deep or not logit:
                failures.append("Mature metrics do not include DeepCut and logistic regression")
            else:
                if int(float(deep["n"])) != 51627 or int(float(deep["positives"])) != 123:
                    failures.append("Mature cohort should contain 51,627 rows and 123 positives")
                checks = [
                    ("DeepCut ROC-AUC", numeric(deep, "roc_auc"), 0.756933, 0.025),
                    ("DeepCut PR-AUC", numeric(deep, "pr_auc"), 0.012439, 0.010),
                    ("Logistic ROC-AUC", numeric(logit, "roc_auc"), 0.683652, 0.005),
                ]
                for name, value, expected_value, tol in checks:
                    if not near(value, expected_value, tol):
                        failures.append(f"{name} outside tolerance: {value:.6f}")
        else:
            failures.append("Missing mature temporal test metrics")

        paired_path = root / "results/final/mature_temporal_test_paired_model_differences.csv"
        if paired_path.exists():
            rows = read_csv(paired_path)
            target = next((r for r in rows if "minus Logistic regression" in r.get("comparison", "") and r.get("metric") == "roc_auc"), None)
            if not target:
                failures.append("Missing paired DeepCut-minus-logistic ROC-AUC comparison")
            else:
                diff = numeric(target, "estimate_difference")
                if not near(diff, 0.073281, 0.025):
                    failures.append(f"Paired ROC-AUC difference outside tolerance: {diff:.6f}")
                if numeric(target, "ci_lower") <= 0:
                    warnings.append("Regenerated paired ROC-AUC lower confidence bound is not above zero")
        else:
            failures.append("Missing paired model-difference table")

        manifest = root / "results/reproducibility_manifest.json"
        if not manifest.exists():
            failures.append("Missing reproducibility manifest")

    print("\nREPRODUCIBILITY VALIDATION")
    print("=" * 27)
    for warning in warnings:
        print(f"WARNING: {warning}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        print(f"\nValidation failed with {len(failures)} issue(s).")
        return 1
    print("PASS: repository organization, labels, file safety, and requested scientific checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
