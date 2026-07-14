#!/usr/bin/env python
"""Copy manuscript-facing outputs into a clean, tracked release tree."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", default=".")
    return p.parse_args()


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_files(src: Path, dst: Path, patterns: tuple[str, ...]) -> list[Path]:
    copied: list[Path] = []
    if not src.exists():
        return copied
    dst.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        for item in sorted(src.glob(pattern)):
            if item.is_file():
                target = dst / item.name
                shutil.copy2(item, target)
                copied.append(target)
    return copied


def relative_or_name(value: object, root: Path) -> object:
    if not isinstance(value, str):
        return value
    try:
        p = Path(value)
        if p.is_absolute():
            try:
                return p.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                return p.name
    except OSError:
        pass
    return value.replace("\\", "/")


def sanitize_json(src: Path, dst: Path, root: Path) -> None:
    data = json.loads(src.read_text(encoding="utf-8-sig"))

    def walk(obj):
        if isinstance(obj, dict):
            return {k: walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [walk(v) for v in obj]
        return relative_or_name(obj, root)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(walk(data), indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    outputs = root / "outputs"
    results = root / "results"
    figures = root / "figures"

    for d in (results / "main", results / "strengthening", results / "final",
              figures / "main", figures / "strengthening", figures / "final"):
        clean_dir(d)

    main_bench = outputs / "accounting_deepcut_ml_benchmark"
    main_data = outputs / "aaer_enhanced_accounting_benchmark"
    main_fig = outputs / "accounting_paper_figures"
    for name in ("model_comparison_summary.csv", "topk_screening_metrics.csv", "deepcut_threshold_summary.csv"):
        shutil.copy2(main_bench / name, results / "main" / name)
    sanitize_json(main_bench / "analysis_design.json", results / "main" / "analysis_design.json", root)
    sanitize_json(main_data / "dataset_build_summary.json", results / "main" / "dataset_build_summary.json", root)
    copy_files(main_fig, figures / "main", ("*.png", "*.pdf", "figure_captions.txt"))

    strength = outputs / "publication_strengthening"
    copy_files(strength / "tables", results / "strengthening", ("*.csv",))
    shutil.copy2(strength / "PUBLICATION_STRENGTHENING_REPORT.md", results / "strengthening" / "PUBLICATION_STRENGTHENING_REPORT.md")
    sanitize_json(strength / "run_configuration.json", results / "strengthening" / "run_configuration.json", root)
    copy_files(strength / "figures", figures / "strengthening", ("*.png", "*.pdf"))

    final = outputs / "final_targeted_corrections"
    copy_files(final / "tables", results / "final", ("*.csv",))
    shutil.copy2(final / "FINAL_TARGETED_CORRECTIONS_REPORT.md", results / "final" / "FINAL_TARGETED_CORRECTIONS_REPORT.md")
    sanitize_json(final / "run_configuration.json", results / "final" / "run_configuration.json", root)
    copy_files(final / "figures", figures / "final", ("*.png", "*.pdf"))

    packages = {}
    for name in ["pandas", "numpy", "scikit-learn", "torch", "matplotlib", "rapidfuzz", "requests", "beautifulsoup4", "lxml", "tqdm", "scipy", "joblib"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    scientific_files = []
    for base in [root / "scripts", root / "config", root / "data" / "labels", results, figures]:
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.name != "reproducibility_manifest.json":
                scientific_files.append({
                    "path": p.relative_to(root).as_posix(),
                    "size_bytes": p.stat().st_size,
                    "sha256": sha256(p),
                })
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "files": scientific_files,
    }
    (results / "reproducibility_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    index_lines = [
        "# Results index",
        "",
        "The tracked results are small manuscript-facing tables and figures. Large datasets, predictions, model caches, and raw SEC files remain local and are excluded by `.gitignore`.",
        "",
        "- `main/`: temporal benchmark summaries and dataset counts.",
        "- `strengthening/`: clustered inference, calibration, cold-start validation, ablations, 10-K robustness, and preliminary horizon sensitivity.",
        "- `final/`: outcome-mature FY2021–FY2022 inference and consistent-seed horizon sensitivity.",
        "- `reproducibility_manifest.json`: source/result hashes and package versions from the latest clean run.",
    ]
    (results / "RESULTS_INDEX.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print("Release-facing results and figures finalized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
