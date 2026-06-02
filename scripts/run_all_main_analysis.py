#!/usr/bin/env python
# -*- coding: utf-8 -*-
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

steps = [
    ["python", "scripts/05_run_accounting_deepcut_ml_benchmark.py"],
    ["python", "scripts/06_make_accounting_paper_figures.py"],
]

for step in steps:
    print("\nRunning:", " ".join(step))
    subprocess.run(step, cwd=ROOT, check=True)

print("\nDONE.")
