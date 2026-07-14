#!/usr/bin/env python
"""Download and extract SEC Financial Statement Data Set quarterly archives.

The SEC requests a descriptive User-Agent. Set SEC_USER_AGENT, for example:
    $env:SEC_USER_AGENT = "Maha Moussa maha.moussa@usu.edu"

Existing complete quarter folders are skipped unless --force is supplied.
"""
from __future__ import annotations

import argparse
import io
import os
import time
import zipfile
from pathlib import Path

import requests

BASE = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/{year}q{quarter}.zip"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", required=True)
    p.add_argument("--start-year", type=int, default=2009)
    p.add_argument("--end-year", type=int, default=2026)
    p.add_argument("--end-quarter", type=int, default=1)
    p.add_argument("--user-agent", default=os.environ.get("SEC_USER_AGENT"))
    p.add_argument("--force", action="store_true")
    p.add_argument("--delay", type=float, default=0.25)
    return p.parse_args()


def complete(folder: Path) -> bool:
    return (folder / "sub.txt").exists() and (folder / "num.txt").exists()


def main() -> int:
    args = parse_args()
    if not args.user_agent:
        raise SystemExit(
            "Set SEC_USER_AGENT to a descriptive value containing a name and contact email."
        )
    root = Path(args.output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": args.user_agent, "Accept-Encoding": "gzip, deflate"})

    downloaded = skipped = 0
    for year in range(args.start_year, args.end_year + 1):
        last_q = args.end_quarter if year == args.end_year else 4
        for quarter in range(1, last_q + 1):
            folder = root / f"{year}q{quarter}"
            if complete(folder) and not args.force:
                print(f"SKIP complete {folder.name}")
                skipped += 1
                continue
            url = BASE.format(year=year, quarter=quarter)
            print(f"GET  {url}")
            response = session.get(url, timeout=120)
            response.raise_for_status()
            folder.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                zf.extractall(folder)
            if not complete(folder):
                raise RuntimeError(f"Archive did not create sub.txt and num.txt in {folder}")
            downloaded += 1
            time.sleep(args.delay)

    print(f"Completed. Downloaded: {downloaded}; skipped: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
