#!/usr/bin/env python
"""Download the SEC Accounting and Auditing Enforcement Releases index."""
from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.sec.gov/enforcement-litigation/accounting-auditing-enforcement-releases"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="sec_aaer_index.csv")
    p.add_argument("--user-agent", default=os.environ.get("SEC_USER_AGENT"))
    p.add_argument("--max-pages", type=int, default=100)
    p.add_argument("--delay", type=float, default=0.5)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.user_agent:
        raise SystemExit(
            "Set SEC_USER_AGENT to a descriptive value containing a name and contact email."
        )
    session = requests.Session()
    session.headers.update({"User-Agent": args.user_agent, "Accept-Encoding": "gzip, deflate"})
    rows: list[dict[str, str]] = []
    empty_pages = 0

    for page in range(args.max_pages):
        url = BASE_URL if page == 0 else f"{BASE_URL}?page={page}"
        response = session.get(url, timeout=60)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        page_rows = soup.select("table tbody tr")
        if not page_rows:
            empty_pages += 1
            if empty_pages >= 2:
                break
            continue
        empty_pages = 0
        before = len(rows)
        for tr in page_rows:
            cells = tr.find_all("td")
            if len(cells) < 2:
                continue
            date_text = cells[0].get_text(" ", strip=True)
            detail_text = cells[1].get_text(" ", strip=True)
            links = [urljoin(BASE_URL, a["href"]) for a in tr.find_all("a", href=True)]
            rows.append(
                {
                    "date": date_text,
                    "details": detail_text,
                    "aaer_numbers": "; ".join(re.findall(r"AAER-\d+", detail_text)),
                    "release_numbers": "; ".join(re.findall(r"Release No\.\s*[^;|]+", detail_text)),
                    "links": " | ".join(links),
                    "source_page": url,
                }
            )
        print(f"Page {page}: {len(rows) - before} rows")
        time.sleep(args.delay)

    if not rows:
        raise RuntimeError("No AAER rows were retrieved; the SEC page structure may have changed.")
    df = pd.DataFrame(rows).drop_duplicates()
    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df):,} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
