import re
import time
from urllib.parse import urljoin
from io import StringIO

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL = "https://www.sec.gov/enforcement-litigation/accounting-auditing-enforcement-releases"

HEADERS = {
    "User-Agent": "Academic research contact: maha.moussa@usu.edu"
}

all_rows = []

# The SEC page shows about 34 pages when listing all AAER items.
# Page 0 is the first page, page 1 is the second page, etc.
for page in tqdm(range(34)):
    if page == 0:
        url = BASE_URL
    else:
        url = f"{BASE_URL}?page={page}"

    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    rows = soup.select("table tbody tr")

    for tr in rows:
        cells = tr.find_all("td")
        if len(cells) < 2:
            continue

        date_text = cells[0].get_text(" ", strip=True)
        detail_text = cells[1].get_text(" ", strip=True)

        links = []
        for a in tr.find_all("a", href=True):
            links.append(urljoin(BASE_URL, a["href"]))

        aaer_numbers = re.findall(r"AAER-\d+", detail_text)
        release_numbers = re.findall(r"Release No\.\s*[^S]+", detail_text)

        all_rows.append({
            "date": date_text,
            "details": detail_text,
            "aaer_numbers": "; ".join(aaer_numbers),
            "release_numbers": "; ".join(release_numbers),
            "links": " | ".join(links),
            "source_page": url
        })

    time.sleep(0.5)

df = pd.DataFrame(all_rows)
df.to_csv("sec_aaer_index.csv", index=False, encoding="utf-8-sig")

print("Saved:", "sec_aaer_index.csv")
print("Rows:", len(df))
print(df.head())