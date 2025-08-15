"""
Scrape Craigslist listings for Computers & Computer Parts.

Usage (local):
    python scrape_data.py --region https://rochester.craigslist.org --limit 400

Outputs:
    data/computers.csv
    data/computer_parts.csv
    data/combined_data.csv  (with columns: title, description, label, text)

Notes:
- Be polite: default sleep(1.0) between requests.
- For regions, use the root like: https://newyork.craigslist.org  or https://sfbay.craigslist.org
"""

from __future__ import annotations
import time, argparse, sys
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

CATEGORIES = {
    "computers": "/search/sys",
    "computer_parts": "/search/syp",
}

def setup_driver(headless: bool = True):
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(ChromeDriverManager().install(), options=opts)

def get_result_links(region_base: str, cat_path: str, limit: int, pause: float = 1.0):
    links = []
    s = 0
    while len(links) < limit:
        url = f"{region_base}{cat_path}?s={s}"
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            break
        soup = BeautifulSoup(r.text, "html.parser")
        posts = soup.select("ul.rows li.result-row a.result-title")
        if not posts:
            break
        for a in posts:
            href = a.get("href")
            if href and href.startswith("http"):
                links.append(href)
                if len(links) >= limit:
                    break
        s += 120  # Craigslist paginates by 120
        time.sleep(pause)
    return links

def parse_listing(driver, url: str, pause: float = 1.0):
    try:
        driver.get(url)
        time.sleep(pause)
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        title_el = soup.select_one("#titletextonly")
        title = title_el.get_text(strip=True) if title_el else ""
        # Description can be in #postingbody
        desc_el = soup.select_one("#postingbody")
        if desc_el:
            # Remove common Craigslist boilerplate
            for tag in desc_el.select("div, p, span"):
                pass
            description = desc_el.get_text("\n", strip=True)
        else:
            description = ""
        return title, description
    except Exception:
        return "", ""

def scrape_category(region_base: str, label: str, cat_path: str, limit: int, pause: float):
    print(f"[scrape] {label} …")
    links = get_result_links(region_base, cat_path, limit, pause)
    print(f"[scrape] found {len(links)} links for {label}")
    driver = setup_driver(headless=True)
    rows = []
    try:
        for i, url in enumerate(links, 1):
            title, desc = parse_listing(driver, url, pause)
            if title or desc:
                rows.append({"title": title, "description": desc})
            if i % 25 == 0:
                print(f"  … {i}/{len(links)} parsed")
    finally:
        driver.quit()
    df = pd.DataFrame(rows)
    df["title"] = df["title"].fillna("").str.strip()
    df["description"] = df["description"].fillna("").str.strip()
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True, help="e.g., https://rochester.craigslist.org")
    ap.add_argument("--limit", type=int, default=400, help="max posts per category")
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between requests")
    args = ap.parse_args()

    base = args.region.rstrip("/")
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    dfs = {}
    for label, path in CATEGORIES.items():
        df = scrape_category(base, label, path, args.limit, args.pause)
        out = data_dir / f"{label}.csv"
        df.to_csv(out, index=False)
        print(f"[save] {out} ({len(df)} rows)")
        dfs[label] = df

    # Combine with weak labels from source
    comb_rows = []
    for label, df in dfs.items():
        for _, r in df.iterrows():
            title = r["title"]
            desc = r["description"]
            text = f"{title} {desc}".strip()
            comb_rows.append({"title": title, "description": desc, "label": label, "text": text})

    combined = pd.DataFrame(comb_rows)
    combined.to_csv(data_dir / "combined_data.csv", index=False)
    print(f"[save] data/combined_data.csv ({len(combined)} rows)")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Example: python scrape_data.py --region https://rochester.craigslist.org --limit 350")
    main()
