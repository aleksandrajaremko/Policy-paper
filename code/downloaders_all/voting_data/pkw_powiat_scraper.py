"""
PKW Powiat-Level Parliamentary Election Scraper
================================================
Downloads Sejm election results at powiat level from danewyborcze.kbw.gov.pl

Elections: 2001, 2005, 2007, 2011, 2015, 2019, 2023

Usage:
    python pkw_powiat_scraper.py --output-dir data/raw_powiat
    python pkw_powiat_scraper.py --output-dir data/raw_powiat --dry-run

Requirements:
    pip install requests beautifulsoup4
"""

import os
import re
import time
import csv
import logging
import argparse
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL    = "https://danewyborcze.kbw.gov.pl/"
REQUEST_DELAY = 1.5
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (academic research) "
        "PKW-powiat-scraper/1.0"
    )
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Direct download URLs — all confirmed from live pages ─────────────────────
# Counts files only (not percentage variants)

ELECTION_FILES = {
    2023: [
        {
            "label":       "wyniki_listy_powiaty_count",
            "description": "Vote counts by party list, by powiat (Sejm 2023)",
            "url":         "dane/2023/sejmsenat/wyniki_gl_na_listy_po_powiatach_sejm_csv.zip",
            "type":        "zip_csv",
        },
        {
            "label":       "wykaz_list",
            "description": "Party list registry — maps list numbers to party names (2023)",
            "url":         "dane/2023/sejmsenat/wykaz_list_sejm_csv.zip",
            "type":        "zip_csv",
        },
    ],
    2019: [
        {
            "label":       "wyniki_listy_powiaty_count",
            "description": "Vote counts by party list, by powiat (Sejm 2019)",
            "url":         "dane/2019/sejmsenat/wyniki_gl_na_listy_po_powiatach_sejm_csv.zip",
            "type":        "zip_csv",
        },
        {
            "label":       "wykaz_list",
            "description": "Party list registry (2019)",
            "url":         "dane/2019/sejmsenat/wykaz_list_sejm_csv.zip",
            "type":        "zip_csv",
        },
    ],
    2015: [
        {
            "label":       "wyniki_listy_powiaty_count",
            "description": "Vote counts by party list, by powiat (Sejm 2015)",
            "url":         "dane/2015/sejmsenat/wyniki_gl_na_listy_po_powiatach_sejm_csv.zip",
            "type":        "zip_csv",
            # If the above URL fails (2015 structure may differ), fall back to scraping
            "fallback_page":    "indexb73b.html?title=Parlament_2015",
            "fallback_pattern": r"powiat.*sejm.*csv|listy.*powiat",
        },
    ],
    2011: [
        {
            "label":       "wyniki_listy_powiaty_count",
            "description": "Vote counts by party list, by powiat (Sejm 2011)",
            "url":         "dane/2011/sejmsenat/2011-sejm-pow-listy.xls",
            "type":        "xls",
        },
    ],
    2007: [
        {
            "label":       "wyniki_listy_powiaty_count",
            "description": "Vote counts by party list, by powiat (Sejm 2007)",
            "url":         "dane/2007/sejm/sejm2007-pow-listy.xls",
            "type":        "xls",
        },
    ],
    2005: [
        {
            "label":       "wyniki_listy_powiaty_count",
            "description": "Vote counts by party list, by powiat (Sejm 2005) — _36797 is counts",
            "url":         "dane/2005/sejm/1456225675_36797.xls",
            "type":        "xls",
        },
    ],
    2001: [
        {
            "label":       "wyniki_listy_powiaty_count",
            "description": "Vote counts by party list, by powiat (Sejm 2001) — scraped",
            "url":         None,  # filename not confirmed — scrape the index page
            "type":        "xls_scraped",
            "scrape_page":    "index2a07.html?title=Wybory_do_Sejmu_w_2001_r.",
            "scrape_pattern": r"powiat",
        },
    ],
}


def make_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def download_file(session, url, dest_path, dry_run=False):
    if dry_run:
        log.info(f"  [DRY RUN] {url}")
        return True, 0, None

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        log.info(f"  SKIP (exists): {dest_path.name}")
        return True, dest_path.stat().st_size, None

    try:
        log.info(f"  GET {url}")
        resp = session.get(url, timeout=30, stream=True)
        resp.raise_for_status()
        total = 0
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(65536):
                f.write(chunk)
                total += len(chunk)
        log.info(f"  OK  {dest_path.name} ({total/1024:.1f} KB)")
        time.sleep(REQUEST_DELAY)
        return True, total, None
    except requests.RequestException as e:
        log.warning(f"  FAIL {url}: {e}")
        return False, 0, str(e)


def scrape_links(session, page_url, pattern, dry_run=False):
    if dry_run:
        log.info(f"  [DRY RUN] Would scrape: {page_url}")
        return []
    try:
        log.info(f"  Scraping: {page_url}")
        resp = session.get(page_url, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        regex = re.compile(pattern, re.IGNORECASE)
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text()
            if regex.search(href) or regex.search(text):
                if any(href.endswith(ext) for ext in [".zip", ".xls", ".xlsx", ".csv"]):
                    links.append(urljoin(page_url, href))
        log.info(f"  Found {len(links)} matching links")
        time.sleep(REQUEST_DELAY)
        return links
    except requests.RequestException as e:
        log.warning(f"  Scrape failed {page_url}: {e}")
        return []


def download_year(session, year, specs, output_dir, dry_run=False):
    log.info(f"\n{'='*55}")
    log.info(f"  YEAR {year}")
    log.info(f"{'='*55}")

    year_dir = output_dir / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for spec in specs:
        label = spec["label"]

        # Direct URL
        if spec.get("url"):
            abs_url  = urljoin(BASE_URL, spec["url"])
            filename = abs_url.split("/")[-1]
            dest     = year_dir / filename
            ok, size, err = download_file(session, abs_url, dest, dry_run)

            # Fallback to page scraping if direct URL fails
            if not ok and spec.get("fallback_page"):
                log.info(f"  Direct URL failed — trying fallback scrape")
                page_url = urljoin(BASE_URL, spec["fallback_page"])
                links    = scrape_links(session, page_url,
                                        spec["fallback_pattern"], dry_run)
                for link in links[:1]:
                    filename = link.split("/")[-1]
                    dest     = year_dir / filename
                    ok, size, err = download_file(session, link, dest, dry_run)

            results.append({
                "year": year, "label": label,
                "url": abs_url, "local_file": str(dest),
                "success": ok, "size_bytes": size, "error": err or "",
                "timestamp": datetime.now().isoformat(),
            })

        # Scrape index page
        elif spec.get("scrape_page"):
            page_url = urljoin(BASE_URL, spec["scrape_page"])
            links    = scrape_links(session, page_url,
                                    spec["scrape_pattern"], dry_run)
            if not links:
                log.warning(f"  {year}: No powiat links found — check page manually")
                log.warning(f"  Page: {page_url}")
                results.append({
                    "year": year, "label": label,
                    "url": page_url, "local_file": "",
                    "success": False, "size_bytes": 0,
                    "error": "No matching links on page",
                    "timestamp": datetime.now().isoformat(),
                })
                continue

            for link in links:
                filename = link.split("/")[-1]
                dest     = year_dir / filename
                ok, size, err = download_file(session, link, dest, dry_run)
                results.append({
                    "year": year, "label": label,
                    "url": link, "local_file": str(dest),
                    "success": ok, "size_bytes": size, "error": err or "",
                    "timestamp": datetime.now().isoformat(),
                })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Download PKW powiat-level Sejm election data"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw_powiat"))
    parser.add_argument("--years", nargs="+", type=int,
                        default=sorted(ELECTION_FILES.keys()))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    log.info("PKW Powiat Election Scraper")
    log.info(f"Years:  {args.years}")
    log.info(f"Output: {args.output_dir}")

    session     = make_session()
    all_results = []

    for year in sorted(args.years):
        if year not in ELECTION_FILES:
            log.warning(f"Year {year} not configured")
            continue
        results = download_year(session, year, ELECTION_FILES[year],
                                args.output_dir, args.dry_run)
        all_results.extend(results)

    if not args.dry_run:
        log_path = args.output_dir / "download_log.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        log.info(f"\nLog saved: {log_path}")

        ok_count   = sum(1 for r in all_results if r["success"])
        fail_count = sum(1 for r in all_results if not r["success"])
        log.info(f"Done: {ok_count} OK, {fail_count} failed")

        if fail_count:
            log.warning("\nFailed downloads — check manually:")
            for r in all_results:
                if not r["success"]:
                    log.warning(f"  {r['year']}: {r['url']}")
            log.warning("\nFor 2001, manually download from:")
            log.warning("  https://danewyborcze.kbw.gov.pl/index2a07.html?title=Wybory_do_Sejmu_w_2001_r.")
            log.warning("  Look for: 'Wyniki głosowania na listy kandydatów w układzie powiatów'")


if __name__ == "__main__":
    main()
    
    
    
