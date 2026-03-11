"""
PKW Gmina-Level Parliamentary Election Scraper
===============================================
Downloads Sejm election results at gmina level from:
https://danewyborcze.kbw.gov.pl

Elections covered: 2001, 2005, 2007, 2011, 2015, 2019, 2023

Structure varies significantly by year — this scraper handles each year's
idiosyncrasies explicitly, with clear logging so you know exactly what was
downloaded and what may need manual attention.

Output: /data/raw/<year>/ directories containing the original files,
        plus a download_log.csv summarising what was retrieved.

Usage:
    python pkw_scraper.py                        # download all years
    python pkw_scraper.py --years 2019 2023      # specific years only
    python pkw_scraper.py --dry-run              # print URLs without downloading

Requirements:
    pip install requests beautifulsoup4 lxml
"""

import os
import re
import time
import logging
import argparse
import csv
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import requests

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_URL = "https://danewyborcze.kbw.gov.pl/"
OUTPUT_DIR = Path(r"C:\Users\jarem\OneDrive - London School of Economics\YEAR 2\1. Policy paper\policy-paper-repo\data\clean\outcome\Elections")
LOG_FILE = Path("\\download_log.csv")

# Polite delay between requests (seconds) — respect the server
REQUEST_DELAY = 1.5

# HTTP session settings
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (academic research) "
        "PKW-electoral-data-scraper/1.0 "
        "Contact: your-email@institution.ac.uk"
    )
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Per-year file specifications ───────────────────────────────────────────────
#
# Each entry specifies the files to download for gmina-level Sejm results.
# Format:
#   "label":  human-readable description
#   "url":    direct URL (relative to BASE_URL) or None if needs page-scraping
#   "page":   index page URL to scrape links from (used for 2007 multi-file years)
#   "pattern": regex to match href links when scraping a page
#
# Priority file for each year = "wyniki głosowania na listy kandydatów po gminach"
# (vote results on candidate lists by gmina) — this gives party list vote totals,
# which is what you need for calculating party vote shares.
#
# For 2019/2023 we also grab the percentage version ("proc") for cross-validation.

ELECTION_FILES = {
    2023: [
        {
            "label": "wyniki_listy_gminy_count",
            "description": "Vote counts by party list, by gmina (Sejm 2023)",
            "url": "dane/2023/sejmsenat/wyniki_gl_na_listy_po_gminach_sejm_csv.zip",
            "type": "zip_csv",
        },
        {
            "label": "wyniki_listy_gminy_pct",
            "description": "Vote percentages by party list, by gmina (Sejm 2023)",
            "url": "dane/2023/sejmsenat/wyniki_gl_na_listy_po_gminach_proc_sejm_csv.zip",
            "type": "zip_csv",
        },
        {
            "label": "wykaz_list",
            "description": "Party list registry — maps list numbers to party names (Sejm 2023)",
            "url": "dane/2023/sejmsenat/wykaz_list_sejm_csv.zip",
            "type": "zip_csv",
        },
        {
            "label": "frekwencja_gminy",
            "description": "Turnout by gmina at 17:00 (Sejm 2023)",
            "url": "dane/2023/sejmsenat/frekwencja_g17_00_gminy.xlsx",
            "type": "xlsx",
        },
    ],
    2019: [
        {
            "label": "wyniki_listy_gminy_count",
            "description": "Vote counts by party list, by gmina (Sejm 2019)",
            "url": "dane/2019/sejmsenat/wyniki_gl_na_listy_po_gminach_sejm_csv.zip",
            "type": "zip_csv",
        },
        {
            "label": "wyniki_listy_gminy_pct",
            "description": "Vote percentages by party list, by gmina (Sejm 2019)",
            "url": "dane/2019/sejmsenat/wyniki_gl_na_listy_po_gminach_proc_sejm_csv.zip",
            "type": "zip_csv",
        },
        {
            "label": "wykaz_list",
            "description": "Party list registry (Sejm 2019)",
            "url": "dane/2019/sejmsenat/wykaz_list_sejm_csv.zip",
            "type": "zip_csv",
        },
        {
            "label": "frekwencja_gminy",
            "description": "Turnout by gmina at 17:00 (Sejm 2019)",
            "url": "dane/2019/sejmsenat/frekwencja_g17_00_gminy.xlsx",
            "type": "xlsx",
        },
    ],
    # 2015: structured like 2019 but needs page scraping to confirm exact filenames
    2015: [
        {
            "label": "wyniki_listy_gminy_count",
            "description": "Vote counts by party list, by gmina (Sejm 2015)",
            "url": "dane/2015/sejmsenat/wyniki_gl_na_listy_po_gminach_sejm_csv.zip",
            "type": "zip_csv",
            "fallback_page": "indexb73b.html?title=Parlament_2015",
            "fallback_pattern": r"gminach.*sejm.*csv",
        },
        {
            "label": "frekwencja_gminy",
            "description": "Turnout by gmina (Sejm 2015)",
            "url": "dane/2015/sejmsenat/frekwencja_g17_00_gminy.xlsx",
            "type": "xlsx",
            "fallback_page": "indexb73b.html?title=Parlament_2015",
            "fallback_pattern": r"frekwencja.*gmin",
        },
    ],
    # 2011: single XLS files (no zip), directly linked from the page
    2011: [
        {
            "label": "wyniki_listy_gminy_count",
            "description": "Vote counts by party list, by gmina (Sejm 2011)",
            "url": "dane/2011/sejmsenat/2011-gl-lis-gm.xls",
            "type": "xls",
        },
        {
            "label": "wyniki_listy_gminy_pct",
            "description": "Vote percentages by party list, by gmina (Sejm 2011)",
            "url": "dane/2011/sejmsenat/2011-gl-lis-gm-proc.xls",
            "type": "xls",
        },
        {
            "label": "komitety",
            "description": "Electoral committees (party list mapping) (Sejm 2011)",
            "url": "dane/2011/sejmsenat/komitety.zip",
            "type": "zip_csv",
        },
    ],
    # 2007: gmina results split by constituency/list — need to scrape the sub-page
    # The page lists many files matching pattern like "07-gm-XX-YY.xls"
    2007: [
        {
            "label": "wyniki_listy_gminy_SCRAPED",
            "description": "Vote results by gmina, per constituency/list (Sejm 2007) — scraped",
            "url": None,  # scraped dynamically
            "type": "xls_multi",
            "scrape_page": "indexa8cd.html?title=Wybory_do_Sejmu_w_2007_r.",
            "scrape_pattern": r"gmin",   # match hrefs containing 'gmin'
        },
        {
            "label": "wyniki_listy_gminy_summary",
            "description": "Summary vote results on lists by gmina (Sejm 2007)",
            "url": "dane/2007/sejmsenat/2007-gl-lis-gm.xls",
            "type": "xls",
            "fallback_page": "indexa8cd.html?title=Wybory_do_Sejmu_w_2007_r.",
            "fallback_pattern": r"gl-lis-gm",
        },
    ],
    # 2005: results on sub-page, older XLS format
    2005: [
        {
            "label": "wyniki_listy_gminy",
            "description": "Vote results by gmina (Sejm 2005) — scraped from sub-page",
            "url": None,
            "type": "xls_multi",
            "scrape_page": "index0cff.html?title=Wybory_do_Sejmu_w_2005_r.",
            "scrape_pattern": r"gmin",
        },
        {
            "label": "frekwencja_gminy",
            "description": "Turnout/electorate by gmina (Sejm/Senate 2005)",
            "url": "dane/2005/2005-parl-frekw-gm.xls",
            "type": "xls",
        },
    ],
    # 2001: gmina-level results confirmed available, older structure
    2001: [
        {
            "label": "wyniki_listy_gminy",
            "description": "Vote results on lists by gmina (Sejm 2001) — scraped",
            "url": None,
            "type": "xls_multi",
            "scrape_page": "index2a07.html?title=Wybory_do_Sejmu_w_2001_r.",
            "scrape_pattern": r"gmin",
        },
    ],
}


# ── Core download functions ────────────────────────────────────────────────────

def make_session():
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def download_file(session, url, dest_path, dry_run=False):
    """Download a single file. Returns (success, bytes_downloaded, error_msg)."""
    if dry_run:
        log.info(f"  [DRY RUN] Would download: {url}")
        return True, 0, None

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        log.info(f"  SKIP (already exists): {dest_path.name}")
        return True, dest_path.stat().st_size, None

    try:
        log.info(f"  GET {url}")
        resp = session.get(url, timeout=30, stream=True)
        resp.raise_for_status()

        total = 0
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
                total += len(chunk)

        log.info(f"  OK  {dest_path.name} ({total/1024:.1f} KB)")
        time.sleep(REQUEST_DELAY)
        return True, total, None

    except requests.RequestException as e:
        log.warning(f"  FAIL {url}: {e}")
        return False, 0, str(e)


def scrape_links(session, page_url, pattern, dry_run=False):
    """
    Fetch a Dane Wyborcze page and return all href links matching `pattern`.
    Returns list of absolute URLs.
    """
    log.info(f"  Scraping page: {page_url}")
    if dry_run:
        log.info(f"  [DRY RUN] Would scrape: {page_url}")
        return []

    try:
        resp = session.get(page_url, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        links = []
        regex = re.compile(pattern, re.IGNORECASE)
        for a in soup.find_all("a", href=True):
            href = a["href"]
            # Match on the href itself OR the link text
            if regex.search(href) or regex.search(a.get_text()):
                abs_url = urljoin(page_url, href)
                # Filter out navigation/wiki links — we only want data files
                if any(href.endswith(ext) for ext in
                       [".zip", ".xls", ".xlsx", ".csv"]):
                    links.append(abs_url)

        log.info(f"  Found {len(links)} matching links")
        time.sleep(REQUEST_DELAY)
        return links

    except requests.RequestException as e:
        log.warning(f"  Could not scrape {page_url}: {e}")
        return []


# ── Main download orchestrator ─────────────────────────────────────────────────

def download_year(session, year, file_specs, dry_run=False):
    """Download all files for a given election year. Returns list of log rows."""
    log.info(f"\n{'='*60}")
    log.info(f"  YEAR {year}")
    log.info(f"{'='*60}")

    year_dir = Path("data/raw") / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for spec in file_specs:
        label = spec["label"]
        description = spec["description"]
        file_type = spec["type"]

        # ── Case 1: Direct URL ─────────────────────────────────────────────
        if spec.get("url"):
            abs_url = urljoin(BASE_URL, spec["url"])
            filename = abs_url.split("/")[-1]
            dest = year_dir / filename

            success, size, err = download_file(session, abs_url, dest, dry_run)

            # If direct URL fails and there's a fallback page, try scraping
            if not success and spec.get("fallback_page"):
                log.info(f"  Trying fallback page for {label}...")
                page_url = urljoin(BASE_URL, spec["fallback_page"])
                links = scrape_links(session, page_url,
                                     spec.get("fallback_pattern", label),
                                     dry_run)
                for link in links[:1]:  # take first match
                    filename = link.split("/")[-1]
                    dest = year_dir / filename
                    success, size, err = download_file(session, link, dest, dry_run)

            results.append({
                "year": year,
                "label": label,
                "description": description,
                "url": abs_url,
                "local_file": str(dest) if not dry_run else "DRY_RUN",
                "success": success,
                "size_bytes": size,
                "error": err or "",
                "timestamp": datetime.now().isoformat(),
            })

        # ── Case 2: Scrape page for multiple files ─────────────────────────
        elif spec.get("scrape_page"):
            page_url = urljoin(BASE_URL, spec["scrape_page"])
            pattern = spec.get("scrape_pattern", "gmin")
            links = scrape_links(session, page_url, pattern, dry_run)

            if not links:
                log.warning(f"  No links found for {label} — manual check needed")
                results.append({
                    "year": year,
                    "label": label,
                    "description": description,
                    "url": page_url,
                    "local_file": "",
                    "success": False,
                    "size_bytes": 0,
                    "error": "No matching links found on page",
                    "timestamp": datetime.now().isoformat(),
                })
                continue

            for link in links:
                filename = link.split("/")[-1]
                dest = year_dir / filename
                success, size, err = download_file(session, link, dest, dry_run)
                results.append({
                    "year": year,
                    "label": label,
                    "description": f"{description} [{filename}]",
                    "url": link,
                    "local_file": str(dest) if not dry_run else "DRY_RUN",
                    "success": success,
                    "size_bytes": size,
                    "error": err or "",
                    "timestamp": datetime.now().isoformat(),
                })

    return results


def save_log(all_results):
    """Write download log to CSV."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "year", "label", "description", "url",
        "local_file", "success", "size_bytes", "error", "timestamp"
    ]
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    log.info(f"\nDownload log saved to: {LOG_FILE}")


def print_summary(all_results):
    """Print a summary of what was downloaded."""
    log.info("\n" + "="*60)
    log.info("DOWNLOAD SUMMARY")
    log.info("="*60)
    by_year = {}
    for r in all_results:
        y = r["year"]
        by_year.setdefault(y, {"ok": 0, "fail": 0, "bytes": 0})
        if r["success"]:
            by_year[y]["ok"] += 1
            by_year[y]["bytes"] += r["size_bytes"]
        else:
            by_year[y]["fail"] += 1

    for year in sorted(by_year):
        s = by_year[year]
        log.info(
            f"  {year}: {s['ok']} files OK, {s['fail']} failed  "
            f"({s['bytes']/1024:.0f} KB total)"
        )

    total_fail = sum(r["fail"] for r in by_year.values())
    if total_fail > 0:
        log.warning(f"\n  {total_fail} files failed — check download_log.csv for details")
        log.warning("  Some older years (2001, 2005, 2007) may need manual download")
        log.warning("  See MANUAL_DOWNLOAD_NOTES below")


MANUAL_DOWNLOAD_NOTES = """
MANUAL DOWNLOAD NOTES
=====================
If the scraper fails for 2001, 2005, or 2007, these years have the most
complex structure on Dane Wyborcze and may require manual navigation.

2001:
  Go to: https://danewyborcze.kbw.gov.pl/index2a07.html?title=Wybory_do_Sejmu_w_2001_r.
  Download: "Wyniki głosowania na listy kandydatów w układzie gmin"
  This is the key gmina-level file.

2005:
  Go to: https://danewyborcze.kbw.gov.pl/index0cff.html?title=Wybory_do_Sejmu_w_2005_r.
  Download: "Wyniki głosowania w wyborach do Sejmu w arkuszach xls"
  The turnout file is at: https://danewyborcze.kbw.gov.pl/dane/2005/2005-parl-frekw-gm.xls

2007:
  Go to: https://danewyborcze.kbw.gov.pl/indexa8cd.html?title=Wybory_do_Sejmu_w_2007_r.
  Results are split by constituency — download all files matching "gmin" in the name.
  Note: You'll need to aggregate these across constituencies to get a national gmina file.

2015:
  If the direct URL fails, go to:
  https://danewyborcze.kbw.gov.pl/indexb73b.html?title=Parlament_2015
  The file structure should match 2019.

IMPORTANT — PARTY LIST CROSSWALK:
After downloading, each year's data identifies parties by list NUMBER, not name.
You must join the "wykaz_list" or "komitety" file for each year to get party names.
For 2011 and earlier, the column headers in the XLS files may directly contain
party names (abbreviated). Inspect each file before building your panel.
"""


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download PKW gmina-level Sejm election data"
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=sorted(ELECTION_FILES.keys()),
        help="Election years to download (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print URLs without downloading",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    log_file = out_dir / "download_log.csv"

    log.info("PKW Parliamentary Election Scraper")
    log.info(f"Years: {args.years}")
    log.info(f"Output: {OUTPUT_DIR}")
    if args.dry_run:
        log.info("Mode: DRY RUN — no files will be downloaded")

    session = make_session()
    all_results = []

    for year in sorted(args.years):
        if year not in ELECTION_FILES:
            log.warning(f"Year {year} not configured — skipping")
            continue
        results = download_year(
            session, year, ELECTION_FILES[year], dry_run=args.dry_run
        )
        all_results.extend(results)

    if not args.dry_run:
        save_log(all_results)
        print_summary(all_results)

    print(MANUAL_DOWNLOAD_NOTES)


if __name__ == "__main__":
    main()