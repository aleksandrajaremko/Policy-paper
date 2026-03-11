#!/usr/bin/env python3
"""
===============================================================================
BDL Income & Wage Downloader — supplement to bdl_gmina_downloader_v2.py
===============================================================================

Downloads gmina-level income proxies and wage data NOT included in v2:
  - PIT per capita (best proxy for disposable income at gmina level)
  - Revenue per capita (total)
  - Own revenue per capita
  - Median monthly gross wage by place of residence (Jan–Aug)
  - Average monthly gross wage by place of residence (Jan–Aug)

USAGE:
  python bdl_income_downloader.py --api-key YOUR_KEY
  python bdl_income_downloader.py --api-key YOUR_KEY --only wages
  python bdl_income_downloader.py --api-key YOUR_KEY --only fiscal

NOTES:
  - Wage data is monthly (Jan–Aug only) and likely recent (2020s onward).
    The script downloads all months; you can average them in post-processing.
  - PIT per capita has long time coverage (2000+) and is the standard
    income proxy in Polish sub-regional research.
  - Disposable income is NOT published at gmina level by GUS — only at
    voivodeship level via household budget surveys.
===============================================================================
"""

import requests
import pandas as pd
import time
import os
import argparse
import logging
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration (same as v2)
# ---------------------------------------------------------------------------

BASE_URL = "https://bdl.stat.gov.pl/api/v1"
GMINA_LEVEL = 6
PAGE_SIZE = 100
DEFAULT_START_YEAR = 2000
DEFAULT_END_YEAR = 2024
REQUEST_DELAY = 0.25
RETRY_DELAY = 30
MAX_RETRIES = 5

# ---------------------------------------------------------------------------
# Variables to download
# ---------------------------------------------------------------------------

# Fiscal income proxies (long time series, all gminas)
FISCAL_VARIABLES = {
    "pit_per_capita": {
        "var_id": 149128,
        "description": "PIT per capita — own revenue share in personal income tax (PLN)",
    },
    "revenue_per_capita": {
        "var_id": 76973,
        "description": "Total budget revenue per capita (PLN)",
    },
    "own_revenue_per_capita": {
        "var_id": 76976,
        "description": "Own revenue per capita (PLN)",
    },
}

# Wage variables — monthly, by place of residence
# These are from the Survey on Distribution of Wages and Salaries,
# likely available only for recent years.
WAGE_VARIABLES = {
    "median_wage_jan": {"var_id": 1750141, "description": "Median gross wage — January (PLN)"},
    "median_wage_feb": {"var_id": 1750147, "description": "Median gross wage — February (PLN)"},
    "median_wage_mar": {"var_id": 1750153, "description": "Median gross wage — March (PLN)"},
    "median_wage_apr": {"var_id": 1750159, "description": "Median gross wage — April (PLN)"},
    "median_wage_may": {"var_id": 1750165, "description": "Median gross wage — May (PLN)"},
    "median_wage_jun": {"var_id": 1750171, "description": "Median gross wage — June (PLN)"},
    "median_wage_jul": {"var_id": 1750177, "description": "Median gross wage — July (PLN)"},
    "median_wage_aug": {"var_id": 1750183, "description": "Median gross wage — August (PLN)"},
    "avg_wage_jan": {"var_id": 1749925, "description": "Average gross wage — January (PLN)"},
    "avg_wage_feb": {"var_id": 1749931, "description": "Average gross wage — February (PLN)"},
    "avg_wage_mar": {"var_id": 1749937, "description": "Average gross wage — March (PLN)"},
    "avg_wage_apr": {"var_id": 1749943, "description": "Average gross wage — April (PLN)"},
    "avg_wage_may": {"var_id": 1749949, "description": "Average gross wage — May (PLN)"},
    "avg_wage_jun": {"var_id": 1749955, "description": "Average gross wage — June (PLN)"},
    "avg_wage_jul": {"var_id": 1749961, "description": "Average gross wage — July (PLN)"},
    "avg_wage_aug": {"var_id": 1749967, "description": "Average gross wage — August (PLN)"},
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("BDL-income")

# ---------------------------------------------------------------------------
# API Client (identical to v2)
# ---------------------------------------------------------------------------

class BDLClient:
    def __init__(self, api_key=None, lang="en"):
        self.session = requests.Session()
        self.lang = lang
        if api_key:
            self.session.headers["X-ClientId"] = api_key
            log.info("Using registered API key")
        else:
            log.info("Anonymous mode (slow — register at api.stat.gov.pl)")
        self.request_count = 0

    def _get_with_years(self, endpoint, params, years=None):
        url = f"{BASE_URL}/{endpoint}"
        params.setdefault("format", "json")
        params.setdefault("lang", self.lang)

        query_parts = [f"{k}={v}" for k, v in params.items()]
        if years:
            query_parts.extend(f"year={y}" for y in years)

        full_url = url + "?" + "&".join(query_parts)

        for attempt in range(MAX_RETRIES):
            try:
                time.sleep(REQUEST_DELAY)
                resp = self.session.get(full_url, timeout=30)
                self.request_count += 1

                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 429:
                    wait = RETRY_DELAY * (attempt + 1)
                    log.warning(f"Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                elif resp.status_code == 404:
                    return None
                else:
                    log.warning(f"HTTP {resp.status_code} (attempt {attempt+1})")
                    time.sleep(RETRY_DELAY)
            except requests.exceptions.RequestException as e:
                log.warning(f"Request error: {e}")
                time.sleep(RETRY_DELAY)

        return None

    def get_data_by_variable(self, var_id, years=None):
        params = {"unit-level": GMINA_LEVEL, "page-size": PAGE_SIZE, "page": 0}
        endpoint = f"data/by-variable/{var_id}"

        all_results = []
        while True:
            data = self._get_with_years(endpoint, params.copy(), years)
            if data is None:
                break

            results = data.get("results", [])
            if not results:
                break

            all_results.extend(results)

            total_records = data.get("totalRecords", 0)
            if len(all_results) >= total_records:
                break

            params["page"] += 1

        return all_results


# ---------------------------------------------------------------------------
# Download logic
# ---------------------------------------------------------------------------

def download_variable(client, var_id, description, years):
    """Download one variable across all gminas. Returns list of row dicts."""
    log.info(f"  Downloading var {var_id}: {description}")

    year_chunks = [years[i:i+10] for i in range(0, len(years), 10)]
    all_rows = []

    for chunk in year_chunks:
        results = client.get_data_by_variable(var_id=str(var_id), years=chunk)
        if not results:
            continue

        for unit_data in results:
            unit_id = unit_data.get("id", "")
            unit_name = unit_data.get("name", "")

            for val in unit_data.get("values", []):
                all_rows.append({
                    "unit_id": unit_id,
                    "unit_name": unit_name,
                    "year": val.get("year"),
                    "value": val.get("val"),
                })

    log.info(f"    → {len(all_rows)} observations")
    return all_rows


def download_group(client, variables, output_dir, start_year, end_year, label):
    """Download a group of variables and merge into a panel CSV."""
    years = list(range(start_year, end_year + 1))
    panels = {}

    log.info(f"\n{'='*70}")
    log.info(f"  DOWNLOADING: {label} ({len(variables)} variables)")
    log.info(f"  Years: {start_year}–{end_year}")
    log.info(f"{'='*70}")

    for key, spec in variables.items():
        var_id = spec["var_id"]
        desc = spec["description"]

        log.info(f"\n--- {key}: {desc} (var_id={var_id}) ---")

        rows = download_variable(client, var_id, desc, years)

        if rows:
            df = pd.DataFrame(rows)
            df.rename(columns={"value": key}, inplace=True)

            # Save individual CSV
            var_path = os.path.join(output_dir, "variables", f"{key}.csv")
            os.makedirs(os.path.dirname(var_path), exist_ok=True)
            df.to_csv(var_path, index=False, encoding="utf-8-sig")

            panels[key] = df
        else:
            log.warning(f"  No data returned for {key}")

    # Merge into one panel
    if panels:
        merged = None
        for key, df in panels.items():
            cols = ["unit_id", "unit_name", "year", key]
            df_slim = df[cols].drop_duplicates(subset=["unit_id", "year"])

            if merged is None:
                merged = df_slim
            else:
                merged = merged.merge(
                    df_slim[["unit_id", "year", key]],
                    on=["unit_id", "year"],
                    how="outer",
                )

        merged.sort_values(["unit_id", "year"], inplace=True)
        merged["powiat_id"] = merged["unit_id"].str[:6]
        merged["voivodship_id"] = merged["unit_id"].str[:2]

        safe_label = label.lower().replace(" ", "_")
        out_path = os.path.join(output_dir, f"bdl_gmina_{safe_label}.csv")
        merged.to_csv(out_path, index=False, encoding="utf-8-sig")
        log.info(f"\n  Saved: {out_path}")
        log.info(f"  Shape: {merged.shape[0]} rows × {merged.shape[1]} cols")
        log.info(f"  Gminas: {merged['unit_id'].nunique()}")

    return panels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download gmina-level income/wage data from BDL",
    )
    parser.add_argument("--api-key", default=None, help="BDL API key")
    parser.add_argument(
        "--start-year", type=int, default=DEFAULT_START_YEAR,
    )
    parser.add_argument(
        "--end-year", type=int, default=DEFAULT_END_YEAR,
    )
    parser.add_argument(
        "--output-dir", default="./bdl_output",
    )
    parser.add_argument(
        "--only", choices=["fiscal", "wages", "all"], default="all",
        help="Download only fiscal proxies, only wages, or all (default: all)",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "variables"), exist_ok=True)

    # File logging
    fh = logging.FileHandler(
        os.path.join(args.output_dir, "bdl_income_download.log"),
        encoding="utf-8",
    )
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(fh)

    client = BDLClient(api_key=args.api_key, lang="en")

    log.info("BDL Income & Wage Downloader")
    log.info(f"Output: {args.output_dir}")
    log.info(f"Mode: {args.only}")

    if args.only in ("fiscal", "all"):
        download_group(
            client, FISCAL_VARIABLES, args.output_dir,
            args.start_year, args.end_year, "fiscal_income_proxies",
        )

    if args.only in ("wages", "all"):
        download_group(
            client, WAGE_VARIABLES, args.output_dir,
            args.start_year, args.end_year, "wages",
        )

    log.info(f"\nDone. Total API requests: {client.request_count}")


if __name__ == "__main__":
    main()
