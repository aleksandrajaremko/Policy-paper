#!/usr/bin/env python3
"""
===============================================================================
BDL EU Funds Downloader — Validation Check
===============================================================================

Downloads gmina-level EU funding data from BDL for cross-checking against
your contract-level datasets. Data is half-yearly (30 June / 31 December).

Covers:
  - NSRF 2007-2013: number of contracts, total value, by program
  - 2014-2020+: number of contracts, total value, EU funds, national funds

NOTE: These are CUMULATIVE figures (total contracted BY that date),
not flows. To get the amount contracted IN a period, difference
consecutive observations.

USAGE:
  python bdl_eufunds_downloader.py --api-key YOUR_KEY
===============================================================================
"""

import requests
import pandas as pd
import time
import os
import argparse
import logging
from datetime import datetime

BASE_URL = "https://bdl.stat.gov.pl/api/v1"
GMINA_LEVEL = 6
PAGE_SIZE = 100
DEFAULT_START_YEAR = 2007
DEFAULT_END_YEAR = 2024
REQUEST_DELAY = 0.25
RETRY_DELAY = 30
MAX_RETRIES = 5

# ---------------------------------------------------------------------------
# Variables — verified against catalogue
# Half-yearly data: "30 June" and "31 December" are separate variables
# ---------------------------------------------------------------------------

EU_VARIABLES = {
    # ===== NSRF 2007-2013 =====

    # Number of signed contracts
    "nsrf_n_contracts_jun": {
        "var_id": 410161,
        "description": "NSRF 2007-2013: number of signed contracts — total (30 June)",
    },
    "nsrf_n_contracts_dec": {
        "var_id": 410185,
        "description": "NSRF 2007-2013: number of signed contracts — total (31 December)",
    },

    # Total value of signed agreements — total across all programs
    # Multiple var_ids exist; trying the first ones from catalogue
    "nsrf_value_signed_1": {
        "var_id": 290318,
        "description": "NSRF 2007-2013: total value subsidy agreements signed — total (variant 1)",
    },
    "nsrf_value_signed_2": {
        "var_id": 290004,
        "description": "NSRF 2007-2013: total value subsidy agreements signed — total (variant 2)",
    },
    "nsrf_value_signed_3": {
        "var_id": 289743,
        "description": "NSRF 2007-2013: total value subsidy agreements signed — total (variant 3)",
    },

    # Total value of completed projects
    "nsrf_value_completed_1": {
        "var_id": 290092,
        "description": "NSRF 2007-2013: total value completed projects — total (variant 1)",
    },
    "nsrf_value_completed_2": {
        "var_id": 290035,
        "description": "NSRF 2007-2013: total value completed projects — total (variant 2)",
    },

    # Number of grant applications
    "nsrf_n_applications_1": {
        "var_id": 289670,
        "description": "NSRF 2007-2013: number of grant applications — total (variant 1)",
    },

    # By operational program — signed contracts
    "nsrf_n_contracts_infra_jun": {
        "var_id": 410162,
        "description": "NSRF: contracts Infrastructure & Environment (30 June)",
    },
    "nsrf_n_contracts_innov_jun": {
        "var_id": 410163,
        "description": "NSRF: contracts Innovative Economy (30 June)",
    },
    "nsrf_n_contracts_humcap_jun": {
        "var_id": 410164,
        "description": "NSRF: contracts Human Capital (30 June)",
    },
    "nsrf_n_contracts_eastpol_jun": {
        "var_id": 410166,
        "description": "NSRF: contracts Development of Eastern Poland (30 June)",
    },

    # ===== 2014-2020 =====

    # Number of grant contracts
    "eu1420_n_contracts_jun": {
        "var_id": 520213,
        "description": "2014-2020: number of grant contracts — total (30 June)",
    },
    "eu1420_n_contracts_dec": {
        "var_id": 520239,
        "description": "2014-2020: number of grant contracts — total (31 December)",
    },

    # Value of grant contracts — total value
    "eu1420_value_total_1": {
        "var_id": 521428,
        "description": "2014-2020: value of grant contracts — total/total (variant 1)",
    },
    "eu1420_value_total_2": {
        "var_id": 521938,
        "description": "2014-2020: value of grant contracts — total/total (variant 2)",
    },
    "eu1420_value_total_3": {
        "var_id": 522177,
        "description": "2014-2020: value of grant contracts — total/total (variant 3)",
    },
    "eu1420_value_total_4": {
        "var_id": 522413,
        "description": "2014-2020: value of grant contracts — total/total (variant 4)",
    },

    # Value — EU community funds only
    "eu1420_value_eu_1": {
        "var_id": 521862,
        "description": "2014-2020: value of contracts — community funds (variant 1)",
    },
    "eu1420_value_eu_2": {
        "var_id": 521911,
        "description": "2014-2020: value of contracts — community funds (variant 2)",
    },

    # Value — national public funds
    "eu1420_value_national_1": {
        "var_id": 521836,
        "description": "2014-2020: value of contracts — national public funds total (variant 1)",
    },

    # By operational program — 2014-2020 contracts
    "eu1420_n_contracts_infra_jun": {
        "var_id": 520212,
        "description": "2014-2020: contracts Infrastructure & Environment (30 June)",
    },
    "eu1420_n_contracts_smart_jun": {
        "var_id": 520211,
        "description": "2014-2020: contracts Smart Growth (30 June)",
    },
    "eu1420_n_contracts_eastpol_jun": {
        "var_id": 520217,
        "description": "2014-2020: contracts Eastern Poland (30 June)",
    },
    "eu1420_n_contracts_know_jun": {
        "var_id": 520218,
        "description": "2014-2020: contracts Knowledge Education Development (30 June)",
    },

    # ===== 2021-2027 (grant agreements — from newer subgroup) =====
    "eu2127_n_agreements_jun": {
        "var_id": 1746943,
        "description": "2021-2027: number of grant agreements/decisions (30 June)",
    },
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("BDL-eufunds")

# ---------------------------------------------------------------------------
# API Client
# ---------------------------------------------------------------------------

class BDLClient:
    def __init__(self, api_key=None, lang="en"):
        self.session = requests.Session()
        self.lang = lang
        if api_key:
            self.session.headers["X-ClientId"] = api_key
            log.info("Using registered API key")
        else:
            log.info("Anonymous mode (slow)")
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


def main():
    parser = argparse.ArgumentParser(
        description="Download gmina-level EU fund data from BDL",
    )
    parser.add_argument("--api-key", default=None, help="BDL API key")
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--output-dir", default="./bdl_output")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "variables"), exist_ok=True)

    fh = logging.FileHandler(
        os.path.join(args.output_dir, "bdl_eufunds_download.log"),
        encoding="utf-8",
    )
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(fh)

    client = BDLClient(api_key=args.api_key, lang="en")

    years = list(range(args.start_year, args.end_year + 1))

    log.info("BDL EU Funds Downloader (validation)")
    log.info(f"Output: {args.output_dir}")
    log.info(f"Years: {args.start_year}–{args.end_year}")
    log.info(f"Variables: {len(EU_VARIABLES)}")

    panels = {}
    empty_vars = []

    for key, spec in EU_VARIABLES.items():
        var_id = spec["var_id"]
        desc = spec["description"]

        log.info(f"\n--- {key}: {desc} (var_id={var_id}) ---")

        rows = download_variable(client, var_id, desc, years)

        if rows:
            df = pd.DataFrame(rows)
            df.rename(columns={"value": key}, inplace=True)

            var_path = os.path.join(args.output_dir, "variables", f"{key}.csv")
            df.to_csv(var_path, index=False, encoding="utf-8-sig")

            panels[key] = df
        else:
            log.warning(f"  No data for {key} — may be wrong variant ID")
            empty_vars.append(key)

    # Merge into panel
    if panels:
        log.info("\n--- Merging into panel ---")
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

        out_path = os.path.join(args.output_dir, "bdl_gmina_eufunds.csv")
        merged.to_csv(out_path, index=False, encoding="utf-8-sig")
        log.info(f"\n  Saved: {out_path}")
        log.info(f"  Shape: {merged.shape[0]} rows × {merged.shape[1]} cols")
        log.info(f"  Gminas: {merged['unit_id'].nunique()}")

    if empty_vars:
        log.warning(f"\n  Variables with no data ({len(empty_vars)}):")
        for v in empty_vars:
            log.warning(f"    {v}")
        log.warning("  This is expected — the 'variant' IDs for value totals")
        log.warning("  may correspond to different half-year snapshots.")
        log.warning("  Check individual variable CSVs to see which ones returned data.")

    log.info(f"\nDone. Total API requests: {client.request_count}")


if __name__ == "__main__":
    main()
