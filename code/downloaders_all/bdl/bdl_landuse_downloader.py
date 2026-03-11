#!/usr/bin/env python3
"""
===============================================================================
BDL Land Use & Urbanisation Downloader
===============================================================================

Downloads gmina-level geodetic land use data from BDL, covering:
  - Built-up and urbanized areas (total, residential, industrial, transport)
  - Total area, agricultural land, forests
  - Population density of built-up areas

These variables come from the national cadaster (ewidencja gruntów)
and allow you to compute urbanisation share = built_up / total_area.

USAGE:
  python bdl_landuse_downloader.py --api-key YOUR_KEY
  python bdl_landuse_downloader.py --api-key YOUR_KEY --start-year 2004
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
# Configuration
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
# Variables — verified against bdl_variable_catalogue_gmina.csv
# All from subgroup: "Geodetic area of the country according to the
# directions of use" (national cadaster data)
# ---------------------------------------------------------------------------

LANDUSE_VARIABLES = {
    # --- TOTAL ---
    "total_area_ha": {
        "var_id": 148578,
        "description": "Total area (ha)",
    },
    "total_area_km2": {
        "var_id": 2018,
        "description": "Total area (km²)",
    },
    # --- BUILT-UP & URBANISED ---
    "builtup_urban_total": {
        "var_id": 148591,
        "description": "Built-up and urbanized areas — total (ha)",
    },
    "builtup_residential": {
        "var_id": 148590,
        "description": "Built-up and urbanized areas — residential (ha)",
    },
    "builtup_industrial": {
        "var_id": 148589,
        "description": "Built-up and urbanized areas — industrial (ha)",
    },
    "builtup_other": {
        "var_id": 148588,
        "description": "Built-up and urbanized areas — other built-up (ha)",
    },
    "urbanized_nonbuilt": {
        "var_id": 148587,
        "description": "Urbanized non-built-up areas (ha)",
    },
    "recreational_areas": {
        "var_id": 148586,
        "description": "Recreational and rest areas (ha)",
    },
    "transport_roads": {
        "var_id": 148585,
        "description": "Transport areas — roads (ha)",
    },
    "transport_railway": {
        "var_id": 148584,
        "description": "Transport areas — railway (ha)",
    },
    "transport_other": {
        "var_id": 148583,
        "description": "Transport areas — other (ha)",
    },
    "minerals": {
        "var_id": 148582,
        "description": "Mineral extraction areas (ha)",
    },
    # --- AGRICULTURAL ---
    "agricultural_total": {
        "var_id": 148577,
        "description": "Agricultural land — total (ha)",
    },
    "arable_land": {
        "var_id": 148576,
        "description": "Agricultural land — arable (ha)",
    },
    # --- FORESTS & WATER ---
    "forest_total": {
        "var_id": 148569,
        "description": "Forest land and woody/bushy land — total (ha)",
    },
    "water_total": {
        "var_id": 148595,
        "description": "Lands under waters — total (ha)",
    },
    # --- OTHER ---
    "wasteland": {
        "var_id": 148580,
        "description": "Wasteland (ha)",
    },
    "land_area": {
        "var_id": 471931,
        "description": "Area of land (ha)",
    },
    # --- DENSITY ---
    "pop_density_builtup": {
        "var_id": 458238,
        "description": "Population density of built-up and urbanized area (persons)",
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
log = logging.getLogger("BDL-landuse")

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
        description="Download gmina-level land use & urbanisation data from BDL",
    )
    parser.add_argument("--api-key", default=None, help="BDL API key")
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--output-dir", default="./bdl_output")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "variables"), exist_ok=True)

    fh = logging.FileHandler(
        os.path.join(args.output_dir, "bdl_landuse_download.log"),
        encoding="utf-8",
    )
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(fh)

    client = BDLClient(api_key=args.api_key, lang="en")

    years = list(range(args.start_year, args.end_year + 1))

    log.info("BDL Land Use & Urbanisation Downloader")
    log.info(f"Output: {args.output_dir}")
    log.info(f"Years: {args.start_year}–{args.end_year}")
    log.info(f"Variables: {len(LANDUSE_VARIABLES)}")

    panels = {}

    for key, spec in LANDUSE_VARIABLES.items():
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
            log.warning(f"  No data for {key}")

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

        # Compute urbanisation share
        if "builtup_urban_total" in merged.columns and "total_area_ha" in merged.columns:
            merged["urban_share"] = (
                pd.to_numeric(merged["builtup_urban_total"], errors="coerce")
                / pd.to_numeric(merged["total_area_ha"], errors="coerce")
            )
            log.info("  Computed urban_share = builtup_urban_total / total_area_ha")

        out_path = os.path.join(args.output_dir, "bdl_gmina_landuse.csv")
        merged.to_csv(out_path, index=False, encoding="utf-8-sig")
        log.info(f"\n  Saved: {out_path}")
        log.info(f"  Shape: {merged.shape[0]} rows × {merged.shape[1]} cols")
        log.info(f"  Gminas: {merged['unit_id'].nunique()}")

    log.info(f"\nDone. Total API requests: {client.request_count}")


if __name__ == "__main__":
    main()
