#!/usr/bin/env python3
"""
===============================================================================
BDL (Bank Danych Lokalnych) Gmina-Level Data Downloader
===============================================================================

Downloads time-series data (2000–present) for all Polish gminas (~2,479 units)
from Poland's Local Data Bank API (bdl.stat.gov.pl).

Data source: GUS (Statistics Poland) — CC BY 4.0 licence
API docs:    https://api.stat.gov.pl/Home/BdlApi?lang=en

USAGE:
------
1. DISCOVERY MODE (recommended first run):
   python bdl_gmina_downloader.py --discover
   → Scans all BDL subjects, finds variables available at gmina level,
     saves a catalogue to 'bdl_variable_catalogue_gmina.csv'

2. DOWNLOAD MODE (uses curated variable list):
   python bdl_gmina_downloader.py --download
   → Downloads data for pre-selected key variables across all gminas
     and years 2000–2024, saves to 'bdl_gmina_panel.csv'

3. DOWNLOAD ALL discovered variables:
   python bdl_gmina_downloader.py --download-all
   → Downloads ALL variables found in the catalogue (can be very large!)

OPTIONS:
  --api-key YOUR_KEY    Use a registered API key for higher rate limits
  --start-year 2000     First year to request (default: 2000)
  --end-year 2024       Last year to request (default: 2024)
  --output-dir ./data   Output directory (default: ./bdl_output)
  --lang en             Language: 'en' or 'pl' (default: en)

RATE LIMITS (anonymous):
  5 req/s | 100 req/15min | 1,000 req/12h | 10,000 req/7d
  → Register at https://api.stat.gov.pl for 10x higher limits

NOTES FOR YOUR RESEARCH:
  - Your project focuses on Poland's 2,479 gminas (NUTS-5 / BDL level 6)
  - Key variables for the "Geography of Discontent" analysis:
    * Demographics (population, migration, age structure)
    * Labour market (unemployment, employment)
    * Local government finances (revenue, expenditure, EU fund absorption)
    * Education (schools, students, graduates)
    * Infrastructure (roads, water, sewage)
    * Business activity (registered entities, new firms)
  - GDP is NOT available at gmina level — use tax revenue or night lights as proxy
  - Electoral data must be sourced separately from PKW (pkw.gov.pl)
===============================================================================
"""

import requests
import pandas as pd
import time
import json
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://bdl.stat.gov.pl/api/v1"
GMINA_LEVEL = 6          # BDL level 6 = gminy (municipalities)
PAGE_SIZE = 100           # max results per page
DEFAULT_START_YEAR = 2000
DEFAULT_END_YEAR = 2024

# Rate limiting: be conservative to avoid hitting limits
REQUEST_DELAY = 0.25      # seconds between requests (anonymous: max 5/s)
RETRY_DELAY = 30          # seconds to wait on rate limit / error
MAX_RETRIES = 5

# ---------------------------------------------------------------------------
# Curated variable IDs known to be available at gmina level
# ---------------------------------------------------------------------------
# These are commonly available variables at level 6. The --discover mode
# will build the full catalogue, but these give you a solid starting panel.
#
# IMPORTANT: Variable IDs can change when GUS restructures BDL.
# Always verify with --discover mode first, then update this dict.
# The IDs below are illustrative — the discovery step will find the
# actual current IDs for your target subjects.
# ---------------------------------------------------------------------------

CURATED_VARIABLES = {
    # --- DEMOGRAPHICS ---
    "population_total": {
        "description": "Total population",
        "search_term": "population total",
        "subject_hint": "K3",  # Ludność (Population)
    },
    "births": {
        "description": "Live births",
        "search_term": "live births",
        "subject_hint": "K3",
    },
    "deaths": {
        "description": "Deaths total",
        "search_term": "deaths total",
        "subject_hint": "K3",
    },
    "migration_net": {
        "description": "Net migration (internal + international)",
        "search_term": "migration balance",
        "subject_hint": "K3",
    },
    # --- LABOUR MARKET ---
    "registered_unemployed": {
        "description": "Registered unemployed persons",
        "search_term": "registered unemployed",
        "subject_hint": "K11",
    },
    "unemployment_rate": {
        "description": "Registered unemployment rate",
        "search_term": "unemployment rate",
        "subject_hint": "K11",
    },
    # --- LOCAL GOVERNMENT FINANCE ---
    "gmina_revenue_total": {
        "description": "Total gmina budget revenue",
        "search_term": "total revenue budget gmina",
        "subject_hint": "K16",
    },
    "gmina_own_revenue": {
        "description": "Gmina own revenue (dochody własne)",
        "search_term": "own revenue",
        "subject_hint": "K16",
    },
    "pit_share": {
        "description": "Share of PIT revenue to gmina",
        "search_term": "personal income tax",
        "subject_hint": "K16",
    },
    "cit_share": {
        "description": "Share of CIT revenue to gmina",
        "search_term": "corporate income tax",
        "subject_hint": "K16",
    },
    "investment_expenditure": {
        "description": "Investment expenditure of gmina",
        "search_term": "investment expenditure",
        "subject_hint": "K16",
    },
    # --- EDUCATION ---
    "primary_school_students": {
        "description": "Students in primary schools",
        "search_term": "primary school students",
        "subject_hint": "K12",
    },
    # --- INFRASTRUCTURE ---
    "water_supply_network_km": {
        "description": "Water supply network length (km)",
        "search_term": "water supply network",
        "subject_hint": "K7",
    },
    "sewage_network_km": {
        "description": "Sewage network length (km)",
        "search_term": "sewage network",
        "subject_hint": "K7",
    },
    # --- BUSINESS / ECONOMY ---
    "registered_entities": {
        "description": "Entities in REGON register",
        "search_term": "entities registered REGON",
        "subject_hint": "K22",
    },
    "new_entities_registered": {
        "description": "Newly registered entities",
        "search_term": "newly registered entities",
        "subject_hint": "K22",
    },
    # --- HOUSING ---
    "dwellings_completed": {
        "description": "Dwellings completed",
        "search_term": "dwellings completed",
        "subject_hint": "K19",
    },
}

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("BDL")

# ---------------------------------------------------------------------------
# API Client
# ---------------------------------------------------------------------------

class BDLClient:
    """Wrapper around the BDL REST API with rate limiting and pagination."""

    def __init__(self, api_key=None, lang="en"):
        self.session = requests.Session()
        self.lang = lang
        if api_key:
            self.session.headers["X-ClientId"] = api_key
            log.info("Using registered API key (higher rate limits)")
        else:
            log.info("Running as anonymous user (lower rate limits)")
        self.request_count = 0

    def _get(self, endpoint, params=None):
        """Make a GET request with retry logic and rate limiting."""
        if params is None:
            params = {}
        params.setdefault("format", "json")
        params.setdefault("lang", self.lang)

        url = f"{BASE_URL}/{endpoint}"

        for attempt in range(MAX_RETRIES):
            try:
                time.sleep(REQUEST_DELAY)
                resp = self.session.get(url, params=params, timeout=30)
                self.request_count += 1

                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 429:
                    wait = RETRY_DELAY * (attempt + 1)
                    log.warning(f"Rate limited (429). Waiting {wait}s...")
                    time.sleep(wait)
                elif resp.status_code == 404:
                    log.debug(f"Not found: {url}")
                    return None
                else:
                    log.warning(
                        f"HTTP {resp.status_code} for {endpoint} "
                        f"(attempt {attempt+1}/{MAX_RETRIES})"
                    )
                    time.sleep(RETRY_DELAY)
            except requests.exceptions.RequestException as e:
                log.warning(f"Request error: {e} (attempt {attempt+1})")
                time.sleep(RETRY_DELAY)

        log.error(f"Failed after {MAX_RETRIES} retries: {endpoint}")
        return None

    def _get_all_pages(self, endpoint, params=None):
        """Paginate through all results for a given endpoint."""
        if params is None:
            params = {}
        params["page-size"] = PAGE_SIZE
        params["page"] = 0

        all_results = []
        while True:
            data = self._get(endpoint, params.copy())
            if data is None:
                break

            results = data.get("results", [])
            if not results:
                break

            all_results.extend(results)

            # Check if more pages
            total_records = data.get("totalRecords", 0)
            if len(all_results) >= total_records:
                break

            params["page"] += 1
            log.debug(
                f"  Page {params['page']}: "
                f"{len(all_results)}/{total_records} records"
            )

        return all_results

    # --- Subjects (topic hierarchy) ---

    def get_subjects(self, parent_id=None):
        """Get subject categories. parent_id=None → top-level subjects."""
        params = {}
        if parent_id:
            params["parent-id"] = parent_id
        return self._get_all_pages("subjects", params)

    # --- Variables ---

    def get_variables(self, subject_id, level=None, year=None):
        """Get variables for a given subject, optionally filtered by level/year."""
        params = {"subject-id": subject_id}
        if level is not None:
            params["level"] = level
        if year is not None:
            params["year"] = year
        return self._get_all_pages("variables", params)

    def search_variables(self, name, level=None):
        """Search variables by name/keyword."""
        params = {"name": name}
        if level is not None:
            params["level"] = level
        return self._get_all_pages("variables/search", params)

    # --- Units (territorial) ---

    def get_units(self, parent_id=None, level=None):
        """Get territorial units. level=6 → gminy."""
        params = {}
        if parent_id:
            params["parent-id"] = parent_id
        if level is not None:
            params["level"] = level
        return self._get_all_pages("units", params)

    # --- Data ---

    def get_data_by_variable(self, var_id, unit_level=6, years=None,
                              unit_parent_id=None):
        """
        Download data for ONE variable across all units at given level.
        This is the most efficient way to get gmina-level panel data.
        """
        params = {"unit-level": unit_level}
        if years:
            for y in years:
                # BDL API accepts multiple year params
                pass
        if unit_parent_id:
            params["unit-parent-id"] = unit_parent_id

        # Build year params manually (requests lib handles list params)
        endpoint = f"data/by-variable/{var_id}"
        params["page-size"] = PAGE_SIZE
        params["page"] = 0

        all_results = []
        while True:
            # Build URL with year params manually
            year_str = ""
            if years:
                year_str = "&".join(f"year={y}" for y in years)

            query_params = params.copy()
            data = self._get_with_years(endpoint, query_params, years)
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

    def _get_with_years(self, endpoint, params, years=None):
        """Handle the BDL API's repeated year parameter format."""
        url = f"{BASE_URL}/{endpoint}"
        params.setdefault("format", "json")
        params.setdefault("lang", self.lang)

        # Build the query string manually to handle repeated 'year' params
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


# ---------------------------------------------------------------------------
# Phase 1: Discovery — find all variables available at gmina level
# ---------------------------------------------------------------------------

def discover_gmina_variables(client, output_dir):
    """
    Crawl the entire BDL subject tree and identify every variable
    that has data at gmina level (level 6).
    Saves results to a CSV catalogue.
    """
    log.info("=" * 70)
    log.info("PHASE 1: DISCOVERING VARIABLES AVAILABLE AT GMINA LEVEL")
    log.info("=" * 70)

    # Step 1: Get all top-level subjects (K-codes: K1, K2, ...)
    top_subjects = client.get_subjects()
    log.info(f"Found {len(top_subjects)} top-level subjects (K-codes)")

    catalogue = []

    for ks in top_subjects:
        ks_id = ks.get("id", "")
        ks_name = ks.get("name", "")
        log.info(f"\n--- Subject: {ks_id} - {ks_name} ---")

        # Step 2: Drill into G-level (groups)
        g_subjects = client.get_subjects(parent_id=ks_id)
        for gs in g_subjects:
            gs_id = gs.get("id", "")
            gs_name = gs.get("name", "")

            # Step 3: Drill into P-level (subgroups)
            p_subjects = client.get_subjects(parent_id=gs_id)
            for ps in p_subjects:
                ps_id = ps.get("id", "")
                ps_name = ps.get("name", "")
                has_vars = ps.get("hasVariables", False)

                if not has_vars:
                    # Some P-levels have further children
                    deeper = client.get_subjects(parent_id=ps_id)
                    for ds in deeper:
                        ds_id = ds.get("id", "")
                        ds_name = ds.get("name", "")
                        if ds.get("hasVariables", False):
                            _fetch_and_catalogue_vars(
                                client, ds_id,
                                ks_name, gs_name, f"{ps_name} > {ds_name}",
                                catalogue
                            )
                else:
                    _fetch_and_catalogue_vars(
                        client, ps_id,
                        ks_name, gs_name, ps_name,
                        catalogue
                    )

    # Save catalogue
    df = pd.DataFrame(catalogue)
    out_path = os.path.join(output_dir, "bdl_variable_catalogue_gmina.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info(f"\nCatalogue saved: {out_path}")
    log.info(f"Total gmina-level variables found: {len(df)}")
    log.info(f"Total API requests made: {client.request_count}")

    return df


def _fetch_and_catalogue_vars(client, subject_id, topic, group, subgroup,
                               catalogue):
    """Fetch variables for a subject and add gmina-level ones to catalogue."""
    variables = client.get_variables(subject_id, level=GMINA_LEVEL)
    if variables:
        log.info(
            f"  {subject_id}: {len(variables)} vars at gmina level "
            f"({subgroup})"
        )
        for v in variables:
            catalogue.append({
                "var_id": v.get("id"),
                "var_name": v.get("n1", v.get("name", "")),
                "var_name_full": v.get("n2", ""),
                "subject_id": subject_id,
                "topic": topic,
                "group": group,
                "subgroup": subgroup,
                "measure_unit": v.get("measureUnitName", ""),
                "level": v.get("level", ""),
                "year_from": v.get("years", [""])[0] if v.get("years") else "",
                "year_to": (
                    v.get("years", [""])[-1] if v.get("years") else ""
                ),
                "n_years": len(v.get("years", [])),
            })


# ---------------------------------------------------------------------------
# Phase 2: Download data
# ---------------------------------------------------------------------------

def download_variable_data(client, var_id, var_name, years, output_dir):
    """
    Download data for a single variable across all gminas.
    Returns a list of dicts: [{unit_id, unit_name, year, value}, ...]
    """
    log.info(f"  Downloading var {var_id}: {var_name}")

    # Chunk years into groups of 10 to keep URL length manageable
    year_chunks = [years[i:i+10] for i in range(0, len(years), 10)]
    all_rows = []

    for chunk in year_chunks:
        results = client.get_data_by_variable(
            var_id=var_id,
            unit_level=GMINA_LEVEL,
            years=chunk
        )
        if not results:
            continue

        for unit_data in results:
            unit_id = unit_data.get("id", "")
            unit_name = unit_data.get("name", "")
            values = unit_data.get("values", [])

            for val in values:
                year = val.get("year")
                value = val.get("val")
                attr_id = val.get("attrId")

                all_rows.append({
                    "unit_id": unit_id,
                    "unit_name": unit_name,
                    "year": year,
                    "value": value,
                    "attr_id": attr_id,
                })

    log.info(f"    → {len(all_rows)} observations")
    return all_rows


def download_curated_variables(client, output_dir, start_year, end_year):
    """
    Strategy: use search to find variable IDs for curated topics,
    then download data for each.
    """
    log.info("=" * 70)
    log.info("PHASE 2: DOWNLOADING CURATED VARIABLES FOR ALL GMINAS")
    log.info(f"Years: {start_year}–{end_year}")
    log.info("=" * 70)

    years = list(range(start_year, end_year + 1))
    all_panels = {}

    for key, spec in CURATED_VARIABLES.items():
        log.info(f"\n--- {key}: {spec['description']} ---")

        # Try to find the variable by searching
        search_results = client.search_variables(
            spec["search_term"], level=GMINA_LEVEL
        )

        if not search_results:
            log.warning(f"  No gmina-level variables found for '{key}'")
            log.info(f"  Try adjusting the search term or check the catalogue")
            continue

        # Take the first (most relevant) result
        var = search_results[0]
        var_id = var.get("id")
        var_name = var.get("n1", var.get("name", key))

        log.info(
            f"  Found: var_id={var_id}, name='{var_name}' "
            f"(out of {len(search_results)} results)"
        )
        if len(search_results) > 1:
            log.info("  Other candidates:")
            for sv in search_results[1:5]:
                log.info(
                    f"    - id={sv.get('id')}: {sv.get('n1', sv.get('name'))}"
                )

        # Download the data
        rows = download_variable_data(
            client, var_id, var_name, years, output_dir
        )

        if rows:
            df = pd.DataFrame(rows)
            df.rename(columns={"value": key}, inplace=True)

            # Save individual variable file
            var_path = os.path.join(output_dir, "variables", f"{key}.csv")
            os.makedirs(os.path.dirname(var_path), exist_ok=True)
            df.to_csv(var_path, index=False, encoding="utf-8-sig")

            all_panels[key] = df

    # Merge all variables into one panel
    if all_panels:
        _merge_panels(all_panels, output_dir)

    log.info(f"\nTotal API requests made: {client.request_count}")


def download_from_catalogue(client, catalogue_path, output_dir,
                             start_year, end_year):
    """Download ALL variables listed in the discovery catalogue."""
    log.info("=" * 70)
    log.info("DOWNLOADING ALL VARIABLES FROM CATALOGUE")
    log.info("=" * 70)

    cat = pd.read_csv(catalogue_path)
    years = list(range(start_year, end_year + 1))
    total = len(cat)

    log.info(f"Catalogue has {total} variables. This will take a while!")
    log.warning(
        f"Estimated API calls: ~{total * len(years) // 10 * 30} "
        f"(depends on gmina count per page)"
    )

    all_panels = {}

    for idx, row in cat.iterrows():
        var_id = str(row["var_id"])
        var_name = str(row.get("var_name", var_id))
        safe_name = f"v{var_id}"

        log.info(f"\n[{idx+1}/{total}] var_id={var_id}: {var_name}")

        rows = download_variable_data(
            client, var_id, var_name, years, output_dir
        )

        if rows:
            df = pd.DataFrame(rows)
            df.rename(columns={"value": safe_name}, inplace=True)

            var_path = os.path.join(
                output_dir, "variables", f"{safe_name}.csv"
            )
            os.makedirs(os.path.dirname(var_path), exist_ok=True)
            df.to_csv(var_path, index=False, encoding="utf-8-sig")

            all_panels[safe_name] = df

        # Progress checkpoint every 50 variables
        if (idx + 1) % 50 == 0:
            log.info(
                f"--- Checkpoint: {idx+1}/{total} variables downloaded ---"
            )
            log.info(f"--- API requests so far: {client.request_count} ---")

    if all_panels:
        _merge_panels(all_panels, output_dir)


def _merge_panels(panels, output_dir):
    """Merge individual variable DataFrames into a unified panel dataset."""
    log.info("\n--- Merging into unified panel dataset ---")

    merged = None
    for key, df in panels.items():
        # Keep only the key columns for merging
        cols = ["unit_id", "unit_name", "year", key]
        df_slim = df[cols].drop_duplicates(subset=["unit_id", "year"])

        if merged is None:
            merged = df_slim
        else:
            merged = merged.merge(
                df_slim[["unit_id", "year", key]],
                on=["unit_id", "year"],
                how="outer"
            )

    if merged is not None:
        # Sort
        merged.sort_values(["unit_id", "year"], inplace=True)

        # Add TERYT-based identifiers
        merged["powiat_id"] = merged["unit_id"].str[:6]  # first 6 digits
        merged["voivodship_id"] = merged["unit_id"].str[:2]  # first 2 digits

        out_path = os.path.join(output_dir, "bdl_gmina_panel.csv")
        merged.to_csv(out_path, index=False, encoding="utf-8-sig")
        log.info(f"Panel dataset saved: {out_path}")
        log.info(
            f"  Shape: {merged.shape[0]} rows × {merged.shape[1]} columns"
        )
        log.info(
            f"  Gminas: {merged['unit_id'].nunique()} | "
            f"Years: {merged['year'].nunique()}"
        )
        log.info(
            f"  Variables: {merged.shape[1] - 4}"  # minus id cols
        )

        # Summary statistics
        summary_path = os.path.join(output_dir, "bdl_panel_summary.txt")
        with open(summary_path, "w") as f:
            f.write("BDL Gmina Panel Dataset — Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Rows: {merged.shape[0]}\n")
            f.write(f"Columns: {merged.shape[1]}\n")
            f.write(f"Unique gminas: {merged['unit_id'].nunique()}\n")
            f.write(f"Year range: {merged['year'].min()} – {merged['year'].max()}\n\n")
            f.write("Variables included:\n")
            for col in merged.columns:
                if col not in ["unit_id", "unit_name", "year",
                               "powiat_id", "voivodship_id"]:
                    non_null = merged[col].notna().sum()
                    f.write(f"  {col}: {non_null} non-null values\n")

        log.info(f"Summary saved: {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download gmina-level time-series data from Poland's BDL API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bdl_gmina_downloader.py --discover
  python bdl_gmina_downloader.py --download --api-key YOUR_KEY
  python bdl_gmina_downloader.py --download-all --start-year 2004
        """,
    )
    parser.add_argument(
        "--discover", action="store_true",
        help="Discover all gmina-level variables and save catalogue"
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download curated key variables for all gminas"
    )
    parser.add_argument(
        "--download-all", action="store_true",
        help="Download ALL variables from catalogue (very large!)"
    )
    parser.add_argument("--api-key", default=None, help="BDL API key")
    parser.add_argument(
        "--start-year", type=int, default=DEFAULT_START_YEAR,
        help=f"First year (default: {DEFAULT_START_YEAR})"
    )
    parser.add_argument(
        "--end-year", type=int, default=DEFAULT_END_YEAR,
        help=f"Last year (default: {DEFAULT_END_YEAR})"
    )
    parser.add_argument(
        "--output-dir", default="./bdl_output",
        help="Output directory (default: ./bdl_output)"
    )
    parser.add_argument(
        "--lang", default="en", choices=["en", "pl"],
        help="Language for variable names (default: en)"
    )

    args = parser.parse_args()

    if not any([args.discover, args.download, args.download_all]):
        parser.print_help()
        print("\n⚠️  Please specify --discover, --download, or --download-all")
        return

    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "variables"), exist_ok=True)

    # Add file logging
    fh = logging.FileHandler(
        os.path.join(args.output_dir, "bdl_download.log"),
        encoding="utf-8"
    )
    fh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    log.addHandler(fh)

    # Initialise client
    client = BDLClient(api_key=args.api_key, lang=args.lang)

    log.info("BDL Gmina Data Downloader")
    log.info(f"Output directory: {args.output_dir}")
    log.info(f"Language: {args.lang}")

    if args.discover:
        discover_gmina_variables(client, args.output_dir)

    if args.download:
        download_curated_variables(
            client, args.output_dir, args.start_year, args.end_year
        )

    if args.download_all:
        cat_path = os.path.join(
            args.output_dir, "bdl_variable_catalogue_gmina.csv"
        )
        if not os.path.exists(cat_path):
            log.info("No catalogue found — running discovery first...")
            discover_gmina_variables(client, args.output_dir)
        download_from_catalogue(
            client, cat_path, args.output_dir,
            args.start_year, args.end_year
        )


if __name__ == "__main__":
    main()
