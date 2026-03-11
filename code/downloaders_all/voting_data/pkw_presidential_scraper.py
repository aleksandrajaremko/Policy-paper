# -*- coding: utf-8 -*-
"""
===============================================================================
PKW Presidential Election Scraper — Gmina Level
===============================================================================

Downloads gmina-level results for Polish presidential elections from
danewyborcze.kbw.gov.pl and wybory.gov.pl.

Usage:
  python pkw_presidential_scraper.py --output-dir "data/Elections/raw_presidential"
  python pkw_presidential_scraper.py --output-dir "data/Elections/raw_presidential" --dry-run

Available elections: 2000, 2005, 2010, 2015, 2020, 2025

Notes:
  - 2000, 2005, 2010: from danewyborcze.kbw.gov.pl (direct XLS/CSV links)
  - 2015: from danewyborcze.kbw.gov.pl (XLS in ZIP)
  - 2020: from wybory.gov.pl (CSV in ZIP)
  - 2025: from wybory.gov.pl (CSV in ZIP) — if available
  
  For Round 1 (first round) — this gives full candidate spectrum.
  Round 2 (runoff) is binary and less informative for populist classification.
===============================================================================
"""

import os
import sys
import argparse
import requests
import zipfile
import io
from pathlib import Path


# =============================================================================
# FILE REGISTRY — gmina-level results, Round 1 (vote counts, not percentages)
# =============================================================================

FILES = {
    2000: {
        'round1_gmina_counts': {
            'url': 'https://danewyborcze.kbw.gov.pl/dane/2000/prezydent/gm-kraj2000.xls',
            'filename': 'prez2000-gm-kraj.xls',
        },
    },
    2005: {
        'round1_gmina_counts': {
            'url': 'https://danewyborcze.kbw.gov.pl/dane/2005/prezydent/1/1456148660_37033.xls',
            'filename': 'prez2005-gm-r1-counts.xls',
        },
        'round2_gmina_counts': {
            'url': 'https://danewyborcze.kbw.gov.pl/dane/2005/prezydent/2/1456148721_37554.xls',
            'filename': 'prez2005-gm-r2-counts.xls',
        },
    },
    2010: {
        'round1_gmina_counts': {
            'url': 'https://danewyborcze.kbw.gov.pl/dane/2010/prezydent/1/pzt2010-wyn-gmn.csv',
            'filename': 'prez2010-gm-r1-counts.csv',
        },
        'round2_gmina_counts': {
            'url': 'https://danewyborcze.kbw.gov.pl/dane/2010/prezydent/2/pzt2010-wyn-gmn.csv',
            'filename': 'prez2010-gm-r2-counts.csv',
        },
    },
    2015: {
        'round1_gmina_counts': {
            'url': 'https://danewyborcze.kbw.gov.pl/dane/2015/prezydent/1/prez2015_1_tura_gm.xls',
            'filename': 'prez2015-gm-r1-counts.xls',
        },
        'round2_gmina_counts': {
            'url': 'https://danewyborcze.kbw.gov.pl/dane/2015/prezydent/2/prez2015_2_tura_gm.xls',
            'filename': 'prez2015-gm-r2-counts.xls',
        },
    },
    2020: {
        'round1_gmina_counts': {
            'url': 'https://danewyborcze.kbw.gov.pl/dane/2020/prezydent/1/wyniki_gl_na_kand_po_gminach_csv.zip',
            'filename': 'prez2020-gm-r1.zip',
            'is_zip': True,
        },
        'round2_gmina_counts': {
            'url': 'https://danewyborcze.kbw.gov.pl/dane/2020/prezydent/2/wyniki_gl_na_kand_po_gminach_csv.zip',
            'filename': 'prez2020-gm-r2.zip',
            'is_zip': True,
        },
    },
    2025: {
        'round1_gmina_counts': {
            'url': 'https://danewyborcze.kbw.gov.pl/dane/2025/prezydent/1/wyniki_gl_na_kandydatow_po_gminach_csv.1752152970.zip',
            'filename': 'prez2025-gm-r1.zip',
            'is_zip': True,
        },
        'round2_gmina_counts': {
            'url': 'https://danewyborcze.kbw.gov.pl/dane/2025/prezydent/2/wyniki_gl_na_kandydatow_po_gminach_w_drugiej_turze_csv.1752152970.zip',
            'filename': 'prez2025-gm-r2.zip',
            'is_zip': True,
        },
    },
}

# Candidate → political alignment mapping for each election
CANDIDATES = {
    2000: {
        'pis_proxy': [],  # PiS didn't exist yet as a major force
        'populist': ['Lepper'],  # Andrzej Lepper (Samoobrona)
        'right': ['Krzaklewski'],  # Marian Krzaklewski (AWS)
        'liberal': ['Kwaśniewski', 'Olechowski'],
    },
    2005: {
        'pis_proxy': ['Kaczyński'],  # Lech Kaczyński
        'populist': ['Lepper'],  # Andrzej Lepper
        'farright': [],
        'liberal': ['Tusk'],  # Donald Tusk (PO)
    },
    2010: {
        'pis_proxy': ['Kaczyński'],  # Jarosław Kaczyński
        'populist': [],
        'farright': ['Korwin'],  # Janusz Korwin-Mikke
        'liberal': ['Komorowski'],  # Bronisław Komorowski (PO)
    },
    2015: {
        'pis_proxy': ['Duda'],  # Andrzej Duda
        'populist': ['Kukiz'],  # Paweł Kukiz
        'farright': ['Korwin-Mikke', 'Braun'],
        'liberal': ['Komorowski'],
    },
    2020: {
        'pis_proxy': ['Duda'],  # Andrzej Duda (incumbent)
        'populist': [],
        'farright': ['Bosak'],  # Krzysztof Bosak (Konfederacja)
        'liberal': ['Trzaskowski'],  # Rafał Trzaskowski (PO)
    },
}


def download_all(output_dir, years=None, dry_run=False, round_filter='round1'):
    """
    Download presidential election gmina-level files.
    
    Parameters
    ----------
    output_dir : str or Path
    years : list of int, optional — defaults to all
    dry_run : bool — if True, just print URLs without downloading
    round_filter : str — 'round1', 'round2', or 'all'
    """
    output_dir = Path(output_dir)
    
    if years is None:
        years = sorted(FILES.keys())
    
    print(f"Presidential Election Scraper")
    print(f"Output: {output_dir}")
    print(f"Years: {years}")
    print(f"{'DRY RUN' if dry_run else 'DOWNLOADING'}")
    print("=" * 60)
    
    for year in years:
        if year not in FILES:
            print(f"\n{year}: No files registered — skipping")
            continue
        
        year_dir = output_dir / str(year)
        
        for file_key, spec in FILES[year].items():
            # Filter by round
            if round_filter != 'all':
                if round_filter not in file_key:
                    continue
            
            url = spec['url']
            filename = spec['filename']
            filepath = year_dir / filename
            
            if dry_run:
                print(f"  [DRY] {year}/{file_key}: {url} → {filepath}")
                continue
            
            if filepath.exists():
                print(f"  {year}/{file_key}: Already exists → {filepath}")
                continue
            
            year_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"  {year}/{file_key}: Downloading {url}...")
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                
                if spec.get('is_zip'):
                    # Save the zip and also extract
                    with open(filepath, 'wb') as f:
                        f.write(r.content)
                    try:
                        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                            zf.extractall(year_dir)
                            print(f"    Extracted: {zf.namelist()}")
                    except zipfile.BadZipFile:
                        print(f"    WARNING: Not a valid ZIP — saved raw file")
                else:
                    with open(filepath, 'wb') as f:
                        f.write(r.content)
                
                print(f"    Saved → {filepath} ({len(r.content)/1024:.0f} KB)")
                
            except requests.exceptions.HTTPError as e:
                print(f"    FAILED (HTTP {e.response.status_code}): {e}")
            except Exception as e:
                print(f"    FAILED: {e}")
    
    print(f"\n{'='*60}")
    print(f"Download complete. Files in: {output_dir}")
    if not dry_run:
        print(f"\nNext step: process with pkw_election_tools.py")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Polish presidential election gmina-level results from PKW"
    )
    parser.add_argument(
        '--output-dir',
        default='data/clean/outcome/Elections/raw_presidential',
        help='Directory to save downloaded files'
    )
    parser.add_argument(
        '--years',
        nargs='+',
        type=int,
        default=None,
        help='Specific years to download (default: all)'
    )
    parser.add_argument(
        '--round',
        choices=['round1', 'round2', 'all'],
        default='round1',
        help='Which round to download (default: round1)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print URLs without downloading'
    )
    
    args = parser.parse_args()
    download_all(args.output_dir, args.years, args.dry_run, args.round)
