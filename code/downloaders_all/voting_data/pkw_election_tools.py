# -*- coding: utf-8 -*-
"""
===============================================================================
PKW ELECTION DATA TOOLS
===============================================================================
1. process_2015_sejm()  — Processes the 2015-gl-lis-gm.xls file into the
                          panel_populist_gmina format
2. Presidential election scraper + processor

Author: [Your name]
Project: Geography of Discontent — LSE Policy Paper
===============================================================================
"""

import pandas as pd
import numpy as np
import os
import re
import requests
import zipfile
import io
from pathlib import Path


# =============================================================================
# PART 1: PROCESS 2015 PARLIAMENTARY ELECTION INTO PANEL FORMAT
# =============================================================================

def process_2015_sejm(xls_path, existing_panel_path=None, output_path=None):
    """
    Processes the 2015 Sejm gmina-level vote counts into the same schema
    as the existing panel_populist_gmina.csv.
    
    Parameters
    ----------
    xls_path : str
        Path to '2015-gl-lis-gm.xls'
    existing_panel_path : str, optional
        Path to existing panel_populist_gmina.csv to append to
    output_path : str, optional
        Where to save the combined panel
    
    Returns
    -------
    DataFrame — the 2015 panel rows (or combined if existing_panel_path given)
    
    Classification (matching existing panel scheme):
    ------------------------------------------------
    POPULIST (broad):  PiS + Kukiz'15 + Samoobrona
    POPULIST (strict): Kukiz'15 + Samoobrona (excl PiS)
    FAR-RIGHT:         PiS + KORWiN + Kukiz'15 + Braun + KNP
    FAR-RIGHT (strict): KORWiN + Braun + KNP
    FAR-LEFT:          None in 2015
    EUROSCEPTIC:       PiS + Kukiz'15 + KORWiN + Braun + KNP + Samoobrona
    """
    
    df = pd.read_excel(xls_path)
    print(f"Loaded 2015 Sejm data: {len(df)} rows")
    
    # Identify party columns by partial match (handles quote variants)
    def find_col(df, pattern):
        matches = [c for c in df.columns if pattern in c]
        if matches:
            return matches[0]
        return None
    
    pis_col = find_col(df, 'Prawo i Sprawiedliwość')
    korwin_col = find_col(df, 'KORWiN')
    kukiz_col = find_col(df, 'Kukiz')
    braun_col = find_col(df, 'Braun')
    knp_col = find_col(df, 'Kongres Nowej Prawicy')
    samoobrona_col = find_col(df, 'Samoobrona')
    valid_col = 'Głosy ważne'
    
    # Validate
    for name, col in [('PiS', pis_col), ('KORWiN', korwin_col), ('Kukiz', kukiz_col),
                       ('Braun', braun_col), ('Valid', valid_col)]:
        if col is None:
            raise ValueError(f"Could not find column for {name}")
        print(f"  {name}: '{col}'")
    
    # Convert to numeric
    party_cols = [c for c in [pis_col, korwin_col, kukiz_col, braun_col, knp_col, samoobrona_col] if c is not None]
    for col in party_cols + [valid_col]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Handle potentially missing columns
    def safe_col(col):
        return df[col] if col is not None else 0
    
    # Build panel row
    panel = pd.DataFrame({
        'teryt': df['TERYT'],
        'gmina_name': df['Gmina'],
        'year': 2015,
        'valid_votes': df[valid_col].astype(int),
        'votes_populist': safe_col(pis_col) + safe_col(kukiz_col) + safe_col(samoobrona_col),
        'votes_populist_strict': safe_col(kukiz_col) + safe_col(samoobrona_col),
        'votes_farright': safe_col(pis_col) + safe_col(korwin_col) + safe_col(kukiz_col) + safe_col(braun_col) + safe_col(knp_col),
        'votes_farright_strict': safe_col(korwin_col) + safe_col(braun_col) + safe_col(knp_col),
        'votes_farleft': 0.0,
        'votes_eurosceptic': safe_col(pis_col) + safe_col(kukiz_col) + safe_col(korwin_col) + safe_col(braun_col) + safe_col(knp_col) + safe_col(samoobrona_col),
    })
    
    # Compute shares
    for vote_col in ['votes_populist', 'votes_populist_strict', 'votes_farright',
                      'votes_farright_strict', 'votes_farleft', 'votes_eurosceptic']:
        share_col = vote_col.replace('votes_', 'share_')
        panel[share_col] = panel[vote_col] / panel['valid_votes']
    
    panel['votes_populist_or_farright'] = panel[['votes_populist', 'votes_farright']].max(axis=1)
    panel['share_populist_or_farright'] = panel['votes_populist_or_farright'] / panel['valid_votes']
    
    # Validate
    national_pis = safe_col(pis_col).sum() / df[valid_col].sum()
    print(f"\nValidation — PiS national share: {national_pis:.1%} (expected ~37.6%)")
    
    # Append to existing panel if provided
    if existing_panel_path and os.path.exists(existing_panel_path):
        existing = pd.read_csv(existing_panel_path, encoding='utf-8-sig')
        # Remove any existing 2015 rows
        existing = existing[existing['year'] != 2015]
        combined = pd.concat([existing, panel], ignore_index=True).sort_values(['teryt', 'year'])
        print(f"Combined panel: {len(combined)} rows, years: {sorted(combined['year'].unique())}")
        
        if output_path:
            combined.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"Saved: {output_path}")
        return combined
    
    if output_path:
        panel.to_csv(output_path, index=False, encoding='utf-8-sig')
    return panel


# =============================================================================
# PART 2: PRESIDENTIAL ELECTION SCRAPER
# =============================================================================

# URL map for presidential elections at gmina level (vote counts)
# We use ROUND 1 results (not runoff) because Round 1 shows full party spectrum
# Round 2 is binary (e.g. Duda vs Trzaskowski) — less informative

PRESIDENTIAL_URLS = {
    2000: {
        'url': 'https://danewyborcze.kbw.gov.pl/dane/2000/prezydent/gm-kraj2000.xls',
        'format': 'xls',
        'round': 1,
        'header': 0,
        'teryt_col': 'TERYT',
        'gmina_col': 'Gmina',
        'valid_col': 'Głosy ważne',
        # Candidates and their political alignment:
        'candidates': {
            # PiS-aligned / right-populist
            'pis_proxy': [],  # No PiS candidate in 2000
            'right': ['Marian Krzaklewski'],  # AWS/Solidarity right
            'populist': ['Andrzej Lepper'],  # Samoobrona
            'farright': [],
            'incumbent_liberal': ['Aleksander Kwaśniewski'],
        },
        'notes': 'Kwaśniewski won in round 1. No clear PiS proxy.',
    },
    2005: {
        'url': 'https://danewyborcze.kbw.gov.pl/dane/2005/prezydent/1/1456148660_37033.xls',
        'format': 'xls',
        'round': 1,
        'header': 0,
        'teryt_col': None,  # Need to inspect
        'gmina_col': None,
        'valid_col': None,
        'candidates': {
            'pis_proxy': ['Lech Kaczyński'],
            'populist': ['Andrzej Lepper'],
            'farright': [],
        },
        'notes': 'Lech Kaczyński = PiS candidate. Won runoff vs Tusk.',
    },
    2010: {
        'url': 'https://danewyborcze.kbw.gov.pl/dane/2010/prezydent/1/pzt2010-wyn-gmn.csv',
        'format': 'csv',
        'round': 1,
        'encoding': 'utf-8',
        'sep': ';',
        'teryt_col': None,
        'gmina_col': None,
        'valid_col': None,
        'candidates': {
            'pis_proxy': ['Jarosław Kaczyński'],
            'populist': [],
            'farright': [],
        },
        'notes': 'J. Kaczyński = PiS candidate. Lost runoff to Komorowski.',
    },
    2015: {
        'url': None,  # Use wybory.gov.pl API or local file
        'format': 'csv',
        'round': 1,
        'candidates': {
            'pis_proxy': ['Andrzej Duda'],
            'populist': ['Paweł Kukiz'],
            'farright': ['Janusz Korwin-Mikke', 'Grzegorz Braun'],
        },
        'notes': 'Duda = PiS candidate. Won runoff vs Komorowski. Kukiz got 20.8% in R1.',
    },
    2020: {
        'url': None,  # Use wybory.gov.pl API
        'format': 'csv',
        'round': 1,
        'candidates': {
            'pis_proxy': ['Andrzej Duda'],
            'populist': [],
            'farright': ['Krzysztof Bosak'],  # Konfederacja
        },
        'notes': 'Duda = PiS incumbent. Won runoff vs Trzaskowski. Bosak = Konfederacja.',
    },
}


def download_presidential_files(output_dir, years=None):
    """
    Downloads presidential election gmina-level results from danewyborcze.kbw.gov.pl.
    
    Parameters
    ----------
    output_dir : str
        Base directory to save files (creates year subdirectories)
    years : list, optional
        Which years to download (default: all available)
    """
    output_dir = Path(output_dir)
    
    if years is None:
        years = [2000, 2005, 2010]  # Only these have direct download URLs
    
    for year in years:
        spec = PRESIDENTIAL_URLS.get(year)
        if not spec or not spec['url']:
            print(f"  {year}: No direct download URL available (use wybory.gov.pl)")
            continue
        
        year_dir = output_dir / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        
        url = spec['url']
        filename = url.split('/')[-1]
        filepath = year_dir / filename
        
        if filepath.exists():
            print(f"  {year}: Already downloaded → {filepath}")
            continue
        
        print(f"  {year}: Downloading {url}...")
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            
            if url.endswith('.zip'):
                with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                    zf.extractall(year_dir)
                    print(f"  {year}: Extracted {zf.namelist()}")
            else:
                with open(filepath, 'wb') as f:
                    f.write(r.content)
                print(f"  {year}: Saved → {filepath}")
        except Exception as e:
            print(f"  {year}: FAILED — {e}")
    
    # For 2015 and 2020, guide user to wybory.gov.pl
    print(f"\n--- Manual downloads needed ---")
    print(f"2015 presidential: go to https://wybory.gov.pl → Prezydent 2015 → download gmina results")
    print(f"2020 presidential: go to https://wybory.gov.pl → Prezydent 2020 → download gmina results")
    print(f"Save files to: {output_dir / '2015'} and {output_dir / '2020'}")


# =============================================================================
# PART 3: PROCESS PRESIDENTIAL RESULTS INTO PANEL
# =============================================================================

def process_presidential_2000(xls_path):
    """Process 2000 presidential election (gmina level)."""
    df = pd.read_excel(xls_path)
    print(f"2000 presidential: {len(df)} rows, columns: {list(df.columns)[:10]}...")
    
    # Find candidate columns (they typically start with candidate name or number)
    # 2000 candidates: Kwaśniewski, Krzaklewski, Olechowski, Lepper, etc.
    candidate_cols = [c for c in df.columns if c not in [
        'Nr okr.', 'TERYT', 'Gmina', 'L. obw.', 'Liczba wyborców',
        'Otrzymane karty', 'Wydane karty', 'Karty wyjęte z urny',
        'Karty nieważne', 'Karty ważne', 'Głosy nieważne', 'Głosy ważne',
    ] and not c.startswith(('w tym', 'Niewykorzystane', 'Liczba wyborców gł'))]
    
    print(f"  Candidate columns: {candidate_cols}")
    
    for c in candidate_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    
    valid_col = 'Głosy ważne' if 'Głosy ważne' in df.columns else 'Karty ważne'
    
    # Find Lepper column (Samoobrona populist)
    lepper_col = [c for c in candidate_cols if 'Lepper' in c]
    lepper_votes = df[lepper_col[0]] if lepper_col else 0
    
    # Find Krzaklewski (AWS right)
    krzaklewski_col = [c for c in candidate_cols if 'Krzaklewski' in c]
    krzaklewski_votes = df[krzaklewski_col[0]] if krzaklewski_col else 0
    
    panel = pd.DataFrame({
        'teryt': df['TERYT'],
        'gmina_name': df['Gmina'],
        'year': 2000,
        'election_type': 'presidential',
        'valid_votes': pd.to_numeric(df[valid_col], errors='coerce').fillna(0).astype(int),
        'votes_pis_proxy': 0,  # No PiS candidate in 2000
        'votes_populist_candidates': lepper_votes,
        'votes_right_candidates': krzaklewski_votes + lepper_votes,
    })
    
    panel['share_pis_proxy'] = panel['votes_pis_proxy'] / panel['valid_votes']
    panel['share_populist_candidates'] = panel['votes_populist_candidates'] / panel['valid_votes']
    panel['share_right_candidates'] = panel['votes_right_candidates'] / panel['valid_votes']
    
    return panel


def process_presidential_generic(filepath, year, format='xls', **kwargs):
    """
    Generic processor for presidential election files.
    Returns a DataFrame with TERYT, gmina_name, year, valid_votes,
    and votes/shares for PiS-proxy and populist candidates.
    
    You'll need to inspect each file's columns first and customize.
    """
    if format == 'xls':
        df = pd.read_excel(filepath, **kwargs)
    elif format == 'csv':
        df = pd.read_csv(filepath, encoding=kwargs.get('encoding', 'utf-8'),
                         sep=kwargs.get('sep', ';'), **{k:v for k,v in kwargs.items() 
                         if k not in ['encoding', 'sep']})
    
    print(f"{year} presidential: {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
    print(f"  → INSPECT COLUMNS and add processing logic for this year")
    
    return df


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PKW Election Data Tools")
    parser.add_argument('--action', choices=['process_2015', 'download_presidential'],
                       required=True)
    parser.add_argument('--input', help='Input file path')
    parser.add_argument('--existing-panel', help='Existing panel CSV to append to')
    parser.add_argument('--output', help='Output path')
    parser.add_argument('--output-dir', help='Output directory for downloads')
    
    args = parser.parse_args()
    
    if args.action == 'process_2015':
        process_2015_sejm(args.input, args.existing_panel, args.output)
    elif args.action == 'download_presidential':
        download_presidential_files(args.output_dir or 'data/clean/outcome/Elections/raw_presidential')
