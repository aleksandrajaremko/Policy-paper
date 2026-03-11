# -*- coding: utf-8 -*-
"""
================================================================================
EU COHESION FUNDS → GMINA-LEVEL PANEL PIPELINE
================================================================================

Consolidates the processing of EU project lists from three programming periods
(2007-2013, 2014-2020, 2021-2027) into a single gmina × year panel dataset
with EU funding amounts and per-capita rates.

Author: [Your name]
Project: Geography of Discontent — LSE Policy Paper
Last updated: 2026-03-09

DATA SOURCES:
  - Lista projektów FE 2007-2013 (Umowy_wszystko_POKL + INNE from KSI SIMIK)
  - Lista projektów FE 2014-2020 (from dane.gov.pl / SL2014)
  - Lista projektów FE 2021-2027 (from dane.gov.pl / CST2021)
  - TERYT register (teryt_klucz_powiaty_gminy_lata_1999_2025)

PIPELINE STEPS (per period):
  1. READ raw Excel project lists
  2. RENAME columns to common schema
  3. FILTER: remove non-cohesion funds, national-level projects
  4. PARSE location strings → extract voivodeship / powiat / gmina
  5. EXPLODE multi-location projects (divide finances equally)
  6. MATCH to TERYT codes (3-tier: gmina → fallback → powiat)
  7. DISTRIBUTE funding over time (quarterly weights from start/end dates)
  8. DISAGGREGATE powiat-only matches to constituent gminas
  9. AGGREGATE to gmina × year panel
  10. COMBINE all periods + merge population for per-capita rates

METHODOLOGICAL DECISIONS (document in paper appendix):
  D1. National-level projects ("Cały Kraj") are EXCLUDED entirely
  D2. Multi-location projects: financial values divided EQUALLY across locations
  D3. Temporal distribution: completed projects spread proportionally by quarter;
      incomplete/undated projects assigned to signing date year (lump sum)
  D4. Powiat-only geocodes: disaggregated EQUALLY to all gminas in that powiat
  D5. Fund filter: 2007-13 all; 2014-20 excludes BAR; 2021-27 keeps EFRR/EFS+/FST/FS
  D6. TERYT matching: 3-tier (full gmina match → fallback by name → powiat only)
  D7. Overlapping years between periods kept separate, then summed in final panel

USAGE:
  from eu_funds_pipeline import EUFundsPipeline
  pipeline = EUFundsPipeline(config)
  pipeline.run_all()
================================================================================
"""

import pandas as pd
import numpy as np
import re
import os
import pickle
import glob
import logging
from datetime import datetime
from pathlib import Path

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("EU_FUNDS")

# ============================================================================
# DECISION LOG — tracks every filtering/transformation decision with row counts
# ============================================================================

class DecisionLog:
    """Tracks methodological decisions and their quantitative impact."""
    
    def __init__(self):
        self.entries = []
    
    def record(self, step, description, rows_before, rows_after, 
               pln_affected=None, period=None):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "period": period or "all",
            "step": step,
            "description": description,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_delta": rows_after - rows_before,
            "pln_affected": pln_affected,
        }
        self.entries.append(entry)
        log.info(
            f"[{step}] {description} | "
            f"rows: {rows_before:,} → {rows_after:,} "
            f"(Δ{rows_after - rows_before:+,})"
            + (f" | PLN affected: {pln_affected:,.0f}" if pln_affected else "")
        )
    
    def to_dataframe(self):
        return pd.DataFrame(self.entries)
    
    def save(self, path):
        self.to_dataframe().to_csv(path, index=False, encoding="utf-8-sig")


# ============================================================================
# 1. TEXT NORMALISATION
# ============================================================================

def norm_text(s):
    """
    Standardises Polish administrative names for TERYT matching.
    Removes prefixes like 'm.st.', 'powiat', dashes, dots, spaces.
    Also strips TERYT-style " od YYYY" suffixes.
    """
    if pd.isna(s):
        return ""
    s = str(s).lower().strip()
    s = re.sub(r'\s+od\s+\d{4}', '', s)           # "nazwa od 2020" → "nazwa"
    s = re.sub(r'\s*\(.*?\)', '', s)                # remove parenthetical notes
    for pat in ['m.st.', 'm.', 'st.', 'miasto', 'powiat', 'gmina', '-', '.', ' ']:
        s = s.replace(pat, '')
    return s


# ============================================================================
# 2. TERYT LOOKUP BUILDER
# ============================================================================

def build_teryt_lookup(path_excel):
    """
    Builds three lookup dictionaries from the TERYT master Excel:
      1. primary:  (woj_id, powiat_norm, gmina_norm) → teryt_7digit
      2. fallback: (woj_id, gmina_norm) → teryt_7digit  [only if unambiguous]
      3. powiat:   (woj_id, powiat_norm) → powiat_4digit
    
    Also builds the powiat→gminas hierarchy for disaggregation.
    
    Returns: (primary_map, fallback_map, powiat_map, hierarchy_map, norm_func)
    """
    log.info(f"Building TERYT lookup from: {path_excel}")
    
    cols = ['region', 'nazwa_powiatu', 'nazwa_gminy', 'teryt_2025']
    try:
        df = pd.read_excel(path_excel, sheet_name='gminy', dtype=str, usecols=cols)
    except ValueError:
        df = pd.read_excel(path_excel, sheet_name='gminy', dtype=str)
    
    woj_id = df['region'].astype(str).str.split('.').str[0].str.zfill(2)
    pow_norm = df['nazwa_powiatu'].apply(norm_text)
    gmi_norm = df['nazwa_gminy'].apply(norm_text)
    target_id = df['teryt_2025'].astype(str).str.split('.').str[0].str.zfill(7)
    
    # 1. Primary: (woj, powiat, gmina) → teryt7
    primary = dict(zip(zip(woj_id, pow_norm, gmi_norm), target_id))
    
    # 2. Fallback: (woj, gmina) → teryt7 [only if unique within voivodeship]
    df['woj_id'] = woj_id
    df['gmi_norm'] = gmi_norm
    df['target_id'] = target_id
    uniq = df.groupby(['woj_id', 'gmi_norm'])['target_id'].nunique()
    valid_fb = uniq[uniq == 1].index
    fb_df = df.set_index(['woj_id', 'gmi_norm']).loc[valid_fb]
    fallback = fb_df['target_id'].to_dict()
    
    # 3. Powiat: (woj, powiat) → powiat_4digit
    df['powiat_4'] = target_id.str[:4]
    pow_df = df[['woj_id', 'nazwa_powiatu', 'powiat_4']].drop_duplicates()
    pow_df['pow_norm'] = pow_df['nazwa_powiatu'].apply(norm_text)
    powiat_map = dict(zip(zip(pow_df['woj_id'], pow_df['pow_norm']), pow_df['powiat_4']))
    
    # 4. Hierarchy: (woj_id, powiat_4) → list of gmina teryt_7digit codes
    hierarchy = {}
    for (w, p4), grp in df.groupby(['woj_id', 'powiat_4']):
        hierarchy[(w, p4)] = grp['target_id'].unique().tolist()
    
    log.info(
        f"TERYT lookup: {len(primary)} primary, {len(fallback)} fallback, "
        f"{len(powiat_map)} powiat, {len(hierarchy)} hierarchy entries"
    )
    
    return primary, fallback, powiat_map, hierarchy, norm_text


# ============================================================================
# 3. VOIVODESHIP CODE MAP
# ============================================================================

VOIV_MAP = {
    'DOLNOŚLĄSKIE': '02', 'KUJAWSKO-POMORSKIE': '04', 'LUBELSKIE': '06',
    'LUBUSKIE': '08', 'ŁÓDZKIE': '10', 'MAŁOPOLSKIE': '12',
    'MAZOWIECKIE': '14', 'OPOLSKIE': '16', 'PODKARPACKIE': '18',
    'PODLASKIE': '20', 'POMORSKIE': '22', 'ŚLĄSKIE': '24',
    'ŚWIĘTOKRZYSKIE': '26', 'WARMIŃSKO-MAZURSKIE': '28',
    'WIELKOPOLSKIE': '30', 'ZACHODNIOPOMORSKIE': '32',
}
# Manual overrides for cities with powiat status where TERYT
# powiat name is truncated and gmina name is ambiguous
CITY_POWIAT_OVERRIDES = {
        ('06', 'białapodlaska'): '0661011',
        ('08', 'zielonagóra'): '0862011',
        # Add others if needed:
        # ('14', 'warszawa'): '1465011',  # Warszawa handled by m.st. prefix
    }

# ============================================================================
# 4. COLUMN RENAME MAPS (per period)
# ============================================================================

# 2007-2013: from KSI SIMIK umowy format
RENAME_200713 = {
    "Numer umowy/aneksu/decyzji": "ID",
    "Tytuł projektu": "project_title",
    "Program Operacyjny <Nazwa>": "program",
    "Oś priorytetowa <Kod>": "priority_code",
    "Działanie <Kod>": "action_code",
    "Poddziałanie <Kod>": "subaction_code",
    "Województwo": "voviodeship",
    "Powiat": "powiat",
    "Gmina": "gmina",
    "Wartość ogółem": "total_value_PLN",
    "Wydatki kwalifikowalne": "eligible_expenses_PLN",
    "Dofinansowanie": "subsidy_PLN",
    "Dofinansowanie UE": "EU_subsidy_PLN",
    "Nazwa beneficjenta": "beneficiary",
    "NIP beneficjenta": "beneficiary_ID",
    "Kod pocztowy": "beneficiary_postal_code",
    "Miejscowość": "beneficiary_city",
    "Województwo.1": "beneficiary_voviodeship",
    "Powiat.1": "beneficiary_powiat",
    "Temat priorytetu": "priority_theme",
    "Forma prawna": "beneficiary_status",
    "Obszar realizacji": "territory_type",
    "Zakończony": "project_completed",
    "Data rozpoczęcia realizacji": "start_date",
    "Data rzeczywistego zakończenia realizacji": "end_date",
    "Data zawarcia umowy / decyzji o dofinansowaniu": "signing_date",
    "Data utworzenia w KSI SIMIK 07-13": "creation_date_KSI_SIMIK_07_12",
    "Fundusz UE": "eu_fund",
    "Miejsce realizacji projektu": "project_place",
    "Data zakończenia realizacji": "end_date",
    "Data podpisania Umowy/Aneksu": "signing_date",
    "Data utworzenia w KSI SIMIK 07-13 Umowy/Aneksu": "creation_date_KSI_SIMIK_07_12",
    "Projekt zakończony (Wniosek o płatność końcową)": "project_completed",
}

# 2014-2020: from SL2014 / dane.gov.pl
# After stripping bilingual suffixes with: columns.str.replace(r'/ .*$', '', regex=True)
RENAME_201420 = {
    "Numer umowy": "ID",
    "Tytuł projektu": "project_title",
    "Program": "program",
    "Priorytet": "priority_code",
    "Działanie": "action_code",
    "Poddziałanie": "subaction_code",
    "Fundusz": "eu_fund",
    "Miejsce realizacji projektu": "project_place",
    "Wartość projektu (w zł, dla projektów EWT w euro)": "total_value_PLN",
    "Wydatki kwalifikowalne (w zł, dla projektów EWT w euro)": "eligible_expenses_PLN",
    "Wartość unijnego dofinansowania (w zł, dla projektów EWT w euro)": "EU_subsidy_PLN",
    "Nazwa beneficjenta": "beneficiary",
    "Data rozpoczęcia realizacji projektu": "start_date",
    "Data zakończenia realizacji projektu": "end_date",
    "Finansowanie zakończone": "project_completed",
    "Typ obszaru, na którym realizowany jest projekt": "territory_type",
}

# 2021-2027: from CST2021 / dane.gov.pl
# After stripping bilingual suffixes with: columns.str.replace(r'/ .*$', '', regex=True)
RENAME_202127 = {
    "Numer umowy/decyzji": "ID",
    "Nazwa projektu": "project_title",
    "Program": "program",
    "Priorytet": "priority_code",
    "Działanie": "action_code",
    "Fundusz": "eu_fund",
    "Wartość projektu (w zł)": "total_value_PLN",
    "Dofinansowanie z UE (w zł)": "EU_subsidy_PLN",
    "Nazwa beneficjenta": "beneficiary",
    "Data rozpoczęcia projektu": "start_date",
    "Data zakończenia projektu": "end_date",
    "Miejsce realizacji projektu": "project_place",
    "Kategoria wsparcia": "intervention_type",
}


# ============================================================================
# 5. LOCATION PARSING (period-specific)
# ============================================================================

def parse_locations_200713(df, decisions, period="2007-2013"):
    """
    2007-2013 already has separate columns: voviodeship, powiat, gmina.
    Also has project_place for some records. We handle both.
    For records where voviodeship == 'Cały kraj', we exclude them (D1).
    """
    n0 = len(df)
    
    # If project_place exists and voviodeship is missing, try parsing project_place
    if 'project_place' in df.columns:
        mask_needs_parse = df['voviodeship'].isna() & df['project_place'].notna()
        if mask_needs_parse.sum() > 0:
            log.info(f"Parsing {mask_needs_parse.sum()} records from project_place column")
            parsed = df.loc[mask_needs_parse, 'project_place'].apply(_extract_woj_pow_gm)
            parsed_df = pd.DataFrame(parsed.tolist(), index=parsed.index)
            for col in ['voviodeship', 'powiat', 'gmina']:
                if col in parsed_df.columns:
                    df.loc[mask_needs_parse, col] = parsed_df[col]
    
    # Remove national-level projects (D1)
    mask_national = df['voviodeship'].astype(str).str.strip().str.lower() == 'cały kraj'
    pln_removed = df.loc[mask_national, 'EU_subsidy_PLN'].sum() if 'EU_subsidy_PLN' in df.columns else 0
    df = df[~mask_national].copy()
    
    decisions.record(
        "D1_EXCLUDE_NATIONAL", 
        f"Removed 'Cały Kraj' projects ({period})",
        n0, len(df), pln_removed, period
    )
    
    return df


def parse_locations_201420(df, decisions, period="2014-2020"):
    """
    2014-2020 has project_place with format:
    "WOJ.: MAŁOPOLSKIE, POW.: krakowski | WOJ.: ŚLĄSKIE, POW.: bielski"
    No gmina in location string — matched via TERYT from other columns or powiat.
    """
    n0 = len(df)
    
    # Remove Cały Kraj (D1)
    if 'project_place' in df.columns:
        mask_national = df['project_place'].astype(str).str.contains('Cały Kraj', na=False)
    else:
        mask_national = df['voviodeship'].astype(str).str.strip().str.lower() == 'cały kraj'
    
    pln_removed = df.loc[mask_national, 'EU_subsidy_PLN'].sum() if 'EU_subsidy_PLN' in df.columns else 0
    df = df[~mask_national].copy()
    decisions.record("D1_EXCLUDE_NATIONAL", f"Removed 'Cały Kraj' ({period})", n0, len(df), pln_removed, period)
    
    # Parse and explode locations (D2)
    n1 = len(df)
    df = _parse_and_explode_woj_pow(df)
    decisions.record("D2_EXPLODE_LOCATIONS", f"Exploded multi-location projects ({period})", n1, len(df), period=period)
    
    return df


def parse_locations_202127(df, decisions, period="2021-2027"):
    """
    2021-2027 has project_place with format:
    "WOJ.: MAZOWIECKIE, POW.: legionowski, GM.: Serock | WOJ.: ..."
    Includes gmina-level location data.
    """
    n0 = len(df)
    
    # Remove Cały Kraj (D1)
    if 'project_place' in df.columns:
        mask_national = df['project_place'].astype(str).str.contains('Cały Kraj', na=False)
    else:
        mask_national = pd.Series(False, index=df.index)
    
    pln_removed = df.loc[mask_national, 'EU_subsidy_PLN'].sum() if 'EU_subsidy_PLN' in df.columns else 0
    df = df[~mask_national].copy()
    decisions.record("D1_EXCLUDE_NATIONAL", f"Removed 'Cały Kraj' ({period})", n0, len(df), pln_removed, period)
    
    # Parse and explode locations with gmina (D2)
    n1 = len(df)
    df = _parse_and_explode_woj_pow_gm(df)
    decisions.record("D2_EXPLODE_LOCATIONS", f"Exploded multi-location projects ({period})", n1, len(df), period=period)
    
    return df


def _extract_woj_pow_gm(place_str):
    """Extract voivodeship, powiat, gmina from a single location string."""
    if pd.isna(place_str):
        return {'voviodeship': None, 'powiat': None, 'gmina': None}
    s = str(place_str)
    data = {'voviodeship': None, 'powiat': None, 'gmina': None}
    woj = re.search(r'WOJ\.:\s*([^,|]+)', s, re.IGNORECASE)
    pow_ = re.search(r'POW\.:\s*([^,|]+)', s, re.IGNORECASE)
    gm = re.search(r'GM\.:\s*([^,|]+)', s, re.IGNORECASE)
    if woj: data['voviodeship'] = woj.group(1).strip()
    if pow_: data['powiat'] = pow_.group(1).strip()
    if gm: data['gmina'] = gm.group(1).strip()
    return data


def _parse_and_explode_woj_pow(df):
    """Parse 2014-2020 format (WOJ/POW only) and explode multi-location rows."""
    df = df.copy()
    
    def extract(place_str):
        if pd.isna(place_str) or place_str == '':
            return []
        entries = str(place_str).split('|')
        result = []
        for e in entries:
            e = e.strip()
            m = re.search(r'WOJ\.:\s*(?P<w>[^,]*)(?:,\s*POW\.:\s*(?P<p>.*))?', e, re.I)
            if m:
                result.append({
                    'voviodeship': m.group('w').strip(),
                    'powiat': m.group('p').strip() if m.group('p') else None,
                })
            else:
                result.append({'voviodeship': e, 'powiat': None})
        return result
    
    df['_locs'] = df['project_place'].apply(extract)
    df['location_count'] = df['_locs'].apply(len).replace(0, 1)
    
    # Divide finances equally across locations (D2)
    fin_cols = [c for c in ['EU_subsidy_PLN', 'total_value_PLN', 'subsidy_PLN', 
                            'eligible_expenses_PLN'] if c in df.columns]
    for col in fin_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) / df['location_count']
    
    df = df.explode('_locs')
    loc_df = df['_locs'].apply(pd.Series)
    df = pd.concat([df.drop(columns=['_locs']), loc_df], axis=1)
    
    # Handle duplicate column names from concat
    if 'voviodeship' in loc_df.columns:
        # loc_df overwrites the original voviodeship column
        pass
    
    return df


def _parse_and_explode_woj_pow_gm(df):
    """Parse 2021-2027 format (WOJ/POW/GM) and explode multi-location rows."""
    df = df.copy()
    
    def extract(place_str):
        if pd.isna(place_str) or place_str == '':
            return []
        entries = str(place_str).split('|')
        result = []
        for e in entries:
            e = e.strip()
            data = {'voviodeship': None, 'powiat': None, 'gmina': None}
            woj = re.search(r'WOJ\.:\s*([^,]+)', e)
            pow_ = re.search(r'POW\.:\s*([^,]+)', e)
            gm = re.search(r'GM\.:\s*([^,]+)', e)
            if woj: data['voviodeship'] = woj.group(1).strip()
            if pow_: data['powiat'] = pow_.group(1).strip()
            if gm: data['gmina'] = gm.group(1).strip()
            result.append(data)
        return result
    
    df['_locs'] = df['project_place'].apply(extract)
    df['location_count'] = df['_locs'].apply(len).replace(0, 1)
    
    fin_cols = [c for c in ['EU_subsidy_PLN', 'total_value_PLN', 'subsidy_PLN',
                            'eligible_expenses_PLN'] if c in df.columns]
    for col in fin_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) / df['location_count']
    
    df = df.explode('_locs')
    loc_df = df['_locs'].apply(pd.Series)
    df = pd.concat([df.drop(columns=['_locs']), loc_df], axis=1)
    
    return df


# ============================================================================
# 6. GEO ID ASSIGNMENT
# ============================================================================

def assign_geo_ids(df, lookup_tuple, decisions=None, period=None):
    """
    Assigns TERYT codes to each row using 3-tier matching (D6):
      1. Full gmina match: (woj_id, powiat_norm, gmina_norm) → teryt7
      2. Fallback: (woj_id, gmina_norm) → teryt7 (only if unambiguous)
      3. Powiat only: (woj_id, powiat_norm) → powiat4 (gmina_id left None)
    """
    primary, fallback, powiat_map, hierarchy, nf = lookup_tuple
    
    n0 = len(df)
    
    def match_row(row):
        v = row.get('voviodeship', '')
        v = str(v).upper().strip() if pd.notna(v) else ''
        woj_id = VOIV_MAP.get(v)
        if not woj_id:
            return pd.Series([None, None, None])
        
        p_norm = nf(row.get('powiat', ''))
        g_norm = nf(row.get('gmina', ''))
        
        # Tier 1: Full match
        t7 = primary.get((woj_id, p_norm, g_norm))
        if not t7:
            # Tier 2: Fallback (gmina name unique in voivodeship)
            t7 = fallback.get((woj_id, g_norm))
        
        # Tier 2b: Manual override for ambiguous cities with powiat status
        if not t7:
            t7 = CITY_POWIAT_OVERRIDES.get((woj_id, g_norm))
        
        if t7:
            return pd.Series([woj_id, t7[:4], t7])
        
        # Tier 3: Powiat only
        p4 = powiat_map.get((woj_id, p_norm))
        if p4:
            return pd.Series([woj_id, p4, None])
        
        return pd.Series([woj_id, None, None])
    
    log.info(f"Assigning geo IDs to {len(df):,} rows...")
    df[['voivodeship_id', 'powiat_id', 'gmina_id']] = df.apply(match_row, axis=1)
    
    n_gmina = df['gmina_id'].notna().sum()
    n_powiat_only = (df['gmina_id'].isna() & df['powiat_id'].notna()).sum()
    n_unmatched = (df['powiat_id'].isna()).sum()
    
    log.info(
        f"Match results: gmina={n_gmina:,} ({n_gmina/n0:.1%}) | "
        f"powiat_only={n_powiat_only:,} ({n_powiat_only/n0:.1%}) | "
        f"unmatched={n_unmatched:,} ({n_unmatched/n0:.1%})"
    )
    
    if decisions:
        decisions.record(
            "D6_TERYT_MATCH", 
            f"Geo matching ({period}): gmina={n_gmina}, powiat_only={n_powiat_only}, unmatched={n_unmatched}",
            n0, n0, period=period
        )
        
    # Clean float artifacts and zero-pad IDs
    for col in ['voivodeship_id', 'powiat_id', 'gmina_id']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'\.0$', '', regex=True)
            df[col] = df[col].replace('None', np.nan).replace('nan', np.nan)
    
    # Zero-pad to correct lengths
    df.loc[df['voivodeship_id'].notna(), 'voivodeship_id'] = df.loc[df['voivodeship_id'].notna(), 'voivodeship_id'].astype(str).str.zfill(2)
    df.loc[df['powiat_id'].notna(), 'powiat_id'] = df.loc[df['powiat_id'].notna(), 'powiat_id'].astype(str).str.zfill(4)
    df.loc[df['gmina_id'].notna(), 'gmina_id'] = df.loc[df['gmina_id'].notna(), 'gmina_id'].astype(str).str.zfill(7)
    
    return df

# ============================================================================
# 7. TEMPORAL DISTRIBUTION
# ============================================================================

def distribute_over_time(df, decisions=None, period=None):
    """
    Distributes project funding across years (D3):
      - Completed projects with valid dates: proportional by quarter
      - Incomplete / missing dates: lump sum at signing date year
    """
    fin_cols = [c for c in ['EU_subsidy_PLN', 'total_value_PLN', 'subsidy_PLN',
                            'eligible_expenses_PLN'] if c in df.columns]
    df = df.loc[:, ~df.columns.duplicated(keep='first')]  # ← ADD THIS LINE
    df = df.copy()
    for c in ['start_date', 'end_date', 'signing_date']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    for c in fin_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    
    # Split into distributable vs lump-sum
    if 'project_completed' in df.columns:
        mask_completed = df['project_completed'].astype(str).str.strip().str.lower() == 'tak'
    else:
        # For 2021-2027: treat projects with end_date < now as completed
        mask_completed = df['end_date'] < pd.Timestamp.now()
    
    mask_dates = (
        df['start_date'].notna() & 
        df['end_date'].notna() & 
        (df['start_date'] <= df['end_date'])
    )
    
    df_dist = df[mask_completed & mask_dates]
    df_lump = df[~(mask_completed & mask_dates)].copy()
    
    log.info(f"Temporal distribution: {len(df_dist):,} distributable, {len(df_lump):,} lump-sum")
    
    # Distribute by quarter
    distributed_rows = []
    for row in df_dist.to_dict('records'):
        if all(row.get(c, 0) == 0 for c in fin_cols):
            continue
        try:
            periods = pd.period_range(start=row['start_date'], end=row['end_date'], freq='Q')
        except:
            periods = []
        
        n_q = len(periods)
        if n_q > 0:
            year_wt = {}
            for p in periods:
                year_wt[p.year] = year_wt.get(p.year, 0) + 1
            for yr, qc in year_wt.items():
                new_row = row.copy()
                new_row['Year'] = int(yr)
                ratio = qc / n_q
                for c in fin_cols:
                    new_row[c] = row[c] * ratio
                distributed_rows.append(new_row)
        else:
            row['Year'] = row['start_date'].year if pd.notna(row.get('start_date')) else 0
            distributed_rows.append(row)
    
    # Lump-sum: assign to signing date year
    if 'signing_date' in df_lump.columns:
        df_lump['Year'] = df_lump['signing_date'].dt.year.fillna(0).astype(int)
    elif 'start_date' in df_lump.columns:
        df_lump['Year'] = df_lump['start_date'].dt.year.fillna(0).astype(int)
    else:
        df_lump['Year'] = 0
    df_lump = df_lump[df_lump['Year'] > 0]
    
    if distributed_rows:
        df_out = pd.concat([pd.DataFrame(distributed_rows), df_lump], ignore_index=True)
    else:
        df_out = df_lump
    
    if decisions:
        decisions.record(
            "D3_TEMPORAL_DIST",
            f"Temporal distribution ({period}): {len(df_dist)} distributed + {len(df_lump)} lump-sum",
            len(df), len(df_out), period=period
        )
    
    return df_out


# ============================================================================
# 8. POWIAT DISAGGREGATION
# ============================================================================

def disaggregate_powiat(df, hierarchy, decisions=None, period=None):
    """
    For rows matched to powiat but not gmina, splits funding equally
    across all gminas in that powiat (D4).
    """
    mask_split = (df['gmina_id'].isna() | (df['gmina_id'] == '')) & df['powiat_id'].notna()
    
    df_clean = df[~mask_split].copy()
    df_dirty = df[mask_split].copy()
    
    if df_dirty.empty:
        log.info("No powiat-only rows to disaggregate")
        return df_clean
    
    log.info(f"Disaggregating {len(df_dirty):,} powiat-only rows to gminas...")
    
    df_dirty['_key'] = list(zip(df_dirty['voivodeship_id'], df_dirty['powiat_id']))
    df_dirty['_gminas'] = df_dirty['_key'].map(hierarchy)
    
    df_valid = df_dirty.dropna(subset=['_gminas']).copy()
    df_valid['_n'] = df_valid['_gminas'].apply(len)
    
    fin_cols = [c for c in ['EU_subsidy_PLN', 'total_value_PLN', 'subsidy_PLN',
                            'eligible_expenses_PLN'] if c in df.columns]
    for c in fin_cols:
        df_valid[c] = df_valid[c] / df_valid['_n']
    
    df_exploded = df_valid.explode('_gminas')
    df_exploded['gmina_id'] = df_exploded['_gminas']
    
    keep_cols = [c for c in df.columns if c not in ['_key', '_gminas', '_n']]
    result = pd.concat([df_clean, df_exploded[keep_cols]], ignore_index=True)
    
    if decisions:
        decisions.record(
            "D4_POWIAT_DISAGG",
            f"Disaggregated {len(df_dirty)} powiat-only rows ({period})",
            len(df), len(result), period=period
        )
    
    return result


# ============================================================================
# 9. AGGREGATION
# ============================================================================

def aggregate_to_panel(df, period_label):
    """Aggregate to gmina × year panel for one programming period."""
    group_cols = ['voivodeship_id', 'powiat_id', 'gmina_id', 'Year']
    fin_cols = [c for c in ['EU_subsidy_PLN', 'total_value_PLN', 'subsidy_PLN',
                            'eligible_expenses_PLN'] if c in df.columns]
    
    df_clean = df.dropna(subset=['gmina_id']).copy()
    # Fix float→string artifacts (e.g. '1001022.0' → '1001022')
    df_clean['gmina_id'] = df_clean['gmina_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    df_clean['powiat_id'] = df_clean['powiat_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    df_clean['voivodeship_id'] = df_clean['voivodeship_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    
    df_clean['Year'] = pd.to_numeric(df_clean['Year'], errors='coerce').fillna(0).astype(int)
    df_clean = df_clean[df_clean['Year'] > 0]
    
    panel = df_clean.groupby(group_cols, as_index=False)[fin_cols].sum()
    panel['programming_period'] = period_label
    
    log.info(
        f"Panel ({period_label}): {len(panel):,} rows, "
        f"{panel['gmina_id'].nunique():,} gminas, "
        f"years {panel['Year'].min()}-{panel['Year'].max()}"
    )
    
    return panel.sort_values(['gmina_id', 'Year'])


# ============================================================================
# 10. MAIN PIPELINE CLASS
# ============================================================================

class EUFundsPipeline:
    """
    End-to-end pipeline for processing EU cohesion fund project lists
    into a gmina × year panel dataset.
    """
    
    def __init__(self, config):
        """
        config: dict with keys:
          - PATH_TERYT: path to teryt_klucz Excel
          - PATH_200713_POKL: path to 2007-2013 POKL Excel
          - PATH_200713_INNE: path to 2007-2013 INNE Excel
          - PATH_201420: path to 2014-2020 project list Excel
          - PATH_202127: path to 2021-2027 project list Excel
          - OUTPUT_DIR: directory for outputs
          - PATH_POPULATION: (optional) path to population CSV for per-capita
        """
        self.config = config
        self.decisions = DecisionLog()
        self.lookup = None
        self.weight_table = None  
        self.panels = {}
    
    def build_lookup(self):
        self.lookup = build_teryt_lookup(self.config['PATH_TERYT'])
        # Load revenue weights if using weighted disaggregation
        wt_path = self.config.get('PATH_WEIGHTS')
        if wt_path and os.path.exists(wt_path):
            self.weight_table = pd.read_csv(wt_path)
            log.info(f"Loaded weight table: {len(self.weight_table):,} rows")
    
    def _disaggregate(self, df, period):
        """Dispatch to weighted or equal disaggregation based on config."""
        method = self.config.get('DISAGG_METHOD', 'equal')
        _, _, _, hierarchy, _ = self.lookup
        
        if method == 'weighted' and self.weight_table is not None:
            log.info(f"Using WEIGHTED disaggregation for {period}")
            from disaggregate_weighted import disaggregate_powiat_weighted
            return disaggregate_powiat_weighted(df, self.weight_table, self.decisions, period)
        else:
            if method == 'weighted':
                log.warning(f"Weighted disaggregation requested but no weight table loaded — falling back to equal split")
            log.info(f"Using EQUAL-SPLIT disaggregation for {period}")
            return disaggregate_powiat(df, hierarchy, self.decisions, period)
    
    def process_200713(self):
        """Process 2007-2013 programming period."""
        log.info("=" * 60)
        log.info("PROCESSING 2007-2013")
        log.info("=" * 60)
        
        # Read both POKL and INNE files, rename EACH before concat
        dfs = []
        for key in ['PATH_200713_POKL', 'PATH_200713_INNE']:
            path = self.config.get(key)
            if path and os.path.exists(path):
                log.info(f"Reading: {path}")
                sheets = pd.read_excel(path, sheet_name=None, header=1)
                for name, sheet_df in sheets.items():
                    # Fix known column name variants BEFORE rename
                    sheet_df.columns = sheet_df.columns.str.strip()
                    if 'Cały kraj/ Województwo' in sheet_df.columns and 'Województwo' not in sheet_df.columns:
                        sheet_df.rename(columns={'Cały kraj/ Województwo': 'Województwo'}, inplace=True)
                    # Rename to common schema
                    sheet_df.rename(columns=RENAME_200713, inplace=True)
                    # Drop duplicate columns within this sheet
                    sheet_df = sheet_df.loc[:, ~sheet_df.columns.duplicated(keep='first')]
                    dfs.append(sheet_df)
        
        if not dfs:
            log.warning("No 2007-2013 files found, skipping")
            return
        
        df = pd.concat(dfs, ignore_index=True)
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
        log.info(f"Raw rows: {len(df):,}")
        
        # Rename
        df.rename(columns=RENAME_200713, inplace=True)
        df = df.loc[:, ~df.columns.duplicated(keep='first')]  # ← ADD THIS
        
        # Parse locations & filter
        df = parse_locations_200713(df, self.decisions, "2007-2013")
        
        # Assign geo IDs
        df = assign_geo_ids(df, self.lookup, self.decisions, "2007-2013")
        # DEBUG: check what date columns actually exist
        print("Columns with 'dat' or 'date':", [c for c in df.columns if 'dat' in c.lower() or 'zakończ' in c.lower() or 'rozpocz' in c.lower()])
        # Distribute over time
        df = distribute_over_time(df, self.decisions, "2007-2013")
        
        # Disaggregate powiat
        # _, _, _, hierarchy, _ = self.lookup
        # df = disaggregate_powiat(df, hierarchy, self.decisions, "2007-2013")
        # Disaggregate powiat (method set in config: 'weighted' or 'equal')
        df = self._disaggregate(df, "2007-2013")     
        # DEBUG: check if 0201011 survived disaggregation
        check = df[df['gmina_id'].astype(str).str.zfill(7) == '0201011']
        print(f"DEBUG: 0201011 rows after disagg: {len(check)}")
        
        # DEBUG: check gmina_id formats
        direct = df[df['gmina_id'].notna()]['gmina_id'].astype(str)
        print(f"Sample gmina_ids: {direct.unique()[:10]}")
        print(f"ID lengths: {direct.str.len().value_counts().to_dict()}")
        print(f"Unique gmina_ids: {direct.nunique()}")
        
        # Aggregate
        panel = aggregate_to_panel(df, "2007-2013")
        self.panels["2007-2013"] = panel
        
        return panel
    
    def process_201420(self):
        """Process 2014-2020 programming period."""
        log.info("=" * 60)
        log.info("PROCESSING 2014-2020")
        log.info("=" * 60)
        
        path = self.config.get('PATH_201420')
        if not path or not os.path.exists(path):
            log.warning("2014-2020 file not found, skipping")
            return
        
        df = pd.read_excel(path, header=2)
        # Strip bilingual column suffixes
        df.columns = df.columns.str.replace(r'/ .*$', '', regex=True)
        log.info(f"Raw rows: {len(df):,}")
        
        df.columns = df.columns.str.replace(r'/ .*$', '', regex=True)
        # Aggressive column cleanup: normalize all whitespace, then strip bilingual suffixes
        df.columns = df.columns.str.replace('\xa0', ' ')      # non-breaking → regular space
        df.columns = df.columns.str.replace(r'\s*/\s*.*$', '', regex=True)  # strip "/ English name"
        df.columns = df.columns.str.strip()
        print("Columns after cleanup:", df.columns.tolist())   # DEBUG — remove once working
        
        # Rename
        df.rename(columns=RENAME_201420, inplace=True)
        print("Columns after rename:", [c for c in df.columns if 'miejsc' in c.lower() or 'woj' in c.lower() or 'place' in c.lower() or 'powiat' in c.lower() or 'gmina' in c.lower()])
        
        # Filter funds (D5)
        n0 = len(df)
        if 'eu_fund' in df.columns:
            df = df[df['eu_fund'] != 'BAR']
        self.decisions.record("D5_FUND_FILTER", "Removed BAR fund (2014-2020)", n0, len(df), period="2014-2020")
        
        # Parse locations & filter
        df = parse_locations_201420(df, self.decisions, "2014-2020")
        
        # Assign geo IDs
        df = assign_geo_ids(df, self.lookup, self.decisions, "2014-2020")
        
        # Distribute over time
        df = distribute_over_time(df, self.decisions, "2014-2020")
        df = df[df['Year'] <= 2025]
        # Disaggregate powiat
        # _, _, _, hierarchy, _ = self.lookup
        # df = disaggregate_powiat(df, hierarchy, self.decisions, "2014-2020")
        df = self._disaggregate(df, "2014-2020")
        
        # Aggregate
        panel = aggregate_to_panel(df, "2014-2020")
        self.panels["2014-2020"] = panel
        
        return panel
    
    def process_202127(self):
        """Process 2021-2027 programming period."""
        log.info("=" * 60)
        log.info("PROCESSING 2021-2027")
        log.info("=" * 60)
        
        path = self.config.get('PATH_202127')
        if not path or not os.path.exists(path):
            log.warning("2021-2027 file not found, skipping")
            return
        
        df = pd.read_excel(path, header=1)
        # Strip bilingual column suffixes
        df.columns = df.columns.str.replace('\xa0', ' ')
        df.columns = df.columns.str.replace(r'\s*/\s*.*$', '', regex=True)
        df.columns = df.columns.str.strip()
        print("Columns after cleanup:", df.columns.tolist())  
        log.info(f"Raw rows: {len(df):,}")
        
        # Rename
        df.rename(columns=RENAME_202127, inplace=True)
        
        # Filter funds (D5)
        n0 = len(df)
        if 'eu_fund' in df.columns:
            keep_funds = ['EFRR', 'EFS+', 'FST', 'FS']
            df = df[df['eu_fund'].isin(keep_funds)]
        self.decisions.record("D5_FUND_FILTER", f"Kept {keep_funds} only (2021-2027)", n0, len(df), period="2021-2027")
        
        # Cap to data available (exclude future)
        n1 = len(df)
        df['end_date'] = pd.to_datetime(df.get('end_date'), errors='coerce')
        # For 2021-2027, we keep projects that have started, not future-only
        
        # Parse locations & filter
        df = parse_locations_202127(df, self.decisions, "2021-2027")
        
        # Assign geo IDs
        df = assign_geo_ids(df, self.lookup, self.decisions, "2021-2027")
        
        # Mark completed (projects ending before now)
        if 'project_completed' not in df.columns:
            df['project_completed'] = np.where(
                df['end_date'] < pd.Timestamp.now(), 'Tak', 'Nie'
            )
        
        # Distribute over time
        df = distribute_over_time(df, self.decisions, "2021-2027")
        
        # Cap at current year
        current_year = datetime.now().year
        n2 = len(df)
        df = df[df['Year'] <= current_year]
        self.decisions.record("CAP_YEAR", f"Capped at year {current_year} (2021-2027)", n2, len(df), period="2021-2027")
        
        # Disaggregate powiat
        # _, _, _, hierarchy, _ = self.lookup
        # df = disaggregate_powiat(df, hierarchy, self.decisions, "2021-2027")
        # Disaggregate powiat (method set in config: 'weighted' or 'equal')
        df = self._disaggregate(df, "2021-2027")
        
        # Aggregate
        panel = aggregate_to_panel(df, "2021-2027")
        self.panels["2021-2027"] = panel
        
        return panel
    
    def combine_panels(self):
        """
        Combine all period panels into one master panel (D7).
        For overlapping years, sums funding from both periods.
        """
        log.info("=" * 60)
        log.info("COMBINING ALL PERIODS")
        log.info("=" * 60)
        
        all_panels = [p for p in self.panels.values() if p is not None]
        if not all_panels:
            log.error("No panels to combine")
            return None
        
        # Concat keeping programming_period
        combined = pd.concat(all_panels, ignore_index=True)
        
        # Also create a version summed across periods (for total per gmina-year)
        fin_cols = [c for c in ['EU_subsidy_PLN', 'total_value_PLN', 'subsidy_PLN',
                                'eligible_expenses_PLN'] if c in combined.columns]
        
        master = combined.groupby(
            ['voivodeship_id', 'powiat_id', 'gmina_id', 'Year'],
            as_index=False
        )[fin_cols].sum()
        
        master = master.sort_values(['gmina_id', 'Year'])
        
        log.info(
            f"Master panel: {len(master):,} rows, "
            f"{master['gmina_id'].nunique():,} gminas, "
            f"years {master['Year'].min()}-{master['Year'].max()}"
        )
        
        return master, combined
    
    def add_population(self, master, pop_path=None):
        import re
        """
        Merge population data and compute per-capita funding.
        Expects pop_path to be a CSV with columns: unit_id, year, population_total
        (from the BDL download).
        """
        pop_path = pop_path or self.config.get('PATH_POPULATION')
        if not pop_path or not os.path.exists(pop_path):
            log.warning("No population file found — skipping per-capita calculation")
            return master
        
        log.info(f"Merging population from: {pop_path}")
        pop = pd.read_csv(pop_path)
        
        # Match on gmina_id (7-digit TERYT → first 7 of unit_id/12-digit BDL code)
        # BDL unit_id format: 12 digits; gmina_id in panel: 7 digits
        # BDL and TERYT use different ID systems — match by name + voivodeship
        pop['bdl_woj'] = pop['unit_id'].astype(str).str.zfill(12).str[2:4]
        pop['name_norm'] = pop['unit_name'].apply(lambda s: re.sub(r'\s+', ' ', str(s).lower().strip()) if pd.notna(s) else '')
        
        # Load TERYT mapping for name → gmina_id bridge
        teryt_path = self.config.get('PATH_TERYT')
        teryt = pd.read_excel(teryt_path, sheet_name='gminy', dtype=str)
        teryt['teryt_7'] = teryt['teryt_2025'].astype(str).str.split('.').str[0].str.zfill(7)
        teryt['woj_2'] = teryt['teryt_7'].str[:2]
        teryt['name_norm'] = teryt['nazwa_gminy'].apply(
            lambda s: re.sub(r'\s*\(.*?\)', '', re.sub(r'\s+od\s+\d{4}', '', str(s).lower().strip())) if pd.notna(s) else ''
        )
        
        # Build bridge: name + woj → teryt_7
        bridge = teryt[['teryt_7', 'woj_2', 'name_norm']].drop_duplicates('teryt_7')
        
        pop = pop.merge(bridge, left_on=['name_norm', 'bdl_woj'], right_on=['name_norm', 'woj_2'], how='left')
        pop['gmina_id'] = pop['teryt_7']
        
        pop_slim = pop[['gmina_id', 'year', 'population_total']].copy()
        pop_slim.rename(columns={'year': 'Year'}, inplace=True)
        pop_slim['Year'] = pd.to_numeric(pop_slim['Year'], errors='coerce').astype(int)
        
        master = master.merge(pop_slim, on=['gmina_id', 'Year'], how='left')
        
        # Compute per-capita
        for col in ['EU_subsidy_PLN', 'total_value_PLN']:
            if col in master.columns:
                pc_col = col.replace('_PLN', '_per_capita_PLN')
                master[pc_col] = master[col] / master['population_total']
                master[pc_col] = master[pc_col].replace([np.inf, -np.inf], np.nan)
        
        pop_matched = master['population_total'].notna().sum()
        log.info(f"Population matched: {pop_matched:,}/{len(master):,} rows ({pop_matched/len(master):.1%})")
        
        return master
    
    def run_all(self):
        """Execute the full pipeline."""
        log.info("=" * 60)
        log.info("EU FUNDS PIPELINE — FULL RUN")
        log.info(f"Started: {datetime.now().isoformat()}")
        log.info("=" * 60)
        
        # Step 0: Build TERYT lookup
        self.build_lookup()
        
        # Steps 1-9: Process each period
        self.process_200713()
        self.process_201420()
        self.process_202127()
        
        # Step 10: Combine
        result = self.combine_panels()
        if result is None:
            return None
        
        master, combined_with_periods = result
        
        # Optional: Add population
        master = self.add_population(master)
        
        # Save outputs
        out_dir = self.config.get('OUTPUT_DIR', './eu_funds_output')
        os.makedirs(out_dir, exist_ok=True)
        
        master.to_csv(os.path.join(out_dir, 'eu_funds_gmina_panel_master.csv'),
                      index=False, encoding='utf-8-sig')
        combined_with_periods.to_csv(
            os.path.join(out_dir, 'eu_funds_gmina_panel_by_period.csv'),
            index=False, encoding='utf-8-sig')
        self.decisions.save(os.path.join(out_dir, 'pipeline_decision_log.csv'))
        
        log.info(f"\nOutputs saved to: {out_dir}")
        log.info(f"  eu_funds_gmina_panel_master.csv — {len(master):,} rows")
        log.info(f"  eu_funds_gmina_panel_by_period.csv — {len(combined_with_periods):,} rows")
        log.info(f"  pipeline_decision_log.csv — {len(self.decisions.entries)} decisions")
        log.info(f"\nCompleted: {datetime.now().isoformat()}")
        
        return master


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Example config — UPDATE THESE PATHS on your machine
    config = {
        'PATH_TERYT': r"data\inputs\shapefiles\polska\teryt_klucz_powiaty_gminy_lata_1999_2025-1.xlsx",
        'PATH_200713_POKL': r"data\inputs\3. treatment\_target_excels\Umowy_wszystko_POKL_30_06_2018.xls",
        'PATH_200713_INNE': r"data\inputs\3. treatment\_target_excels\Umowy_wszystko_INNE_30_06_2018.xls",
        'PATH_201420': r"data\inputs\3. treatment\_target_excels\Lista_projektow_FE_2014_2020_04012026.xlsx",
        'PATH_202127': r"data\inputs\3. treatment\_target_excels\Lista_projektow_FE_2021_2027_01022026.xlsx",
        'OUTPUT_DIR': r"data\clean\treatment\eu_flows\final",
        'PATH_POPULATION': None,  # Set to BDL population CSV path when available
    }
    
    pipeline = EUFundsPipeline(config)
    master = pipeline.run_all()
