# -*- coding: utf-8 -*-
import pandas as pd 
import numpy as np
import re
import os
import pickle
import glob

# ==========================================
# 1. HELPER FUNCTIONS (GLOBAL)
# ==========================================

def norm_text(s):
    """
    Standardizes Polish administrative names for matching.
    Removes prefixes like 'm.st.', 'powiat', and time markers.
    """
    if pd.isna(s): return ""
    s = str(s).lower()
    # Remove " od YYYY" pattern (common in TERYT)
    s = re.sub(r'\s+od\s+\d{4}', '', s)
    
    # Remove administrative prefixes/suffixes and punctuation
    for pat in ['m.st.', 'm.', 'st.', 'miasto', 'powiat', '-', '.', ' ']:
        s = s.replace(pat, '')
    return s

def is_sme(form_string):
    form_string = str(form_string).lower()
    return any(term in form_string for term in ['mikro', 'małe', 'średnie'])

# ==========================================
# 2. DATA READING & CLEANING
# ==========================================

def read_and_parse(file_name, file_path):
    full_path = os.path.join(file_path, f"{file_name}.csv")
    
    # Note: 'voviodeship' contains a typo (voivodeship) but we keep it for consistency with your pipeline
    dtype_map = {
        "ID": str, "program": str, "priority_code": str, 
        "action_code": str, "subaction_code": str, 
        "voviodeship": str, "powiat": str, "gmina": str, 
        "beneficiary_ID": str, "beneficiary_postal_code": str,
        "project_completed": str
    }
    
    date_cols = ['start_date', 'end_date', 'signing_date', 'creation_date_KSI_SIMIK_07_12']
    
    try:
        df = pd.read_csv(full_path, dtype=dtype_map, parse_dates=date_cols, low_memory=False)
    except FileNotFoundError:
        print(f"File not found: {full_path}")
        return None
        
    return df

def parse_and_explode_locations(df):
    """
    Parses 'project_place' column with format "WOJ.: X | WOJ.: Y, POW.: Z".
    Splits rows and divides financial values.
    """
    df = df.copy()
    
    def extract_locs(place_str):
        if pd.isna(place_str) or place_str == '':
            return []
        
        if 'Cały Kraj' in str(place_str):
            return [{'voivodeship_raw': 'Cały Kraj', 'powiat_raw': None}]

        entries = str(place_str).split('|')
        parsed_entries = []
        
        for entry in entries:
            entry = entry.strip()
            match = re.search(r'WOJ\.:\s*(?P<woj>[^,]*)(?:,\s*POW\.:\s*(?P<pow>.*))?', entry, re.IGNORECASE)
            
            if match:
                w = match.group('woj').strip()
                p = match.group('pow')
                if p: p = p.strip()
                parsed_entries.append({'voivodeship_raw': w, 'powiat_raw': p})
            else:
                parsed_entries.append({'voivodeship_raw': entry, 'powiat_raw': None})
                
        return parsed_entries

    df['parsed_locs'] = df['project_place'].apply(extract_locs)
    df['location_count'] = df['parsed_locs'].apply(len).replace(0, 1)
    
    fin_cols = [c for c in ['total_value_PLN', 'eu_fund', 'total_value', 'dofinansowanie_ue'] if c in df.columns]
    for col in fin_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df[col] = df[col] / df['location_count']

    df_exploded = df.explode('parsed_locs')
    loc_data = df_exploded['parsed_locs'].apply(pd.Series)
    
    df_final = pd.concat([df_exploded.drop(['parsed_locs'], axis=1), loc_data], axis=1)
    
    # Renames extracted columns to match assign_geo_ids expectations
    return df_final.rename(columns={'voivodeship_raw': 'voviodeship', 'powiat_raw': 'powiat'})

# ==========================================
# 3. TERYT BUILDER (FIXED)
# ==========================================

def build_teryt_lookup_optimized(path_excel):
    print(f"Loading TERYT data from: {path_excel}...")
    
    # Load data
    cols_to_load = ['region', 'nazwa_powiatu', 'nazwa_gminy', 'teryt_2025', 'zmiana_opis']
    try:
        df_g = pd.read_excel(path_excel, sheet_name='gminy', dtype=str, usecols=cols_to_load)
    except ValueError:
        df_g = pd.read_excel(path_excel, sheet_name='gminy', dtype=str)

    # Używamy globalnej funkcji norm_text (zdefiniowanej na górze pliku), a nie lokalnej!
    woj_id = df_g['region'].astype(str).str.split('.').str[0].str.zfill(2)
    pow_norm = df_g['nazwa_powiatu'].apply(norm_text)
    gmi_norm = df_g['nazwa_gminy'].apply(norm_text)
    target_id = df_g['teryt_2025'].astype(str).str.split('.').str[0].str.zfill(7)

    # 1. Primary Map
    primary_lookup = dict(zip(zip(woj_id, pow_norm, gmi_norm), target_id))

    # 2. Fallback Map
    df_g['woj_id'] = woj_id
    df_g['gmi_norm'] = gmi_norm
    df_g['target_id'] = target_id
    unique_check = df_g.groupby(['woj_id', 'gmi_norm'])['target_id'].nunique()
    valid_fallbacks = unique_check[unique_check == 1].index
    fallback_df = df_g.set_index(['woj_id', 'gmi_norm']).loc[valid_fallbacks]
    fallback_lookup = fallback_df['target_id'].to_dict()
    
    # 3. Powiat Map
    df_g['powiat_id_4'] = df_g['target_id'].str[:4]
    pow_df = df_g[['woj_id', 'nazwa_powiatu', 'powiat_id_4']].drop_duplicates()
    pow_df['pow_norm'] = pow_df['nazwa_powiatu'].apply(norm_text)
    
    powiat_lookup = dict(zip(zip(pow_df['woj_id'], pow_df['pow_norm']), pow_df['powiat_id_4']))
    
    print("TERYT Lookup built successfully.")
    
    # --- KLUCZOWA ZMIANA ---
    # Zwracamy 'norm_text' (funkcja globalna), a nie 'simple_norm' (lokalna)
    return primary_lookup, fallback_lookup, powiat_lookup, norm_text

# ==========================================
# 3. TEMPORAL DISTRIBUTION (RESTORED!)
# ==========================================

def distribute_funding_over_time(df, cols_to_distribute=None):
    """
    Splits project funding across years based on start/end dates.
    """
    if cols_to_distribute is None:
        cols_to_distribute = ['EU_subsidy_PLN', 'total_value_PLN', 'subsidy_PLN', 'eligible_expenses_PLN', 'eu_fund', 'total_value']

    df = df.copy()
    
    # Ensure dates
    for c in ['start_date', 'end_date', 'signing_date']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')

    # Ensure numerics
    existing_cols = [c for c in cols_to_distribute if c in df.columns]
    for col in existing_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Filter
    mask_completed = df['project_completed'].astype(str).str.strip().str.lower() == 'tak'
    mask_dates = (df['start_date'].notna()) & (df['end_date'].notna()) & (df['start_date'] <= df['end_date'])
    
    df_dist = df[mask_completed & mask_dates]
    df_lump = df[~(mask_completed & mask_dates)].copy()

    distributed_rows = []
    records = df_dist.to_dict('records')
    
    print(f"Distributing funding over time for {len(records)} rows...")
    
    for row in records:
        if all(row[c] == 0 for c in existing_cols):
            continue

        try:
            periods = pd.period_range(start=row['start_date'], end=row['end_date'], freq='Q')
        except:
            periods = []
            
        n_quarters = len(periods)

        if n_quarters > 0:
            year_weights = {}
            for p in periods:
                year_weights[p.year] = year_weights.get(p.year, 0) + 1
            
            for year, q_count in year_weights.items():
                new_row = row.copy()
                new_row['Year'] = int(year)
                ratio = q_count / n_quarters
                for col in existing_cols:
                    new_row[col] = row[col] * ratio
                distributed_rows.append(new_row)
        else:
            row['Year'] = row['start_date'].year
            distributed_rows.append(row)

    # Handle Lump Sum
    if 'signing_date' in df_lump.columns:
        df_lump['Year'] = df_lump['signing_date'].dt.year.fillna(0).astype(int)
        df_lump = df_lump[df_lump['Year'] != 0]

    if distributed_rows:
        df_distributed = pd.DataFrame(distributed_rows)
        return pd.concat([df_distributed, df_lump], ignore_index=True)
    
    return df_lump


# ==========================================
# 4. ASSIGNER & AGGREGATOR
# ==========================================

def assign_geo_ids(df, lookup_tuple):
    primary_map, fallback_map, powiat_map, norm_func = lookup_tuple
    
    voiv_map = {
        'DOLNOŚLĄSKIE': '02', 'KUJAWSKO-POMORSKIE': '04', 'LUBELSKIE': '06', 
        'LUBUSKIE': '08', 'ŁÓDZKIE': '10', 'MAŁOPOLSKIE': '12', 
        'MAZOWIECKIE': '14', 'OPOLSKIE': '16', 'PODKARPACKIE': '18', 
        'PODLASKIE': '20', 'POMORSKIE': '22', 'ŚLĄSKIE': '24', 
        'ŚWIĘTOKRZYSKIE': '26', 'WARMIŃSKO-MAZURSKIE': '28', 
        'WIELKOPOLSKIE': '30', 'ZACHODNIOPOMORSKIE': '32'
    }

    def get_ids_row(row):
        # Look for 'voviodeship' (with typo) OR 'voviodeship_raw'
        v_name = row.get('voviodeship')
        if pd.isna(v_name):
            # Try alternative column names if user skipped renaming
            v_name = row.get('Województwo') or row.get('voivodeship')
        
        v_name = str(v_name).upper() if v_name else ''
        
        woj_id = voiv_map.get(v_name)
        if not woj_id: return pd.Series([None, None, None, None])

        p_raw = row.get('powiat', '')
        g_raw = row.get('gmina', '')
        
        p_norm = norm_func(p_raw)
        g_norm = norm_func(g_raw)
        
        # Strategy 1: Full Gmina Match
        teryt7 = primary_map.get((woj_id, p_norm, g_norm))
        if not teryt7:
            teryt7 = fallback_map.get((woj_id, g_norm))
            
        if teryt7:
            return pd.Series([woj_id, teryt7[:4], teryt7, teryt7])
        
        # Strategy 2: Powiat Only Match
        powiat_id = powiat_map.get((woj_id, p_norm))
        if powiat_id:
            return pd.Series([woj_id, powiat_id, None, None])
            
        return pd.Series([woj_id, None, None, None])

    df[['voivodeship_id', 'powiat_id', 'gmina_id', 'city_id']] = df.apply(get_ids_row, axis=1)
    return df

def disaggregate_powiat_funding(df, path_to_pickle):
    with open(path_to_pickle, 'rb') as f:
        powiat_structure = pickle.load(f)
        
    mask_split = (df['gmina_id'].isna() | (df['gmina_id'] == '')) & (df['powiat_id'].notna())
    
    df_clean = df[~mask_split].copy()
    df_dirty = df[mask_split].copy()
    
    if df_dirty.empty:
        return df_clean

    print(f"Disaggregating {len(df_dirty)} rows...")
    df_dirty['lookup_key'] = list(zip(df_dirty['voivodeship_id'], df_dirty['powiat_id']))
    df_dirty['target_gminas'] = df_dirty['lookup_key'].map(powiat_structure)
    
    df_valid = df_dirty.dropna(subset=['target_gminas']).copy()
    df_valid['divisor'] = df_valid['target_gminas'].apply(len)
    
    fin_cols = [c for c in ['EU_subsidy_PLN', 'total_value_PLN', 'subsidy_PLN', 'eligible_expenses_PLN', 'eu_fund', 'total_value'] if c in df.columns]
    
    for col in fin_cols:
        df_valid[col] = df_valid[col] / df_valid['divisor']
        
    df_exploded = df_valid.explode('target_gminas')
    df_exploded['gmina_id'] = df_exploded['target_gminas']
    df_exploded['city_id'] = df_exploded['target_gminas']
    
    cols_to_keep = df.columns.tolist()
    cols_to_keep = [c for c in cols_to_keep if c not in ['divisor', 'target_gminas', 'lookup_key']]
    
    return pd.concat([df_clean, df_exploded[cols_to_keep]], ignore_index=True)

def aggregate_funding_by_gmina(df):
    group_cols = ['voivodeship_id', 'powiat_id', 'gmina_id', 'Year']
    fin_cols = ['EU_subsidy_PLN', 'total_value_PLN', 'subsidy_PLN', 'eligible_expenses_PLN', 'eu_fund', 'total_value']
    existing_fin = [c for c in fin_cols if c in df.columns]
    
    agg_dict = {col: 'sum' for col in existing_fin}
    df_clean = df.dropna(subset=['gmina_id'])
    
    df_panel = df_clean.groupby(group_cols, as_index=False).agg(agg_dict)
    return df_panel.sort_values(by=['gmina_id', 'Year'])





# --- CONFIGURATION ---
# Path to the folder where your clean CSVs are saved

def get_period_from_filename(filename):
    """
    Extracts the programming period based on specific substrings in the filename.
    """
    fname = os.path.basename(filename)
    
    if "200713" in fname:
        return "2007-2013"
    elif "20142020" in fname:
        return "2014-2020"
    elif "202127" in fname:
        return "2021-2027"
    else:
        return "Unknown"

def knit_directory_with_periods(folder_path):
    """
    Scans directory for 'powiat'/'gmina' files, adds a 'programming_period' column 
    based on the filename, and combines them.
    """
    
    # 1. Find files
    powiat_paths = glob.glob(os.path.join(folder_path, "*powiat*.csv"))
    gmina_paths  = glob.glob(os.path.join(folder_path, "*gmina*.csv"))
    
    print(f"Found {len(powiat_paths)} Powiat files.")
    print(f"Found {len(gmina_paths)} Gmina files.")
    
    # Helper to load, tag, and concat
    def load_tag_concat(file_list, level_name):
        if not file_list:
            print(f"No files found for {level_name}. Skipping.")
            return None
            
        dfs = []
        for f in file_list:
            print(f"  Loading: {os.path.basename(f)}...")
            df = pd.read_csv(f, low_memory= False)
            
            # A. Add Programming Period Column
            period = get_period_from_filename(f)
            df['programming_period'] = period
            
            # B. Ensure Year is numeric
            if 'Year' in df.columns:
                df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)
            
            dfs.append(df)
            
        full_df = pd.concat(dfs, ignore_index=True)
        
        # C. Aggregation Logic
        # We now group by ID + Year + Programming_Period to keep periods distinct 
        # (even if years overlap, e.g. 2015 might appear in both periods' closure/start)
        
        if level_name == 'powiat':
            group_cols = ['voivodeship_id', 'powiat_id', 'Year', 'programming_period']
        else:
            group_cols = ['voivodeship_id', 'powiat_id', 'gmina_id', 'Year', 'programming_period']
            
        # Sum financial columns
        fin_cols = ['EU_subsidy_PLN', 'total_value_PLN', 'subsidy_PLN', 'eligible_expenses_PLN']
        existing_fin = [c for c in fin_cols if c in full_df.columns]
        
        # Group and sum
        full_df = full_df.groupby(group_cols, as_index=False)[existing_fin].sum()
        
        return full_df.sort_values(by=['Year', 'programming_period'])

    # 2. Process Datasets
    print("\n--- Processing Powiat Datasets ---")
    df_final_powiat = load_tag_concat(powiat_paths, 'powiat')
    
    print("\n--- Processing Gmina Datasets ---")
    df_final_gmina = load_tag_concat(gmina_paths, 'gmina')
    
    return df_final_powiat, df_final_gmina
