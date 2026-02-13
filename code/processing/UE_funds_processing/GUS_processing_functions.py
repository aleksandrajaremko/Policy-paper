# infer place from beneficiary keywords 

# Logic of the processing here is:
# IF Implementation_Gmina is "Cały kraj" AND (Legal_Form in safe_local_tier OR is_sme(Legal_Form)):
#    THEN Final_Gmina = Beneficiary_City
# ELSE:
#    KEEP "Systemic/National"
# -*- coding: utf-8 -*-
import pandas as pd 
import numpy as np
import re
import geopandas as gpd                        
import os
from functools import lru_cache
import pickle


############################################
############################################
#       HELPER FUNCTIONS AND OBJECTS
############################################
############################################

safe_gmina_forms = [
    'stowarzyszenie', 'fundacja', 'gminna samorządowa jednostka organizacyjna', 
    'powiatowa samorządowa jednostka organizacyjna', 'wspólnota samorządowa - gmina', 
    'wspólnota samorządowa - powiat', 'szkoła lub placówka oświatowa', 
    'publiczny zakład opieki zdrowotnej', 'Kościół Katolicki', 'jednostka naukowa',
    'Stowarzyszenia i organizacje społeczne', 'związek zawodowy',  'organizacja społeczna oddzielnie nie wymieniona',
]

# Function to catch all SMEs
def is_sme(form_string):
    form_string = str(form_string).lower()
    return any(term in form_string for term in ['mikro', 'małe', 'średnie'])


date_columns = ['start_date', 'end_date', 'signing_date', 'creation_date_KSI_SIMIK_07_12']

rename_dict = {
                "Numer umowy/aneksu/decyzji" : "ID",
                "Tytuł projektu" : "project_title",
                "Program Operacyjny <Nazwa>" : 'program',
                "Oś priorytetowa <Kod>" : 'priority_code',
                "Działanie <Kod>" : "action_code",
                "Poddziałanie <Kod>" : "subaction_code",
                "Województwo" : "voviodeship",
                "Powiat" : "powiat",
                "Gmina" : "gmina",
                "Wartość ogółem" : "total_value_PLN",
                "Wydatki kwalifikowalne" : "eligible_expenses_PLN",
                "Dofinansowanie" : "subsidy_PLN",
                "Dofinansowanie UE" : "EU_subsidy_PLN",
                "Nazwa beneficjenta" : "beneficiary",
                "NIP beneficjenta" : "beneficiary_ID",
                "Kod pocztowy" : "beneficiary_postal_code",
                "Miejscowość" : "beneficiary_city",
                "Województwo.1" : "beneficiary_voviodeship",
                "Powiat.1" : "beneficiary_powiat",
                "Temat priorytetu" : "priority_theme",
                "Forma prawna" : "beneficiary_status",
                "Obszar realizacji" : "terriority_type",
                "Projekt zakończony (Wniosek o płatność końcową)" : "project_completed",
                "Data podpisania Umowy/Aneksu" : "signing_date",
                "Data utworzenia w KSI SIMIK 07-13 Umowy/Aneksu" : "creation_date_KSI_SIMIK_07_12",
                "Data rozpoczęcia realizacji" : "start_date",
                "Data zakończenia realizacji" : "end_date"
}

# Define consistent dtypes
data_types = {
        "ID": str,
        "program": str,
        "priority_code": str,
        "action_code": str,
        "subaction_code": str,
        "voviodeship": str,
        "powiat": str,
        "gmina": str,
        "beneficiary_ID": str,
        "beneficiary_postal_code": str,
        "beneficiary_city": str,
        "beneficiary_voviodeship": str,
        "beneficiary_powiat": str,
        "priority_theme": str,
        "beneficiary_status": str,
        "terriority_type": str,
        "project_completed": str
}
############################################
############################################
# --- 0. READ AND PARSE DATETIME CSV FUNCTION ---
############################################
############################################

def read_and_parse(
    file_name, 
    date_cols = ['start_date', 'end_date', 'signing_date', 'creation_date_KSI_SIMIK_07_12'],
    file_path=r"C:\Users\jarem\OneDrive - London School of Economics\YEAR 2\1. Policy paper\policy-paper-repo\data\clean\treatment\eu_flows\intermediary"
    ):
    
    """
    Reads a CSV file with consistent dtype settings and parses datetime columns.

    Args:
        file_path (str): Folder path where the CSV is located.
        file_name (str): Name of the CSV file without extension.
        date_cols (list of str): List of columns to parse as datetime.

    Returns:
        pd.DataFrame: DataFrame with specified dtypes and parsed datetime columns.
    """
    # Full path to the file
    full_path = os.path.join(file_path, f"{file_name}.csv")
    
    
    # Read the CSV
    df = pd.read_csv(
        full_path, 
        dtype=data_types, 
        parse_dates=date_cols,
        low_memory=False
    )
    
    return df




############################################
############################################
# --- 1. GEOGRAPHIC INFERENCE FUNCTION ---
############################################
############################################


def infer_geography(df):
    """
    Step 1: Infers missing location data based on beneficiary legal status.
    Updates 'gmina', 'powiat', 'voviodeship' columns in-place.
    """
    def apply_inference(row):
        vov = row.get('voviodeship')
        gmi = row.get('gmina')
        pow_ = row.get('powiat')
        
        is_national = (vov == 'Cały kraj') or (pd.isnull(gmi) and pd.isnull(pow_))
        
        if is_national:
            legal = str(row.get('beneficiary_status', '')).lower()
            
            # Priority 1: Gmina Tier
            if ('gmina' in legal) or is_sme(row.get('beneficiary_status')) or \
                (row.get('beneficiary_status') in safe_gmina_forms):
                row['gmina'] = row.get('beneficiary_city')
                row['powiat'] = row.get('beneficiary_powiat')
                row['voviodeship'] = row.get('beneficiary_voviodeship')
            
            # Priority 2: Powiat Tier
            elif 'powiat' in legal:
                row['powiat'] = row.get('beneficiary_powiat')
                row['voviodeship'] = row.get('beneficiary_voviodeship')
                
        return row

    # Apply and return
    return df.apply(apply_inference, axis=1)

############################################
###########################################
# --- 2. TEMPORAL DISSAGREGATION FUNCTION ---
############################################
############################################

def distribute_funding_over_time(df, cols_to_distribute=None):
    """
    Optimized version: Splits project funding across years.
    Uses itertuples and dicts to prevent kernel hangs on large datasets.
    """
    if cols_to_distribute is None:
        cols_to_distribute = ['EU_subsidy_PLN', 'total_value_PLN', 'subsidy_PLN', 'eligible_expenses_PLN']

    # 1. Prepare Data
    df = df.copy()
    
    # Ensure date columns are datetime (just in case)
    for c in ['start_date', 'end_date', 'signing_date']:
        df[c] = pd.to_datetime(df[c], errors='coerce')

    # Handle numeric columns safely
    existing_cols = [c for c in cols_to_distribute if c in df.columns]
    for col in existing_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 2. Filter using vectorization (Fast)
    mask_completed = df['project_completed'].astype(str).str.strip().str.lower() == 'tak'
    mask_valid_dates = (df['start_date'].notna()) & (df['end_date'].notna()) & (df['start_date'] <= df['end_date'])
    mask_distribute = mask_completed & mask_valid_dates
    
    df_dist = df[mask_distribute]
    df_lump = df[~mask_distribute].copy()

    # 3. Fast Iteration using itertuples
    distributed_rows = []
    
    # Pre-calculate indices for columns to access them quickly inside loop
    # We convert the dataframe to a list of dicts for fastest access/copying
    records = df_dist.to_dict('records')
    
    for row in records:
        # Check if there is money to distribute
        if all(row[c] == 0 for c in existing_cols):
            continue

        # Generate quarters
        try:
            periods = pd.period_range(start=row['start_date'], end=row['end_date'], freq='Q')
        except:
            # Fallback for out-of-bounds dates
            periods = []
            
        n_quarters = len(periods)

        if n_quarters > 0:
            # Calculate annual portions
            # Logic: {2010: 0.25, 2011: 0.75} based on quarter count
            year_weights = {}
            for p in periods:
                year_weights[p.year] = year_weights.get(p.year, 0) + 1
            
            # Create new rows
            for year, quarters_in_year in year_weights.items():
                # Shallow copy the dictionary (Very Fast)
                new_row = row.copy() 
                new_row['Year'] = int(year)
                
                # Apply ratio
                ratio = quarters_in_year / n_quarters
                for col in existing_cols:
                    new_row[col] = row[col] * ratio
                
                distributed_rows.append(new_row)
        else:
            # Fallback: Assign to start year
            row['Year'] = row['start_date'].year
            distributed_rows.append(row)

    # 4. Handle Lump Sum rows
    # Assign to signing year
    df_lump['Year'] = df_lump['signing_date'].dt.year.fillna(0).astype(int)
    # Filter out rows where Year is 0 (missing signing date)
    df_lump = df_lump[df_lump['Year'] != 0]

    # 5. Recombine
    if distributed_rows:
        df_distributed = pd.DataFrame(distributed_rows)
        return pd.concat([df_distributed, df_lump], ignore_index=True)
    else:
        return df_lump

#################################################
#################################################
# --- 3. ASSIGN STATISTICAL ID TO GEOGRAPHIES ---
#################################################
#################################################

# 1. LOAD THE MAPS (Instant)
PICKLE_PATH = r"C:\Users\jarem\OneDrive - London School of Economics\YEAR 2\1. Policy paper\policy-paper-repo\data\outputs\teryt_lookup\teryt_lookup_2025.pkl"

with open(PICKLE_PATH, 'rb') as f:
    primary_map, fallback_map = pickle.load(f)

# 2. DEFINE HELPER (Must match the notebook logic)
def norm_text(s):
    if pd.isna(s): return ""
    s = str(s).lower()
    s = re.sub(r'\s+od\s+\d{4}', '', s) # Removes ' od 2002'
    for pat in ['m.st.', 'm.', 'st.', 'miasto', 'powiat', '-', '.', ' ']:
        s = s.replace(pat, '')
    return s

# 3. THE ASSIGNMENT FUNCTION
def assign_geo_ids(df):
    
    voiv_map = {
        'DOLNOŚLĄSKIE': '02', 'KUJAWSKO-POMORSKIE': '04', 'LUBELSKIE': '06', 
        'LUBUSKIE': '08', 'ŁÓDZKIE': '10', 'MAŁOPOLSKIE': '12', 
        'MAZOWIECKIE': '14', 'OPOLSKIE': '16', 'PODKARPACKIE': '18', 
        'PODLASKIE': '20', 'POMORSKIE': '22', 'ŚLĄSKIE': '24', 
        'ŚWIĘTOKRZYSKIE': '26', 'WARMIŃSKO-MAZURSKIE': '28', 
        'WIELKOPOLSKIE': '30', 'ZACHODNIOPOMORSKIE': '32'
    }

    def get_ids_row(row):
        # A. Resolve Voivodeship
        v_name = str(row.get('voviodeship', '')).upper()
        woj_id = voiv_map.get(v_name)
        if not woj_id: return pd.Series([None, None, None, None])

        # B. Normalize Inputs
        p_raw = row.get('powiat', '')
        g_raw = row.get('gmina', '')
        
        # Stop if no gmina
        if pd.isna(g_raw) or str(g_raw).strip() == '':
             return pd.Series([woj_id, None, None, None])

        p_norm = norm_text(p_raw)
        g_norm = norm_text(g_raw)
        
        # C. Lookup (Primary -> Fallback)
        teryt7 = primary_map.get((woj_id, p_norm, g_norm))
        
        if not teryt7:
            teryt7 = fallback_map.get((woj_id, g_norm))
            
        # D. Return
        if teryt7:
            return pd.Series([woj_id, teryt7[:4], teryt7, teryt7]) # Woj, Pow, Gmi, City
        else:
            return pd.Series([woj_id, None, None, None])

    # Apply
    df[['voivodeship_id', 'powiat_id', 'gmina_id', 'city_id']] = df.apply(get_ids_row, axis=1)
    return df


def aggregate_funding_by_gmina(df):
    """
    Aggregates project-level data to Gmina-Year level.
    Sums financial columns and keeps geographical identifiers.
    """
    # 1. Define columns to Group By
    # We group by ID to be precise, but also keep the names for readability.
    # Note: We include 'voivodeship_id' and 'powiat_id' to maintain hierarchy.
    group_cols = ['voivodeship_id', 'powiat_id', 'gmina_id', 'Year']
    
    # 2. Define Financial Columns to Sum
    # Add any other financial columns you have in your dataset
    fin_cols = [
        'EU_subsidy_PLN', 
        'total_value_PLN', 
        'subsidy_PLN', 
        'eligible_expenses_PLN'
    ]
    # Filter to only use columns that actually exist in your dataframe
    existing_fin_cols = [c for c in fin_cols if c in df.columns]
    
    # 3. Define Aggregation Logic
    # Sum financial data, take the 'first' name found for the location
    agg_dict = {col: 'sum' for col in existing_fin_cols}
    
    # Keep human-readable names (Optional but recommended)
    for name_col in ['voviodeship', 'powiat', 'gmina']:
        if name_col in df.columns:
            agg_dict[name_col] = 'first'
            
    # 4. Perform Aggregation
    # We drop rows where gmina_id is NaN (e.g. projects assigned to "Whole Voivodeship")
    # If you want to keep them, remove the dropna() line.
    df_clean = df.dropna(subset=['gmina_id'])
    
    df_gmina_level = df_clean.groupby(group_cols, as_index=False).agg(agg_dict)
    
    # 5. Sort for clean panel structure
    df_gmina_level = df_gmina_level.sort_values(by=['gmina_id', 'Year'])
    
    return df_gmina_level





##############################################
##############################################
# --- 5. SPATIAL DISAGGREGATION FUNCTION ---
##############################################
##############################################

def disaggregate_powiat_funding(df, path_to_pickle):
    """
    Implements OPTION B:
    If Gmina ID is missing but Powiat ID exists, split the funding 
    equally among all Gminas in that Powiat.
    """
    # 1. Load the Hierarchy Map
    with open(path_to_pickle, 'rb') as f:
        powiat_structure = pickle.load(f)
        
    # 2. Identify Rows to Split
    # Condition: Gmina is NaN/Empty AND Powiat is Valid
    mask_split = (df['gmina_id'].isna() | (df['gmina_id'] == '')) & (df['powiat_id'].notna())
    
    df_clean = df[~mask_split].copy()
    df_dirty = df[mask_split].copy()
    
    if df_dirty.empty:
        print("No Powiat-only rows found. Returning original.")
        return df_clean

    print(f"Disaggregating {len(df_dirty)} powiat-level rows...")

    # 3. Map Powiats to List of Gminas
    # We create a temporary column 'target_gminas' containing the list [0201011, 0201022...]
    # We lookup using a tuple index (woj_id, powiat_id)
    df_dirty['lookup_key'] = list(zip(df_dirty['voivodeship_id'], df_dirty['powiat_id']))
    df_dirty['target_gminas'] = df_dirty['lookup_key'].map(powiat_structure)
    
    # Filter out rows where map failed (orphaned codes) - rare but possible
    df_valid_split = df_dirty.dropna(subset=['target_gminas']).copy()
    
    # 4. Calculate Split Factor
    # If a powiat has 5 gminas, divisor is 5.
    df_valid_split['divisor'] = df_valid_split['target_gminas'].apply(len)
    
    # 5. Divide Financial Columns
    # List all money columns you want to split
    fin_cols = [c for c in ['EU_subsidy_PLN', 'total_value_PLN', 'subsidy_PLN', 'eligible_expenses_PLN'] if c in df.columns]
    
    for col in fin_cols:
        # Divide value by number of gminas
        df_valid_split[col] = df_valid_split[col] / df_valid_split['divisor']
        
    # 6. Explode (The Magic Step)
    # This turns 1 row with a list of 5 gminas into 5 rows, copying all other data
    df_exploded = df_valid_split.explode('target_gminas')
    
    # 7. Final Cleanup
    df_exploded['gmina_id'] = df_exploded['target_gminas']
    
    # Keep columns consistent with original df
    cols_to_keep = df.columns.tolist()
    df_final_split = df_exploded[cols_to_keep]
    
    # 8. Recombine with the clean data
    df_result = pd.concat([df_clean, df_final_split], ignore_index=True)
    
    print(f"Result: Expanded to {len(df_result)} rows (added {len(df_result) - len(df)} generated rows).")
    
    return df_result





###################################################################
###################################################################
########################## 2014- 2020##############################

def parse_and_explode_locations(df):
    """
    Parses 'project_place' column:
    1. Splits multiple locations (separated by '|')
    2. Extracts Voivodeship and Powiat
    3. Explodes rows (one row per location)
    4. Divides financial values by the number of locations
    """
    # Work on a copy
    df = df.copy()
    
    # 1. Parsing Logic
    def extract_locs(place_str):
        if pd.isna(place_str) or place_str == '':
            return []
        
        # Handle "Cały Kraj" explicitly
        if 'Cały Kraj' in place_str:
            return [{'voivodeship_raw': 'Cały Kraj', 'powiat_raw': None}]

        # Split by pipe '|'
        entries = place_str.split('|')
        parsed_entries = []
        
        for entry in entries:
            entry = entry.strip()
            # Regex to capture WOJ and optional POW
            # Pattern: Look for "WOJ.:" then content, then optional ", POW.:" then content
            match = re.search(r'WOJ\.:\s*(?P<woj>[^,]*)(?:,\s*POW\.:\s*(?P<pow>.*))?', entry, re.IGNORECASE)
            
            if match:
                w = match.group('woj').strip()
                p = match.group('pow')
                if p: p = p.strip()
                
                parsed_entries.append({'voivodeship_raw': w, 'powiat_raw': p})
            else:
                # Fallback if format is weird
                parsed_entries.append({'voivodeship_raw': entry, 'powiat_raw': None})
                
        return parsed_entries

    # Apply extraction -> Creates a column of lists
    df['parsed_locs'] = df['project_place'].apply(extract_locs)
    
    # 2. Calculate Split Factor (How many locations per project?)
    df['location_count'] = df['parsed_locs'].apply(len)
    
    # Avoid division by zero (if location was empty)
    df['location_count'] = df['location_count'].replace(0, 1)
    
    # 3. Split Financials
    # List of money columns to divide
    fin_cols = [c for c in ['total_value_PLN', 'eu_fund'] if c in df.columns]
    
    for col in fin_cols:
        # Convert to numeric just in case
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df[col] = df[col] / df['location_count']

    # 4. Explode
    # This turns 1 row into N rows
    df_exploded = df.explode('parsed_locs')
    
    # 5. Extract Dict keys to columns
    # Safe way to unpack the dictionaries into columns
    loc_data = df_exploded['parsed_locs'].apply(pd.Series)
    
    # Merge back
    df_final = pd.concat([df_exploded.drop(['parsed_locs'], axis=1), loc_data], axis=1)
    
    # 6. Renaming for compatibility with your existing Assign ID function
    # Your assign_geo_ids expects 'voviodeship' and 'powiat'
    df_final = df_final.rename(columns={
        'voivodeship_raw': 'voviodeship', 
        'powiat_raw': 'powiat'
    })
    
    return df_final