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
##################################################
##################################################
# --- 0. READ AND PARSE DATETIME CSV FUNCTION ---
##################################################
#################################################

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

# # 1. LOAD THE MAPS (Instant)
PICKLE_PATH = r"C:\Users\jarem\OneDrive - London School of Economics\YEAR 2\1. Policy paper\policy-paper-repo\data\outputs\teryt_lookup\teryt_lookup_2025.pkl"  

# with open(PICKLE_PATH, 'rb') as f:
#     primary_map, fallback_map = pickle.load(f)

# 2. DEFINE HELPER (Must match the notebook logic)
def norm_text(s):
    if pd.isna(s): return ""
    s = str(s).lower()
    s = re.sub(r'\s+od\s+\d{4}', '', s) # Removes ' od 2002'
    for pat in ['m.st.', 'm.', 'st.', 'miasto', 'powiat', '-', '.', ' ']:
        s = s.replace(pat, '')
    return s

# 3. THE ASSIGNMENT FUNCTION
# def assign_geo_ids(df):
    
#     voiv_map = {
#         'DOLNOŚLĄSKIE': '02', 'KUJAWSKO-POMORSKIE': '04', 'LUBELSKIE': '06', 
#         'LUBUSKIE': '08', 'ŁÓDZKIE': '10', 'MAŁOPOLSKIE': '12', 
#         'MAZOWIECKIE': '14', 'OPOLSKIE': '16', 'PODKARPACKIE': '18', 
#         'PODLASKIE': '20', 'POMORSKIE': '22', 'ŚLĄSKIE': '24', 
#         'ŚWIĘTOKRZYSKIE': '26', 'WARMIŃSKO-MAZURSKIE': '28', 
#         'WIELKOPOLSKIE': '30', 'ZACHODNIOPOMORSKIE': '32'
#     }

#     def get_ids_row(row):
#         # A. Resolve Voivodeship
#         v_name = str(row.get('voviodeship', '')).upper()
#         woj_id = voiv_map.get(v_name)
#         if not woj_id: return pd.Series([None, None, None, None])

#         # B. Normalize Inputs
#         p_raw = row.get('powiat', '')
#         g_raw = row.get('gmina', '')
        
#         # Stop if no gmina
#         if pd.isna(g_raw) or str(g_raw).strip() == '':
#              return pd.Series([woj_id, None, None, None])

#         p_norm = norm_text(p_raw)
#         g_norm = norm_text(g_raw)
        
#         # C. Lookup (Primary -> Fallback)
#         teryt7 = primary_map.get((woj_id, p_norm, g_norm))
        
#         if not teryt7:
#             teryt7 = fallback_map.get((woj_id, g_norm))
            
#         # D. Return
#         if teryt7:
#             return pd.Series([woj_id, teryt7[:4], teryt7, teryt7]) # Woj, Pow, Gmi, City
#         else:
#             return pd.Series([woj_id, None, None, None])

#     # Apply
#     df[['voivodeship_id', 'powiat_id', 'gmina_id', 'city_id']] = df.apply(get_ids_row, axis=1)
#     return df


# def aggregate_funding_by_gmina(df):
#     """
#     Aggregates project-level data to Gmina-Year level.
#     Sums financial columns and keeps geographical identifiers.
#     """
#     # 1. Define columns to Group By
#     # We group by ID to be precise, but also keep the names for readability.
#     # Note: We include 'voivodeship_id' and 'powiat_id' to maintain hierarchy.
#     group_cols = ['voivodeship_id', 'powiat_id', 'gmina_id', 'Year']
    
#     # 2. Define Financial Columns to Sum
#     # Add any other financial columns you have in your dataset
#     fin_cols = [
#         'EU_subsidy_PLN', 
#         'total_value_PLN', 
#         'subsidy_PLN', 
#         'eligible_expenses_PLN'
#     ]
#     # Filter to only use columns that actually exist in your dataframe
#     existing_fin_cols = [c for c in fin_cols if c in df.columns]
    
#     # 3. Define Aggregation Logic
#     # Sum financial data, take the 'first' name found for the location
#     agg_dict = {col: 'sum' for col in existing_fin_cols}
    
#     # Keep human-readable names (Optional but recommended)
#     for name_col in ['voviodeship', 'powiat', 'gmina']:
#         if name_col in df.columns:
#             agg_dict[name_col] = 'first'
            
#     # 4. Perform Aggregation
#     # We drop rows where gmina_id is NaN (e.g. projects assigned to "Whole Voivodeship")
#     # If you want to keep them, remove the dropna() line.
#     df_clean = df.dropna(subset=['gmina_id'])
    
#     df_gmina_level = df_clean.groupby(group_cols, as_index=False).agg(agg_dict)
    
#     # 5. Sort for clean panel structure
#     df_gmina_level = df_gmina_level.sort_values(by=['gmina_id', 'Year'])
    
#     return df_gmina_level





# ##############################################
# ##############################################
# # --- 5. SPATIAL DISAGGREGATION FUNCTION ---
# ##############################################
# ##############################################

# def disaggregate_powiat_funding(df, path_to_pickle):
#     """
#     Implements OPTION B:
#     If Gmina ID is missing but Powiat ID exists, split the funding 
#     equally among all Gminas in that Powiat.
#     """
#     # 1. Load the Hierarchy Map
#     with open(path_to_pickle, 'rb') as f:
#         powiat_structure = pickle.load(f)
        
#     # 2. Identify Rows to Split
#     # Condition: Gmina is NaN/Empty AND Powiat is Valid
#     mask_split = (df['gmina_id'].isna() | (df['gmina_id'] == '')) & (df['powiat_id'].notna())
    
#     df_clean = df[~mask_split].copy()
#     df_dirty = df[mask_split].copy()
    
#     if df_dirty.empty:
#         print("No Powiat-only rows found. Returning original.")
#         return df_clean

#     print(f"Disaggregating {len(df_dirty)} powiat-level rows...")

#     # 3. Map Powiats to List of Gminas
#     # We create a temporary column 'target_gminas' containing the list [0201011, 0201022...]
#     # We lookup using a tuple index (woj_id, powiat_id)
#     df_dirty['lookup_key'] = list(zip(df_dirty['voivodeship_id'], df_dirty['powiat_id']))
#     df_dirty['target_gminas'] = df_dirty['lookup_key'].map(powiat_structure)
    
#     # Filter out rows where map failed (orphaned codes) - rare but possible
#     df_valid_split = df_dirty.dropna(subset=['target_gminas']).copy()
    
#     # 4. Calculate Split Factor
#     # If a powiat has 5 gminas, divisor is 5.
#     df_valid_split['divisor'] = df_valid_split['target_gminas'].apply(len)
    
#     # 5. Divide Financial Columns
#     # List all money columns you want to split
#     fin_cols = [c for c in ['EU_subsidy_PLN', 'total_value_PLN', 'subsidy_PLN', 'eligible_expenses_PLN'] if c in df.columns]
    
#     for col in fin_cols:
#         # Divide value by number of gminas
#         df_valid_split[col] = df_valid_split[col] / df_valid_split['divisor']
        
#     # 6. Explode (The Magic Step)
#     # This turns 1 row with a list of 5 gminas into 5 rows, copying all other data
#     df_exploded = df_valid_split.explode('target_gminas')
    
#     # 7. Final Cleanup
#     df_exploded['gmina_id'] = df_exploded['target_gminas']
    
#     # Keep columns consistent with original df
#     cols_to_keep = df.columns.tolist()
#     df_final_split = df_exploded[cols_to_keep]
    
#     # 8. Recombine with the clean data
#     df_result = pd.concat([df_clean, df_final_split], ignore_index=True)
    
#     print(f"Result: Expanded to {len(df_result)} rows (added {len(df_result) - len(df)} generated rows).")
    
#     return df_result


def assign_geo_ids(df, lookup_tuple):
    """
    Assigns TERYT IDs.
    CRITICAL CHANGE: Now accepts and uses 'powiat_map' to handle rows 
    that have a Powiat but no Gmina.
    """
    # Unpack all 4 elements (Ensure your pickle/builder returns these 4)
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
        # A. Resolve Voivodeship
        v_name = str(row.get('voviodeship', '')).upper()
        woj_id = voiv_map.get(v_name)
        if not woj_id: return pd.Series([None, None, None, None])

        # B. Normalize Inputs
        p_raw = row.get('powiat', '')
        g_raw = row.get('gmina', '')
        
        p_norm = norm_func(p_raw)
        g_norm = norm_func(g_raw)
        
        # C. Strategy 1: Full Gmina Match
        teryt7 = primary_map.get((woj_id, p_norm, g_norm))
        if not teryt7:
            teryt7 = fallback_map.get((woj_id, g_norm))
            
        if teryt7:
            # Success: Return Full Hierarchy
            return pd.Series([woj_id, teryt7[:4], teryt7, teryt7])
        
        # D. Strategy 2: Powiat Only Match (CRITICAL FOR DISAGGREGATION)
        # If Gmina failed or was missing, try to find just the Powiat ID
        powiat_id = powiat_map.get((woj_id, p_norm))
        
        if powiat_id:
            # We found the Powiat ID (e.g. '0201'), but Gmina is None
            # This allows disaggregate_powiat_funding to pick this row up later!
            return pd.Series([woj_id, powiat_id, None, None])
            
        # E. Failure
        return pd.Series([woj_id, None, None, None])

    # Apply
    df[['voivodeship_id', 'powiat_id', 'gmina_id', 'city_id']] = df.apply(get_ids_row, axis=1)
    return df

###################################################################
###################################################################
########################## 2014- 2020 ##############################

def parse_and_explode_locations(df):
    """
    Parsuje kolumnę 'project_place' w nowym formacie:
    1. Rozdziela wiele lokalizacji (separator '|')
    2. Wyciąga Województwo i Powiat używając Regex.
    3. Rozbija wiersze (explode) - jeden wiersz na lokalizację.
    4. Dzieli kwoty finansowe przez liczbę lokalizacji.
    """
    # Pracujemy na kopii, żeby nie zmieniać oryginału
    df = df.copy()
    
    # 1. Logika wyciągania danych z tekstu
    def extract_locs(place_str):
        if pd.isna(place_str) or place_str == '':
            return []
        
        # Obsługa "Cały Kraj" - przypisujemy specjalną wartość lub None
        if 'Cały Kraj' in str(place_str):
            return [{'voivodeship_raw': 'Cały Kraj', 'powiat_raw': None}]

        # Krok A: Podział po znaku pipe '|'
        entries = str(place_str).split('|')
        parsed_entries = []
        
        for entry in entries:
            entry = entry.strip()
            # Krok B: Regex wyciągający WOJ i opcjonalnie POW
            # Szuka frazy "WOJ.:", bierze tekst do przecinka, potem opcjonalnie "POW.:"
            match = re.search(r'WOJ\.:\s*(?P<woj>[^,]*)(?:,\s*POW\.:\s*(?P<pow>.*))?', entry, re.IGNORECASE)
            
            if match:
                w = match.group('woj').strip()
                p = match.group('pow')
                if p: 
                    p = p.strip()
                
                parsed_entries.append({'voivodeship_raw': w, 'powiat_raw': p})
            else:
                # Fallback, jeśli format jest dziwny (np. sama nazwa bez prefixu)
                parsed_entries.append({'voivodeship_raw': entry, 'powiat_raw': None})
                
        return parsed_entries

    # 2. Aplikujemy funkcję - tworzy kolumnę list słowników
    print("Parsowanie lokalizacji...")
    df['parsed_locs'] = df['project_place'].apply(extract_locs)
    
    # 3. Liczymy ile lokalizacji ma dany projekt (do podziału pieniędzy)
    df['location_count'] = df['parsed_locs'].apply(len)
    # Zabezpieczenie przed dzieleniem przez 0
    df['location_count'] = df['location_count'].replace(0, 1)
    
    # 4. Dzielimy finanse
    # Lista kolumn finansowych do podzielenia (dostosuj do swoich nazw w tym pliku)
    fin_cols = [c for c in ['total_value_PLN', 'eu_fund', 'total_value', 'dofinansowanie_ue'] if c in df.columns]
    
    for col in fin_cols:
        # Konwersja na liczby (na wszelki wypadek)
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        # Podział
        df[col] = df[col] / df['location_count']

    # 5. Explode - Rozbicie wierszy
    # Z jednego wiersza z listą [Lok1, Lok2] robi dwa wiersze
    print(f"Rozbijanie wierszy (Explode). Liczba wierszy przed: {len(df)}")
    df_exploded = df.explode('parsed_locs')
    print(f"Liczba wierszy po: {len(df_exploded)}")
    
    # 6. Zamiana słowników na kolumny
    # Rozpakowuje kolumnę 'parsed_locs' ({'voivodeship_raw': '...', 'powiat_raw': '...'}) na osobne kolumny
    loc_data = df_exploded['parsed_locs'].apply(pd.Series)
    
    # Łączymy z resztą danych
    df_final = pd.concat([df_exploded.drop(['parsed_locs'], axis=1), loc_data], axis=1)
    
    # 7. Zmiana nazw na kompatybilne z Twoją funkcją assign_geo_ids
    # Twoja funkcja spodziewa się 'voviodeship' i 'powiat'
    df_final = df_final.rename(columns={
        'voivodeship_raw': 'voviodeship', 
        'powiat_raw': 'powiat'
    })
    
    return df_final


################### VERSION 1
# def build_teryt_lookup_optimized(path_excel):
#     print(f"Loading TERYT data from: {path_excel}...")
    
#     # Load data
#     cols_to_load = ['region', 'nazwa_powiatu', 'nazwa_gminy', 'teryt_2025', 'zmiana_opis']
#     try:
#         df_g = pd.read_excel(path_excel, sheet_name='gminy', dtype=str, usecols=cols_to_load)
#     except ValueError:
#         df_g = pd.read_excel(path_excel, sheet_name='gminy', dtype=str)

#     # --- 1. NORMALIZATION ---
#     def vectorize_norm(series):
#         return (series.astype(str).str.lower()
#                 .str.replace(r'\s+od\s+\d{4}', '', regex=True)
#                 .str.replace('m.st.', '', regex=False)
#                 .str.replace('m.', '', regex=False)
#                 .str.replace('st.', '', regex=False)
#                 .str.replace('miasto', '', regex=False)
#                 .str.replace('powiat', '', regex=False)
#                 .str.replace('-', '', regex=False)
#                 .str.replace('.', '', regex=False)
#                 .str.replace(' ', '', regex=False))

#     woj_id = df_g['region'].astype(str).str.split('.').str[0].str.zfill(2)
#     pow_norm = vectorize_norm(df_g['nazwa_powiatu'])
#     gmi_norm = vectorize_norm(df_g['nazwa_gminy'])
#     target_id = df_g['teryt_2025'].astype(str).str.split('.').str[0].str.zfill(7)

#     # --- 2. BUILD MAPS ---
    
#     # Map A: Full Gmina Lookup
#     primary_lookup = dict(zip(zip(woj_id, pow_norm, gmi_norm), target_id))

#     # Map B: Fallback Gmina Lookup (Unique names)
#     df_g['woj_id'] = woj_id
#     df_g['gmi_norm'] = gmi_norm
#     df_g['target_id'] = target_id
#     unique_check = df_g.groupby(['woj_id', 'gmi_norm'])['target_id'].nunique()
#     valid_fallbacks = unique_check[unique_check == 1].index
#     fallback_df = df_g.set_index(['woj_id', 'gmi_norm']).loc[valid_fallbacks]
#     fallback_lookup = fallback_df['target_id'].to_dict()
    
#     # --- Map C: POWIAT LOOKUP (New & Critical) ---
#     # This enables matching rows that only have "Powiat" and no "Gmina"
#     df_g['powiat_id_4'] = df_g['target_id'].str[:4]
#     # Drop duplicates to get unique powiat list
#     pow_df = df_g[['woj_id', 'nazwa_powiatu', 'powiat_id_4']].drop_duplicates()
#     pow_df['pow_norm'] = vectorize_norm(pow_df['nazwa_powiatu'])
    
#     powiat_lookup = dict(zip(zip(pow_df['woj_id'], pow_df['pow_norm']), pow_df['powiat_id_4']))
    
#     # --- 3. HISTORICAL LOGIC (Renames/Absorptions) ---
#     # (Existing logic for renames... kept brief for clarity)
#     mask_rename = df_g['zmiana_opis'].str.contains('zmiana nazwy z', na=False, case=False)
#     if mask_rename.any():
#         # ... [Keep your existing historical parsing logic here] ...
#         pass 

#     print("TERYT Lookup built successfully.")

#     # Define simple norm for the Assigner to use
#     def simple_norm(s):
#         if pd.isna(s): return ""
#         s = str(s).lower()
#         s = re.sub(r'\s+od\s+\d{4}', '', s)
#         for pat in ['m.st.', 'm.', 'st.', 'miasto', 'powiat', '-', '.', ' ']:
#             s = s.replace(pat, '')
#         return s

#     # RETURN 4 ITEMS (Primary, Fallback, POWIAT, NormFunc)
#     return primary_lookup, fallback_lookup, powiat_lookup, simple_norm




def build_teryt_lookup_optimized(path_excel):
    print(f"Loading TERYT data from: {path_excel}...")
    
    # Load data
    cols_to_load = ['region', 'nazwa_powiatu', 'nazwa_gminy', 'teryt_2025', 'zmiana_opis']
    try:
        df_g = pd.read_excel(path_excel, sheet_name='gminy', dtype=str, usecols=cols_to_load)
    except ValueError:
        df_g = pd.read_excel(path_excel, sheet_name='gminy', dtype=str)

    # --- 1. NORMALIZATION ---
    def vectorize_norm(series):
        return (series.astype(str).str.lower()
                .str.replace(r'\s+od\s+\d{4}', '', regex=True)
                .str.replace('m.st.', '', regex=False)
                .str.replace('m.', '', regex=False)
                .str.replace('st.', '', regex=False)
                .str.replace('miasto', '', regex=False)
                .str.replace('powiat', '', regex=False)
                .str.replace('-', '', regex=False)
                .str.replace('.', '', regex=False)
                .str.replace(' ', '', regex=False))

    woj_id = df_g['region'].astype(str).str.split('.').str[0].str.zfill(2)
    pow_norm = vectorize_norm(df_g['nazwa_powiatu'])
    gmi_norm = vectorize_norm(df_g['nazwa_gminy'])
    target_id = df_g['teryt_2025'].astype(str).str.split('.').str[0].str.zfill(7)

    # --- 2. BUILD MAPS ---
    
    # Map A: Full Gmina Lookup
    primary_lookup = dict(zip(zip(woj_id, pow_norm, gmi_norm), target_id))

    # Map B: Fallback Gmina Lookup (Unique names)
    df_g['woj_id'] = woj_id
    df_g['gmi_norm'] = gmi_norm
    df_g['target_id'] = target_id
    unique_check = df_g.groupby(['woj_id', 'gmi_norm'])['target_id'].nunique()
    valid_fallbacks = unique_check[unique_check == 1].index
    fallback_df = df_g.set_index(['woj_id', 'gmi_norm']).loc[valid_fallbacks]
    fallback_lookup = fallback_df['target_id'].to_dict()
    
    # --- Map C: POWIAT LOOKUP (New & Critical) ---
    # This enables matching rows that only have "Powiat" and no "Gmina"
    df_g['powiat_id_4'] = df_g['target_id'].str[:4]
    # Drop duplicates to get unique powiat list
    pow_df = df_g[['woj_id', 'nazwa_powiatu', 'powiat_id_4']].drop_duplicates()
    pow_df['pow_norm'] = vectorize_norm(pow_df['nazwa_powiatu'])
    
    powiat_lookup = dict(zip(zip(pow_df['woj_id'], pow_df['pow_norm']), pow_df['powiat_id_4']))
    
    # --- 3. HISTORICAL LOGIC (Renames/Absorptions) ---
    # (Existing logic for renames... kept brief for clarity)
    mask_rename = df_g['zmiana_opis'].str.contains('zmiana nazwy z', na=False, case=False)
    if mask_rename.any():
        # ... [Keep your existing historical parsing logic here] ...
        pass 

    print("TERYT Lookup built successfully.")

    # Define simple norm for the Assigner to use
    def simple_norm(s):
        if pd.isna(s): return ""
        s = str(s).lower()
        s = re.sub(r'\s+od\s+\d{4}', '', s)
        for pat in ['m.st.', 'm.', 'st.', 'miasto', 'powiat', '-', '.', ' ']:
            s = s.replace(pat, '')
        return s

    # RETURN 4 ITEMS (Primary, Fallback, POWIAT, NormFunc)
    return primary_lookup, fallback_lookup, powiat_lookup, simple_norm


######################################################################
########################### AGGGREGATION ########################
#########################################################################


# --- CONFIGURATION ---
# Path to the folder where your clean CSVs are saved
PATH_TO_OUTPUTS = r"C:\Users\jarem\OneDrive - London School of Economics\YEAR 2\1. Policy paper\policy-paper-repo\data\clean\treatment\eu_flows\final"  # <--- UPDATE THIS PATH

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
            df = pd.read_csv(f)
            
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

# --- EXECUTE ---
df_powiat_all, df_gmina_all = knit_directory_with_periods(PATH_TO_OUTPUTS)

# --- SAVE ---
if df_powiat_all is not None:
    print(f"\nFinal Powiat Panel: {df_powiat_all.shape}")
    print(df_powiat_all['programming_period'].value_counts())
    df_powiat_all.to_csv("Master_Powiat_Panel_With_Periods.csv", index=False)

if df_gmina_all is not None:
    print(f"Final Gmina Panel: {df_gmina_all.shape}")
    df_gmina_all.to_csv("Master_Gmina_Panel_With_Periods.csv", index=False)