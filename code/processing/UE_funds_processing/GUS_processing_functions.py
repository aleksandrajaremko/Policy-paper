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
    
    # Define consistent dtypes
    dtypes = {
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
    
    # Read the CSV
    df = pd.read_csv(
        full_path, 
        dtype=dtypes, 
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

# def build_teryt_lookups(
#     path_excel = r"C:\Users\jarem\OneDrive - London School of Economics\YEAR 2\1. Policy paper\policy-paper-repo\data\inputs\shapefiles\polska\teryt_klucz_powiaty_gminy_lata_1999_2025-1.xlsx"
#     ):
#     # Load Data
#     df_g = pd.read_excel(path_excel, sheet_name='gminy', dtype=str)
    
#     primary_lookup = {}
#     fallback_candidates = {} 

#     # Define Norm Function INSIDE builder and return it, or define it globally
#     def norm(s):
#         if pd.isna(s): return ""
#         s = str(s).lower()
#         s = re.sub(r'\s+od\s+\d{4}', '', s) 
#         s = s.replace('m.st.', '').replace('m.', '').replace('st.', '')
#         s = s.replace('miasto', '').replace('powiat', '')
#         s = s.replace('-', '').replace('.', '').replace(' ', '')
#         return s

#     for _, row in df_g.iterrows():
#         woj_id = str(row['region']).split('.')[0].zfill(2)
#         pow_norm = norm(row['nazwa_powiatu'])
#         gmi_norm = norm(row['nazwa_gminy'])
#         target_id = str(row['teryt_2025']).split('.')[0].zfill(7)
        
#         # 1. Primary Map
#         primary_lookup[(woj_id, pow_norm, gmi_norm)] = target_id
        
#         # 2. Fallback Candidate
#         fb_key = (woj_id, gmi_norm)
#         if fb_key not in fallback_candidates: fallback_candidates[fb_key] = []
#         fallback_candidates[fb_key].append(target_id)
        
#         # 3. Historical
#         desc = str(row.get('zmiana_opis', ''))
#         if 'zmiana nazwy z' in desc:
#             match = re.search(r'zmiana nazwy z\s+(.+?)\s+na', desc, re.IGNORECASE)
#             if match:
#                 old_name = norm(match.group(1))
#                 primary_lookup[(woj_id, pow_norm, old_name)] = target_id
#                 fb_key_hist = (woj_id, old_name)
#                 if fb_key_hist not in fallback_candidates: fallback_candidates[fb_key_hist] = []
#                 fallback_candidates[fb_key_hist].append(target_id)

#     # Filter Fallback to unique only
#     fallback_lookup = {k: v[0] for k, v in fallback_candidates.items() if len(set(v)) == 1}
            
#     # Return the maps AND the normalization function to be used later
#     return primary_lookup, fallback_lookup, norm

# # --- CONFIGURATION ---
# # Set your default path here so you don't have to type it every time
# DEFAULT_TERYT_PATH = r"C:\Users\jarem\OneDrive - London School of Economics\YEAR 2\1. Policy paper\policy-paper-repo\data\inputs\shapefiles\polska\teryt_klucz_powiaty_gminy_lata_1999_2025-1.xlsx"

# # --- 1. THE BUILDER (With Caching) ---
# # @lru_cache(maxsize=1)
# def build_teryt_lookups(path_excel):
#     """
#     Reads TERYT Excel and builds lookup maps.
#     Cached: Subsequent calls with the same path are instant.
#     """
#     print(f"Loading TERYT data from: {path_excel}...")
    
#     # Load Data
#     try:
#         df_g = pd.read_excel(path_excel, sheet_name='gminy', dtype=str)
#     except FileNotFoundError:
#         raise FileNotFoundError(f"Could not find TERYT file at: {path_excel}")
    
#     primary_lookup = {}
#     fallback_candidates = {}

#     # Define Norm Function
#     def norm(s):
#         if pd.isna(s): return ""
#         s = str(s).lower()
#         # Specific TERYT fixes
#         s = re.sub(r'\s+od\s+\d{4}', '', s) # Remove ' od 2002'
#         s = s.replace('m.st.', '').replace('m.', '').replace('st.', '')
#         s = s.replace('miasto', '').replace('powiat', '')
#         s = s.replace('-', '').replace('.', '').replace(' ', '')
#         return s

#     for _, row in df_g.iterrows():
#         woj_id = str(row['region']).split('.')[0].zfill(2)
#         pow_norm = norm(row['nazwa_powiatu'])
#         gmi_norm = norm(row['nazwa_gminy'])
#         target_id = str(row['teryt_2025']).split('.')[0].zfill(7)
        
#         # 1. Primary Map
#         primary_lookup[(woj_id, pow_norm, gmi_norm)] = target_id
        
#         # 2. Fallback Candidates
#         fb_key = (woj_id, gmi_norm)
#         if fb_key not in fallback_candidates: fallback_candidates[fb_key] = []
#         fallback_candidates[fb_key].append(target_id)
        
#         # 3. Historical Names
#         desc = str(row.get('zmiana_opis', ''))
#         if 'zmiana nazwy z' in desc:
#             match = re.search(r'zmiana nazwy z\s+(.+?)\s+na', desc, re.IGNORECASE)
#             if match:
#                 old_name = norm(match.group(1))
#                 primary_lookup[(woj_id, pow_norm, old_name)] = target_id
                
#                 fb_key_hist = (woj_id, old_name)
#                 if fb_key_hist not in fallback_candidates: fallback_candidates[fb_key_hist] = []
#                 fallback_candidates[fb_key_hist].append(target_id)

#     # Filter Fallback to Unique Only
#     fallback_lookup = {k: v[0] for k, v in fallback_candidates.items() if len(set(v)) == 1}
    
#     print("TERYT Lookup built successfully.")
#     return primary_lookup, fallback_lookup, norm


# # --- 2. THE ASSIGNER (Smart Wrapper) ---
# def assign_geo_ids(df, lookup_tuple=None, path_excel=DEFAULT_TERYT_PATH):
#     """
#     Assigns TERYT IDs to the dataframe.
    
#     Parameters:
#     - df: The dataframe to process.
#     - lookup_tuple: (Optional) The output of build_teryt_lookup_advanced.
#     - path_excel: (Optional) Path to the Excel file if lookup_tuple is missing.
#       Defaults to the DEFAULT_TERYT_PATH constant.
#     """
    
#     # 1. Auto-build lookup if not provided
#     if lookup_tuple is None:
#         lookup_tuple = build_teryt_lookups(path_excel)
        
#     primary_map, fallback_map, norm_func = lookup_tuple
    
#     # 2. Hardcoded Voivodeship Map
#     voiv_map = {
#         'DOLNOŚLĄSKIE': '02', 'KUJAWSKO-POMORSKIE': '04', 'LUBELSKIE': '06', 
#         'LUBUSKIE': '08', 'ŁÓDZKIE': '10', 'MAŁOPOLSKIE': '12', 
#         'MAZOWIECKIE': '14', 'OPOLSKIE': '16', 'PODKARPACKIE': '18', 
#         'PODLASKIE': '20', 'POMORSKIE': '22', 'ŚLĄSKIE': '24', 
#         'ŚWIĘTOKRZYSKIE': '26', 'WARMIŃSKO-MAZURSKIE': '28', 
#         'WIELKOPOLSKIE': '30', 'ZACHODNIOPOMORSKIE': '32'
#     }
    
#     # 3. Apply Logic
#     def get_ids_row(row):
#         v_name = str(row.get('voviodeship', '')).upper()
#         woj_id = voiv_map.get(v_name)
#         if not woj_id: return pd.Series([None, None, None, None])

#         p_raw = row.get('powiat', '')
#         g_raw = row.get('gmina', '')
        
#         if pd.isna(g_raw) or str(g_raw).strip() == '':
#              return pd.Series([woj_id, None, None, None])

#         p_norm = norm_func(p_raw)
#         g_norm = norm_func(g_raw)
        
#         # Try Primary
#         teryt7 = primary_map.get((woj_id, p_norm, g_norm))
#         # Try Fallback
#         if not teryt7:
#             teryt7 = fallback_map.get((woj_id, g_norm))
            
#         if teryt7:
#             teryt4 = teryt7[:4]
#             return pd.Series([woj_id, teryt4, teryt7, teryt7])
#         else:
#             return pd.Series([woj_id, None, None, None])

#     # Assign columns
#     df[['voivodeship_id', 'powiat_id', 'gmina_id', 'city_id']] = df.apply(get_ids_row, axis=1)
    
#     return df


# ############################################
# ############################################
# # --- 4. SPATIAL DISAGGREGATION FUNCTION ---
# ############################################
# ############################################

# def disaggregate_powiat_funding(df, teryt_map):
#     """
#     Step 3: Splits Powiat-only funding equally among constituent Gminas.
#     teryt_map: Dict { 'normalized_powiat_name': ['Gmina A', 'Gmina B'] }
#     """
#     def norm_key(s):
#         return str(s).lower().replace('powiat', '').replace(' ', '').replace('.', '').replace('-', '')

#     # Identify rows needing split
#     mask_split = df['gmina'].isnull() & df['powiat'].notnull()
    
#     df_ready = df[~mask_split].copy()
#     df_to_split = df[mask_split].copy()
    
#     disaggregated_rows = []
    
#     if not df_to_split.empty:
#         df_to_split['powiat_key'] = df_to_split['powiat'].apply(norm_key)
        
#         for p_key, group in df_to_split.groupby('powiat_key'):
#             gminas = teryt_map.get(p_key, [])
            
#             if gminas:
#                 count = len(gminas)
#                 group['EU_subsidy_PLN'] /= count
                
#                 # Replicate rows
#                 group_repeated = group.loc[group.index.repeat(count)].copy()
#                 group_repeated['gmina'] = gminas * len(group)
                
#                 disaggregated_rows.append(group_repeated)
#             else:
#                 # Keep unmapped powiats
#                 group['gmina'] = 'Unknown/Unmapped'
#                 disaggregated_rows.append(group)
    
#     if disaggregated_rows:
#         return pd.concat([df_ready, pd.concat(disaggregated_rows)], ignore_index=True)
    
#     return df_ready



