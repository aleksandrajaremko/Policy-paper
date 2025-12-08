import re 
import pandas as pd
import unicodedata
import glob

OFFICIAL_POWIAT_COUNT = {
    'dolnoslaskie': 30,
    'kujawsko-pomorskie': 23,
    'lubelskie': 24,
    'lubuskie': 13,
    'lodzkie': 24,
    'malopolskie': 22,
    'mazowieckie': 42,
    'opolskie': 12,
    'podkarpackie': 18,
    'podlaskie': 17,
    'pomorskie': 20,
    'slaskie': 40,
    'swietokrzyskie': 14,
    'warminsko-mazurskie': 22,
    'wielkopolskie': 34,
    'zachodniopomorskie': 18,
}


############## DATA CLEANING FUNCTIONS #####################

def clean_english_colnames(df, normalize='keep'):
    """
    Keep only the English text after '/' in column names.
    normalize: 'keep' -> keep English text as-is (trimmed)
               'snake' -> lowercase, non-alphanum removed, spaces -> _
    Returns a copy with cleaned column names and resolves duplicates by suffixing _1, _2...
    """
    cleaned = []
    for c in df.columns:
        s = str(c).replace('\xa0', ' ').replace('\n', ' ').strip()
        if '/' in s:
            s = s.split('/')[-1].strip()
        s = ' '.join(s.split())
        if normalize == 'snake':
            s = s.lower()
            s = re.sub(r'[^\w\s]', '', s)
            s = re.sub(r'\s+', '_', s)
        cleaned.append(s)

    # resolve duplicates
    seen = {}
    final = []
    for name in cleaned:
        if name in seen:
            seen[name] += 1
            final.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            final.append(name)
    df2 = df.copy()
    df2.columns = final
    return df2

def standardize_polish_name(name):
    """
    Standardizes Polish administrative unit names for merging.
    - Normalizes Polish characters (diacritics).
    - Lowercases and removes common prefixes like 'powiat', 'm.', 'gmina'.
    - Trims whitespace.
    - Handles 'm. st. warszawa'.
    """
    if pd.isna(name):
        return None
    
    # Lowercase and strip
    s = str(name).lower().strip()

    # Normalize Polish characters (e.g., 'ł' -> 'l', 'ą' -> 'a')
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8')

    # Handle special cases like 'm. st. warszawa'
    s = re.sub(r'^m\.\s*st\.\s+', '', s)

    # Remove common prefixes
    s = re.sub(r'^(powiat|gmina|m\.)\s+', '', s)

    # Remove any remaining non-alphanumeric characters except spaces
    s = re.sub(r'[^a-z0-9\s]', '', s)

    return ' '.join(s.split()) # Normalize whitespace


############## LOCATION EXTRACTION AND CLEANING #####################

def extract_location_parts(series: pd.Series) -> pd.DataFrame:
    """
    Uses regex to extract WOJ, POW, and GMIN from a Series of location strings.

    Args:
        series (pd.Series): A pandas Series containing location strings.

    Returns:
        pd.DataFrame: A DataFrame with 'wojewodztwo', 'powiat', and 'gmina' columns.
    """
    # Define regex patterns for each part.
    # The pattern looks for the keyword (e.g., "WOJ."), a colon, optional whitespace,
    # and then captures all characters until it hits a comma or the end of the string.
    patterns = {
        'wojewodztwo': r'WOJ\.\s*:\s*([^,]+)',
        'powiat':      r'POW\.\s*:\s*([^,]+)',
        'gmina':       r'GMIN\.\s*:\s*([^,]+)'
    }

    # Apply .str.extract() for each pattern and combine the results.
    # .extract() returns a DataFrame with the captured group.
    # We use .get(0, default=pd.NA) to safely handle cases where a pattern is not found.
    location_df = pd.DataFrame()
    for col, pat in patterns.items():
        # Extract returns a DataFrame; we want the first (and only) capture group.
        extracted = series.str.extract(pat, expand=False)
        location_df[col] = extracted.str.strip() # Clean up whitespace

    return location_df


def clean_extract_and_filter_locations(df: pd.DataFrame, location_col: str) -> pd.DataFrame:
    """
    Cleans, extracts location details, filters, and transforms a DataFrame.

    This function will:
    1. Split a location column by '|' and create new rows for each location.
    2. Divide specified financial columns by the number of locations for that project.
    3. Extract 'wojewodztwo', 'powiat', and 'gmina' into separate columns.
    4. Filter out rows for non-Polish locations (containing 'oblast') or 'Cały Kraj'.

    Args:
        df (pd.DataFrame): The input DataFrame.
        location_col (str): The name of the column with pipe-separated locations.

    Returns:
        pd.DataFrame: A new, transformed and filtered DataFrame.
    """
    # --- 1. Identify Columns and Setup ---
    cols_to_divide = [
        'Total project value (PLN, for ETC projects EUR)',
        'Total eligible expenditure (PLN, for ETC projects EUR)',
        'Amount of EU co-financing (PLN, for ETC projects EUR)'
    ]

    # Validate that all necessary columns exist
    for col in cols_to_divide + [location_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame.")

    df_processed = df.copy()

    # --- 2. Split Locations and Calculate Divisor ---
    df_processed['location_list'] = df_processed[location_col].str.split('|')
    df_processed['location_count'] = df_processed['location_list'].apply(len)

    # --- 3. Explode DataFrame ---
    df_exploded = df_processed.explode('location_list')
    
    # Clean up the temporary location string
    df_exploded['location_list'] = df_exploded['location_list'].str.strip()

    # --- 4. Divide Financials (Done *before* filtering) ---
    for col in cols_to_divide:
        df_exploded[col] = pd.to_numeric(df_exploded[col], errors='coerce')
        df_exploded.loc[:, col] = df_exploded[col] / df_exploded['location_count']

    # --- 5. Extract Location Parts using the helper function ---
    location_details = extract_location_parts(df_exploded['location_list'])
    df_final = pd.concat([
        df_exploded.reset_index(drop=True),
        location_details.reset_index(drop=True)
    ], axis=1)

    # --- 6. Filter Out Unwanted Rows ---
    # Condition 1: Location is 'Cały Kraj'
    is_caly_kraj = df_final['location_list'] == 'Cały Kraj'

    # Condition 2: Location is non-Polish (contains 'oblast')
    non_polish_keywords = [
        'oblast', 'Východné Slovensko', 'Stredné Slovensko', 'Sachsen',
        'Lietuva', 'Mecklenburg-Vorpommern', 'Sydsverig', 'Sjalland',
        'Smaland med öarna', 'Hovedstaden',  'Lviv', 'Lutsk', 'Görlitz', 'Ternopil', 'Klaipėdos apskritis','Uzhhorod',
        'Bautzen','Rivne','Ivano-Frankivsk', 'Kovel', 'Østsjælland', 'Vest- og Sydsjælland', 'Mukachevo', 'Kalush',
        'Chervonohrad', 'Khust', 'Sarny', 'Kalmar län', 'Volodymyr-Volynskyi', 'Drohobych', 'Chortkiv', 'Telšių apskritis',
        'Nadvirna','Sambir','Varash','Kauno apskritis','Kronobergs län','Dubno','Yavoriv','Kremenets','Zolochiv','Berehiv',
        'Stryi','Rakhiv','Tyachiv','Tauragės apskritis']
    # Create a regex pattern by joining keywords with the OR operator '|'
    regex_pattern = '|'.join(non_polish_keywords)
    is_non_polish = df_final['location_list'].str.contains(
        regex_pattern, case=False, na=False, regex=True
    )

    # Combine conditions and keep rows that are NOT unwanted
    rows_to_keep = ~is_caly_kraj & ~is_non_polish
    df_final = df_final[rows_to_keep].copy()

    # --- 7. Final Cleanup ---
    # Drop intermediate columns
    df_final = df_final.drop(columns=[location_col, 'location_list', 'location_count'])

    return df_final


############# TREATMENT MATRIX #####################


def create_treatment_matrix(cleaned_df: pd.DataFrame, geo_level: str = 'powiat') -> pd.DataFrame:
    """
    Aggregates cleaned project data to create a treatment matrix, distributing
    voivodeship-level funds to their respective powiats.

    Args:
        cleaned_df (pd.DataFrame): The DataFrame after cleaning and exploding locations.
        geo_level (str): The geographic level for aggregation. Currently, the
                        distribution logic is implemented specifically for 'powiat'.

    Returns:
        pd.DataFrame: A DataFrame where each row is a unique geographic unit,
                    with aggregated and distributed financial data.
    """
    if geo_level != 'powiat':
        raise NotImplementedError("Distribution logic is currently only implemented for geo_level='powiat'.")

    # --- Define official powiat counts per voivodeship as a source of truth ---
    OFFICIAL_POWIAT_COUNT = {
        'dolnoslaskie': 30,
        'kujawsko-pomorskie': 23,
        'lubelskie': 24,
        'lubuskie': 14,
        'lodzkie': 24,
        'malopolskie': 22,
        'mazowieckie': 42,
        'opolskie': 12,
        'podkarpackie': 25,
        'podlaskie': 17,
        'pomorskie': 20,
        'slaskie': 36,
        'swietokrzyskie': 14,
        'warminsko-mazurskie': 21,
        'wielkopolskie': 35,
        'zachodniopomorskie': 21
    }

    # --- Define columns for aggregation ---
    financial_cols = [
        'Total project value (PLN, for ETC projects EUR)',
        'Total eligible expenditure (PLN, for ETC projects EUR)',
        'Amount of EU co-financing (PLN, for ETC projects EUR)'
    ]
    
    rate_col = 'Union co-financing rate (%)'
    cofinancing_val_col = 'Amount of EU co-financing (PLN, for ETC projects EUR)'

    # --- 1. Standardize names for reliable grouping and joining ---
    # Use a copy to avoid SettingWithCopyWarning
    df = cleaned_df.copy()
    df['wojewodztwo_std'] = df['wojewodztwo'].apply(standardize_polish_name)
    df['powiat_std'] = df['powiat'].apply(standardize_polish_name)

    # --- Prepare for Weighted Average Calculation ---
    if rate_col in df.columns and cofinancing_val_col in df.columns:
        df[rate_col] = pd.to_numeric(df[rate_col], errors='coerce')
        # This column will be the numerator for the weighted average
        df['rate_numerator'] = df[rate_col] * df[cofinancing_val_col]
        # Add numerator to the list of financial columns to be summed
        financial_cols_with_numerator = financial_cols + ['rate_numerator']
    else:
        financial_cols_with_numerator = financial_cols

    # --- 2. Separate data into two groups ---
    df_with_powiat = df.dropna(subset=['powiat_std']).copy()

    # --- 3. Calculate the base treatment matrix from data with known powiats ---
    agg_dict = {col: 'sum' for col in financial_cols_with_numerator}
    contract_col_name = 'Contracts number' if 'Contracts number' in df_with_powiat.columns else 'Contract number'
    agg_dict[contract_col_name] = pd.Series.nunique
    agg_dict['wojewodztwo'] = 'first'
    agg_dict['powiat'] = 'first'
    
    # Group by standardized name but keep original names in columns
    treatment_matrix = df_with_powiat.groupby('powiat_std').agg(agg_dict)
    treatment_matrix.index.name = 'powiat_std' # Rename index for clarity

    # --- Final Cleanup ---
    treatment_matrix.index.name = 'powiat_std'
    treatment_matrix = treatment_matrix.rename(columns={contract_col_name: 'number_of_unique_projects'})

    # Calculate the final weighted average co-financing rate
    if 'rate_numerator' in treatment_matrix.columns and cofinancing_val_col in treatment_matrix.columns:
        # Use .where to avoid division by zero
        treatment_matrix[rate_col] = treatment_matrix['rate_numerator'] / treatment_matrix[cofinancing_val_col].where(treatment_matrix[cofinancing_val_col] != 0)
        treatment_matrix.drop(columns=['rate_numerator'], inplace=True)

    treatment_matrix = treatment_matrix.sort_values(
        by='Amount of EU co-financing (PLN, for ETC projects EUR)',
        ascending=False
    )

    return treatment_matrix


def create_panel_treatment_matrix(cleaned_df: pd.DataFrame, start_date_col: str, end_date_col: str, all_powiats_path: str) -> pd.DataFrame:
    """
    Creates a panel treatment matrix, distributing project funds over time.

    This function assumes funds are distributed evenly across each year of a project's duration.
    It also handles the distribution of voivodeship-only funds to their constituent powiats on a per-year basis.

    Args:
        cleaned_df (pd.DataFrame): DataFrame after cleaning and location extraction.
        start_date_col (str): The column name for project start dates.
        end_date_col (str): The column name for project end dates.
        all_powiats_path (str): Path to the comprehensive CSV file of all powiats.

    Returns:
        pd.DataFrame: A panel DataFrame indexed by 'powiat' and 'year', showing annual funding.
    """
    # --- 1. Setup and Data Preparation ---
    df = cleaned_df.copy()
    
    # Ensure date columns are in datetime format
    df[start_date_col] = pd.to_datetime(df[start_date_col], errors='coerce')
    df[end_date_col] = pd.to_datetime(df[end_date_col], errors='coerce')

    # Drop rows where dates are essential but missing
    df.dropna(subset=[start_date_col, end_date_col], inplace=True)

    # Extract year and calculate project duration
    df['start_year'] = df[start_date_col].dt.year
    df['end_year'] = df[end_date_col].dt.year
    # Duration is number of years involved, e.g., 2020-2022 is 3 years.
    df['duration_years'] = (df['end_year'] - df['start_year']) + 1
    # Ensure duration is at least 1 to prevent division by zero
    df['duration_years'] = df['duration_years'].apply(lambda x: max(x, 1))
    
    rate_col = 'Union co-financing rate (%)'

    # --- 2. Calculate Annual Funding ---
    financial_cols = [
        'Total project value (PLN, for ETC projects EUR)',
        'Total eligible expenditure (PLN, for ETC projects EUR)',
        'Amount of EU co-financing (PLN, for ETC projects EUR)'
    ]
    cofinancing_val_col = 'Amount of EU co-financing (PLN, for ETC projects EUR)'

    annual_financial_cols = []
    for col in financial_cols:
        annual_col = f"annual_{col}"
        annual_financial_cols.append(annual_col)
        # Distribute total funding evenly across the project's duration
        df[annual_col] = df[col] / df['duration_years']

    # --- Prepare for Weighted Average Calculation ---
    if rate_col in df.columns and cofinancing_val_col in df.columns:
        df[rate_col] = pd.to_numeric(df[rate_col], errors='coerce')
        # This column will be the numerator for the weighted average
        annual_cofinancing_val_col = f"annual_{cofinancing_val_col}"
        df['rate_numerator'] = df[rate_col] * df[annual_cofinancing_val_col]

    # --- 3. Expand DataFrame to Panel Format ---
    # Create a new row for each year a project was active
    panel_rows = []
    for _, row in df.iterrows():
        for year in range(int(row['start_year']), int(row['end_year']) + 1):
            new_row = row.to_dict()
            new_row['year'] = year
            panel_rows.append(new_row)
    
    if not panel_rows:
        return pd.DataFrame() # Return empty if no valid projects

    panel_df = pd.DataFrame(panel_rows)

    # --- 4. Aggregate by Powiat and Year (using logic from create_treatment_matrix) ---
    # Standardize names for aggregation
    panel_df['wojewodztwo_std'] = panel_df['wojewodztwo'].apply(standardize_polish_name)
    panel_df['powiat_std'] = panel_df['powiat'].apply(standardize_polish_name)

    # Separate data
    df_with_powiat = panel_df.dropna(subset=['powiat_std']).copy()

    # Aggregate direct funding
    agg_dict = {col: 'sum' for col in annual_financial_cols}
    agg_dict['wojewodztwo'] = 'first'
    agg_dict['powiat'] = 'first' # Keep the original powiat name
    agg_dict['wojewodztwo_std'] = 'first'
    # Add sum for the weighted average numerator
    if 'rate_numerator' in df_with_powiat.columns:
        agg_dict['rate_numerator'] = 'sum'

    # Add aggregation for counting unique projects
    contract_col_name = None
    if 'Contract number' in df_with_powiat.columns:
        contract_col_name = 'Contract number'
    elif 'Contracts number' in df_with_powiat.columns:
        contract_col_name = 'Contracts number'
    
    if contract_col_name:
        agg_dict[contract_col_name] = pd.Series.nunique
    
    panel_matrix = df_with_powiat.groupby(['wojewodztwo_std', 'powiat_std', 'year']).agg(agg_dict)

    # --- 6. Final Cleanup ---
    panel_matrix.index.names = ['wojewodztwo_std', 'powiat_std', 'year']
    if 'wojewodztwo_std' in panel_matrix.columns:
        panel_matrix = panel_matrix.drop(columns=['wojewodztwo_std'])
    
    # Rename columns to remove 'annual_' prefix for clarity
    final_col_names = {old: new for old, new in zip(annual_financial_cols, financial_cols)}
    panel_matrix = panel_matrix.rename(columns=final_col_names)

    # Calculate the final weighted average co-financing rate
    if 'rate_numerator' in panel_matrix.columns and cofinancing_val_col in panel_matrix.columns:
        # Use .where to avoid division by zero
        panel_matrix[rate_col] = panel_matrix['rate_numerator'] / panel_matrix[cofinancing_val_col].where(panel_matrix[cofinancing_val_col] != 0)
        panel_matrix.drop(columns=['rate_numerator'], inplace=True)

    # Rename the project count column for clarity
    if contract_col_name:
        panel_matrix.rename(columns={contract_col_name: 'number_of_projects'}, inplace=True)

    # --- 7. Expand to a complete, balanced panel ---
    print("Expanding to a complete, balanced panel...")
    try:
        # Load the comprehensive list of all powiats from the user-provided path
        all_powiats_df = pd.read_csv(all_powiats_path, delimiter=';', header=4)
        # This assumes the name columns in your TERC file are 'NAZWA' and 'NAZWA_WOJ'. Adjust if different.
        powiat_name_col = 'NAZWA' 
        woj_name_col = 'NAZWA_WOJ'
        if woj_name_col not in all_powiats_df.columns:
            raise KeyError(f"The reference TERC file is missing a voivodeship name column. Expected '{woj_name_col}'.")

        all_powiats_df['wojewodztwo_std'] = all_powiats_df[woj_name_col].apply(standardize_polish_name)
        all_powiats_df['powiat_std'] = all_powiats_df[powiat_name_col].apply(standardize_polish_name)
        
        # Create a list of unique (wojewodztwo_std, powiat_std) tuples
        all_powiats_tuples = all_powiats_df[['wojewodztwo_std', 'powiat_std']].drop_duplicates().to_records(index=False)
        
        print(f"Successfully loaded {len(all_powiats_tuples)} unique powiats from reference file.")
    except Exception as e:
        print(f"ERROR: Could not load or process the comprehensive powiats file at '{all_powiats_path}'.")
        print(f"Details: {e}")
        print("Returning the sparse panel instead.")
        return panel_matrix.sort_index()

    # Determine the full range of years from the data, handle case where panel_matrix is empty
    if not panel_matrix.empty:
        min_year = int(panel_matrix.index.get_level_values('year').min())
        max_year = int(panel_matrix.index.get_level_values('year').max())
        all_years = range(min_year, max_year + 1)
    else: # If no projects, create an empty panel for a default year range
        all_years = range(pd.Timestamp.now().year - 10, pd.Timestamp.now().year + 1)

    # Create the complete MultiIndex
    # Create a DataFrame with all combinations of powiats and years
    all_combinations = pd.MultiIndex.from_product([all_powiats_tuples, all_years], names=['powiat_info', 'year']).to_frame(index=False)
    all_combinations[['wojewodztwo_std', 'powiat_std']] = pd.DataFrame(all_combinations['powiat_info'].tolist(), index=all_combinations.index)
    all_combinations = all_combinations.drop(columns='powiat_info').set_index(['wojewodztwo_std', 'powiat_std', 'year'])

    # Join the actual funding data onto this complete grid
    complete_panel = all_combinations.join(panel_matrix).fillna(0)

    # For the new rows, fill the original 'powiat' name from a map
    name_mapper = all_powiats_df[['wojewodztwo_std', 'powiat_std', woj_name_col, powiat_name_col]].drop_duplicates(subset=['wojewodztwo_std', 'powiat_std'])
    complete_panel = complete_panel.reset_index().merge(name_mapper, on=['wojewodztwo_std', 'powiat_std'], how='left').set_index(['wojewodztwo_std', 'powiat_std', 'year'])
    complete_panel.rename(columns={woj_name_col: 'wojewodztwo', powiat_name_col: 'powiat'}, inplace=True)

    return complete_panel.sort_index()


def flatten_and_rename_panel(panel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Flattens a panel DataFrame by resetting its index and renames columns
    to a standardized, snake_case format for easier analysis.

    Args:
        panel_df (pd.DataFrame): A DataFrame with a MultiIndex (e.g., from 
                                create_panel_treatment_matrix).

    Returns:
        pd.DataFrame: A flattened DataFrame with cleaned column names.
    """
    if not isinstance(panel_df.index, pd.MultiIndex):
        print("Warning: Input DataFrame does not have a MultiIndex. "
            "Flattening may not be necessary.")

    # --- 1. Flatten the DataFrame ---
    # This converts the 'powiat' and 'year' index levels into columns.
    flat_df = panel_df.reset_index()

    # --- 2. Define column mapping ---
    rename_map = {
        'Total project value (PLN, for ETC projects EUR)': 'total_project_value',
        'Total eligible expenditure (PLN, for ETC projects EUR)': 'total_eligible_expenditure',
        'Amount of EU co-financing (PLN, for ETC projects EUR)': 'cofinancing_value',
        'Union co-financing rate (%)': 'eu_cofinancing_rate',
        'Project implemented under competitive or non-competitive procedure': 'is_competitive',
        'Funding completed': 'is_completed',
        'number_of_unique_projects': 'project_count'
    }

    # --- 3. Rename the columns ---
    # The .rename() method will ignore any keys in the map that are not found in the DataFrame's columns.
    renamed_df = flat_df.rename(columns=rename_map)

    return renamed_df
