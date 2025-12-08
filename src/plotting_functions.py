import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd


# Import the standardization function from your other file
from .functions import standardize_polish_name

powiats = gpd.read_parquet(r'data/clean/geodata/powiaty.parquet')
# Create the standardized column on the fly
powiats['powiat_std'] = powiats['JPT_NAZWA_'].apply(standardize_polish_name)

# Create a complete label map from the geodataframe
COMPLETE_LABEL_MAP = powiats.set_index('powiat_std')['JPT_NAZWA_']
ALL_POWIATS_STD_LIST = sorted(powiats['powiat_std'].unique())

def plot_treatment_heatmap(
    panel_df: pd.DataFrame, 
    start_year: int = 2014, 
    end_year: int = 2029
):
    """
    Generates a heatmap to visualize project funding of powiats over time. This heavy-duty
    debug version uses log-scaling to make even small values visible.

    Args:
        panel_df (pd.DataFrame): A flattened panel DataFrame with 'powiat', 'powiat_std', 
                                'year', and 'cofinancing_value' columns.
        start_year (int): The first year for the x-axis.
        end_year (int): The last year for the x-axis.
    """
    required_cols = ['powiat', 'powiat_std', 'year', 'cofinancing_value']
    if not all(col in panel_df.columns for col in required_cols):
        raise ValueError(f"Input DataFrame must contain {required_cols} columns.")

    print("Preparing data for plotting...")
    
    df = panel_df.copy()
    
    print("\n--- Input Data Sanity Check ---")
    print("Statistics for 'cofinancing_value':")
    print(df['cofinancing_value'].describe())
    if df['cofinancing_value'].sum() == 0:
        print("CRITICAL: The 'cofinancing_value' column in your input data is all zeros. No data to plot.")
        return

    # Pivot on the actual cofinancing_value
    treatment_matrix = df.pivot_table(
        index='powiat_std', 
        columns='year', 
        values='cofinancing_value', # Use the continuous value
        aggfunc='sum' # Sum up funding if multiple projects in a year
    )

    print("\n--- Post-Pivot Sanity Check ---")
    non_zero_cells = (treatment_matrix > 0).sum().sum()
    print(f"Number of non-zero cells in the matrix before reindexing: {non_zero_cells}")
    if non_zero_cells == 0:
        print("CRITICAL: The matrix is all zeros after pivoting. Check your input 'powiat_std' and 'year' columns.")

    # --- Diagnostic Check ---
    project_powiats_std = set(treatment_matrix.index)
    geo_powiats_std = set(ALL_POWIATS_STD_LIST)
    intersection = project_powiats_std.intersection(geo_powiats_std)

    print("\n--- Debugging Information ---")
    print(f"Found {len(project_powiats_std)} unique standardized powiats in your project data.")
    print(f"Found {len(geo_powiats_std)} unique standardized powiats in the reference geodata file.")
    print(f"Number of matching powiats between sources: {len(intersection)}")

    if len(intersection) == 0:
        print("\nCRITICAL ERROR: There are no matching powiat names between your project data and the geodata file.")
        print("This is why the plot is empty. Please check for inconsistencies in the source names or the standardization logic.")
        print("\nExamples from project data (standardized):", list(project_powiats_std)[:5])
        print("Examples from geodata file (standardized):", list(geo_powiats_std)[:5])
        return # Stop execution to prevent plotting an empty graph

    all_powiats_std = ALL_POWIATS_STD_LIST
    all_years = range(start_year, end_year + 1)

    # Reindex first using the full standardized list to get the full matrix
    reindexed_matrix = treatment_matrix.reindex(index=all_powiats_std, columns=all_years, fill_value=0)

    print("\n--- Post-Reindex Sanity Check ---")
    print(f"Matrix shape after reindexing: {reindexed_matrix.shape}")
    reindex_sum = reindexed_matrix.sum().sum()
    print(f"Sum of all values after reindexing: {reindex_sum:,.2f}")
    if reindex_sum == 0 and non_zero_cells > 0:
        print("CRITICAL: Data was lost during reindexing. This indicates the name matching failed despite the diagnostic check.")

    # Now, rename the index of the full matrix using the complete label map
    renamed_matrix = reindexed_matrix.rename(index=COMPLETE_LABEL_MAP)

    print("\n--- Post-Rename Sanity Check ---")
    print(f"Matrix shape after renaming: {renamed_matrix.shape}")
    rename_sum = renamed_matrix.sum().sum()
    print(f"Sum of all values after renaming: {rename_sum:,.2f}")
    if rename_sum == 0 and reindex_sum > 0:
        print("CRITICAL: Data was lost during the .rename() operation. This points to an issue with COMPLETE_LABEL_MAP.")

    print(f"Plotting matrix with shape: {treatment_matrix.shape} (Powiats x Years)")

    # --- Plotting ---
    num_powiats = len(treatment_matrix.index)
    fig_height = max(10, num_powiats / 10)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(20, fig_height))

    # --- Final Plot Styling ---
    # Apply a log transform to the data to make small values visible.
    # np.log1p is used because it handles zeros correctly (log(1+x)).
    plot_data = np.log1p(renamed_matrix)
    
    # Use a perceptually uniform colormap
    cmap = "viridis"

    sns.heatmap(
        plot_data,
        cmap=cmap,
        linewidths=0.5,
        linecolor='lightgray',
        cbar=True,  # Show the color bar to see the funding scale
        ax=ax
    )

    ax.set_title('Temporal Heatmap of Powiat Funding (Log Scale)', fontsize=20, pad=20)
    ax.set_xlabel('Year of Treatment', fontsize=14)
    ax.set_ylabel('Powiat', fontsize=14)
    
    # Set x-axis ticks and labels
    ax.set_xticks(ax.get_xticks()) # Avoids a UserWarning
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # Adjust y-tick font size for readability with many powiats
    ax.tick_params(axis='y', labelsize=8 if num_powiats > 100 else 10)

    plt.tight_layout()
    plt.show()
