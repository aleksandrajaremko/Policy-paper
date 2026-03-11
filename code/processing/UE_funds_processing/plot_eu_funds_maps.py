# =============================================================================
# EU FUNDS PER CAPITA — YEAR-BY-YEAR GMINA MAPS
# Run this in your notebook after loading the master panel
# =============================================================================

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# --- CONFIG ---
SHP_PATH = r"C:\Users\jarem\OneDrive - London School of Economics\YEAR 2\1. Policy paper\policy-paper-repo\data\inputs\shapefiles\polska\gminy\gminy.shp"
MASTER_PATH = r"data\clean\treatment\eu_flows\final\eu_funds_gmina_panel_master.csv"
# Or use the variable `master` if already loaded in notebook

# --- LOAD ---
gminy_shp = gpd.read_file(SHP_PATH)
master = pd.read_csv(MASTER_PATH, encoding='utf-8-sig')

# --- PREP: 6-digit match (handles TERYT type reclassifications) ---
master['gmina_id_6'] = master['gmina_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(7).str[:6]
gminy_shp['gmina_id_6'] = gminy_shp['JPT_KOD_JE'].astype(str).str.zfill(7).str[:6]

# --- FIGURE: Small multiples, 2007-2024 ---
years = list(range(2007, 2025))
ncols = 6
nrows = 3

fig, axes = plt.subplots(nrows, ncols, figsize=(24, 14))
axes = axes.flatten()

# Fixed color scale across all years for comparability
vmin = 0
vmax = 3000  # PLN per capita — adjust if needed

for i, year in enumerate(years):
    ax = axes[i]
    
    # Get per-capita for this year
    yr_data = master[master['Year'] == year][['gmina_id_6', 'EU_subsidy_per_capita_PLN']].copy()
    
    # If multiple rows per gmina_id_6 (shouldn't happen but just in case), take sum
    yr_data = yr_data.groupby('gmina_id_6')['EU_subsidy_per_capita_PLN'].sum().reset_index()
    
    # Merge
    merged = gminy_shp.merge(yr_data, on='gmina_id_6', how='left')
    
    # Plot
    merged.plot(
        column='EU_subsidy_per_capita_PLN',
        cmap='YlOrRd',
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        missing_kwds={'color': 'lightgrey'},
        linewidth=0.05,
        edgecolor='grey',
    )
    
    # Total for annotation
    total_bln = master[master['Year'] == year]['EU_subsidy_PLN'].sum() / 1e9
    ax.set_title(f'{year}\n({total_bln:.1f} bln PLN)', fontsize=10, fontweight='bold')
    ax.axis('off')

# Remove empty axes
for j in range(len(years), len(axes)):
    axes[j].set_visible(False)

# Colorbar
sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.02, pad=0.04, aspect=50)
cbar.set_label('EU Cohesion Fund Subsidy per Capita (PLN)', fontsize=12)

fig.suptitle('EU Cohesion Fund Allocation per Capita by Gmina (2007–2024)', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('eu_funds_per_capita_year_maps.png', dpi=200, bbox_inches='tight')
plt.show()
print("Map saved: eu_funds_per_capita_year_maps.png")


# =============================================================================
# BONUS: Total cumulative per capita map (all years summed)
# =============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 12))

cumulative = master.groupby('gmina_id_6')['EU_subsidy_per_capita_PLN'].sum().reset_index()
merged_cum = gminy_shp.merge(cumulative, on='gmina_id_6', how='left')

merged_cum.plot(
    column='EU_subsidy_per_capita_PLN',
    cmap='YlOrRd',
    ax=ax,
    legend=True,
    legend_kwds={'label': 'Cumulative EU Subsidy per Capita (PLN, 2007-2024)', 'shrink': 0.6},
    missing_kwds={'color': 'lightgrey'},
    linewidth=0.1,
    edgecolor='grey',
)
ax.set_title('Cumulative EU Cohesion Fund per Capita by Gmina (2007–2024)', fontsize=14, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig('eu_funds_per_capita_cumulative_map.png', dpi=200, bbox_inches='tight')
plt.show()
print("Map saved: eu_funds_per_capita_cumulative_map.png")
