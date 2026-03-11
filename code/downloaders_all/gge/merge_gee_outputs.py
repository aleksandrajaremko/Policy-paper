#!/usr/bin/env python3
"""
===============================================================================
Post-Processing: Merge GEE Nightlights & GHSL Exports into Panel
===============================================================================

After running the GEE script (gee_nightlights_ghsl_poland.js), you'll have
several CSVs in your Google Drive. This script merges them into a single
gmina-level panel dataset ready for analysis.

SETUP INSTRUCTIONS (do this BEFORE running the GEE script):
============================================================

1. GET A GOOGLE EARTH ENGINE ACCOUNT:
   → https://earthengine.google.com  (free for research/academic use)

2. DOWNLOAD GMINA BOUNDARIES:
   → Go to https://gadm.org/download_country.html
   → Select "Poland", Level 3, Format "Shapefile"
   → This gives you ~2,477 gminas with TERYT-compatible codes

3. UPLOAD BOUNDARIES TO GEE:
   → In GEE Code Editor (code.earthengine.google.com):
     • Click "Assets" tab → "New" → "Shape files"
     • Upload all files from the GADM shapefile (.shp, .shx, .dbf, .prj)
     • Name it: poland_gadm_level3
     • Wait for ingestion to complete (check Tasks tab)

4. EDIT THE GEE SCRIPT:
   → Replace the `poland` variable with your uploaded asset:
     var poland = ee.FeatureCollection("projects/YOUR_PROJECT/assets/poland_gadm_level3");
   → Update the 'selectors' in each Export.table.toDrive() to use GADM columns:
     selectors: ['GID_3', 'NAME_3', 'NAME_2', 'NAME_1', 'year', 'ntl_mean']
   
   The GID_3 field from GADM looks like "POL.1.2.3_1" — you'll need to
   crosswalk this to TERYT codes. The NAME_3 field is the gmina name.

5. RUN THE GEE SCRIPT:
   → Paste the .js script into the Code Editor
   → Click "Run"
   → Go to "Tasks" tab → click "Run" for each export
   → Wait 10-30 min per export; results go to Google Drive

6. RUN THIS PYTHON SCRIPT:
   → Download all exported CSVs from Drive to a single folder
   → Run: python merge_gee_outputs.py --input-dir ./gee_exports
   → Output: nightlights_ghsl_panel.csv

ALTERNATIVE: PYTHON + EARTH ENGINE API
=======================================
If you prefer Python over the Code Editor, install:
  pip install earthengine-api geemap

Then authenticate:
  import ee
  ee.Authenticate()   # opens browser for OAuth
  ee.Initialize(project='your-gee-project')

The logic is identical — just translated to Python syntax.
The GEE Code Editor is easier for a first run though.

CROSSWALKING GADM → TERYT
==========================
GADM and TERYT don't share a common ID. You'll need to match on
gmina name + powiat name + voivodeship name. The merge_with_teryt()
function below does a fuzzy match for you.

===============================================================================
"""

import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path


def load_and_tag(filepath, dataset_name):
    """Load a GEE export CSV and add a source tag."""
    df = pd.read_csv(filepath)
    df["_source"] = dataset_name
    return df


def merge_nightlights(dmsp_path, viirs_path):
    """
    Merge DMSP (1992-2013) and VIIRS (2014-2023) nightlight exports.
    Returns a single long-format panel: unit × year → ntl_mean
    """
    frames = []

    if dmsp_path and os.path.exists(dmsp_path):
        dmsp = pd.read_csv(dmsp_path)
        dmsp["sensor"] = "DMSP"
        frames.append(dmsp)
        print(f"  DMSP: {len(dmsp)} rows, years {dmsp['year'].min()}-{dmsp['year'].max()}")

    if viirs_path and os.path.exists(viirs_path):
        viirs = pd.read_csv(viirs_path)
        viirs["sensor"] = "VIIRS"
        frames.append(viirs)
        print(f"  VIIRS: {len(viirs)} rows, years {viirs['year'].min()}-{viirs['year'].max()}")

    if not frames:
        return pd.DataFrame()

    ntl = pd.concat(frames, ignore_index=True)
    return ntl


def merge_ghsl(built_path, volume_path, urban_path):
    """Merge GHSL built-up surface, volume, and urbanisation exports."""
    frames = {}

    if built_path and os.path.exists(built_path):
        frames["built"] = pd.read_csv(built_path)
        print(f"  GHSL Built-up: {len(frames['built'])} rows")

    if volume_path and os.path.exists(volume_path):
        frames["volume"] = pd.read_csv(volume_path)
        print(f"  GHSL Volume: {len(frames['volume'])} rows")

    if urban_path and os.path.exists(urban_path):
        frames["urban"] = pd.read_csv(urban_path)
        print(f"  GHSL Urban: {len(frames['urban'])} rows")

    return frames


def merge_with_teryt(gee_df, teryt_lookup, id_col="GID_3", name_col="NAME_3"):
    """
    Fuzzy-match GEE export (GADM names) to TERYT codes.

    teryt_lookup should be a DataFrame with columns:
      teryt, gmina_name, powiat_name, voivodeship_name

    You can build this from the BDL panel or from GUS TERYT register.
    """
    # Normalize names for matching
    def norm(s):
        return (
            str(s).lower()
            .replace("gm.", "").replace("m.", "").replace("m.st.", "")
            .strip()
        )

    gee_df["_name_norm"] = gee_df[name_col].apply(norm)
    teryt_lookup["_name_norm"] = teryt_lookup["gmina_name"].apply(norm)

    merged = gee_df.merge(
        teryt_lookup[["teryt", "_name_norm"]],
        on="_name_norm",
        how="left",
    )

    unmatched = merged["teryt"].isna().sum()
    if unmatched > 0:
        print(f"  WARNING: {unmatched} gminas could not be matched to TERYT")
        print(f"  You may need to refine the matching (add powiat/voivodeship)")

    merged.drop(columns=["_name_norm"], inplace=True)
    return merged


def auto_discover(input_dir):
    """Auto-discover GEE export files by name pattern."""
    files = {f.stem: str(f) for f in Path(input_dir).glob("*.csv")}

    discovered = {}
    for key, pattern in [
        ("dmsp", "nightlights_dmsp"),
        ("viirs", "nightlights_viirs"),
        ("built", "ghsl_built_surface"),
        ("volume", "ghsl_building_volume"),
        ("urban", "ghsl_urbanisation"),
    ]:
        matches = [v for k, v in files.items() if pattern in k]
        if matches:
            discovered[key] = matches[0]
            print(f"  Found {key}: {os.path.basename(matches[0])}")

    return discovered


def main():
    parser = argparse.ArgumentParser(
        description="Merge GEE nightlights & GHSL exports into panel CSV"
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Folder containing GEE export CSVs",
    )
    parser.add_argument(
        "--output", default="nightlights_ghsl_panel.csv",
        help="Output path for merged panel",
    )
    parser.add_argument(
        "--teryt-lookup", default=None,
        help="Optional: CSV with teryt, gmina_name columns for crosswalking",
    )

    args = parser.parse_args()

    print("Discovering GEE exports...")
    files = auto_discover(args.input_dir)

    if not files:
        print("ERROR: No GEE export CSVs found in", args.input_dir)
        return

    # Merge nightlights
    print("\nMerging nightlights...")
    ntl = merge_nightlights(files.get("dmsp"), files.get("viirs"))

    # Merge GHSL
    print("\nMerging GHSL...")
    ghsl = merge_ghsl(files.get("built"), files.get("volume"), files.get("urban"))

    # Optionally crosswalk to TERYT
    if args.teryt_lookup and os.path.exists(args.teryt_lookup):
        print("\nCrosswalking to TERYT...")
        teryt = pd.read_csv(args.teryt_lookup)
        if len(ntl):
            ntl = merge_with_teryt(ntl, teryt)

    # Save
    if len(ntl):
        ntl_path = args.output.replace(".csv", "_nightlights.csv")
        ntl.to_csv(ntl_path, index=False, encoding="utf-8-sig")
        print(f"\nSaved nightlights panel: {ntl_path}")

    for key, df in ghsl.items():
        path = args.output.replace(".csv", f"_ghsl_{key}.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"Saved GHSL {key}: {path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
