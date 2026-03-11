"""
Powiat-Level Populist Vote Share Panel Builder
===============================================
Builds a panel dataset of populist/far-right vote shares at powiat level
from PKW election data + PopuList 3.0 classifications.

Save this as: powiat_panel_builder.py
Run with:    python powiat_panel_builder.py
"""

import os
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# PATHS — only thing you need to change
# ============================================================
populist_path = Path(r"C:\Users\jarem\OneDrive - London School of Economics\YEAR 2\1. Policy paper\policy-paper-repo\data\inputs\election_data\Populists_dataset\The Populist 3.0.csv")
raw_path      = Path(r"C:\Users\jarem\OneDrive - London School of Economics\YEAR 2\1. Policy paper\policy-paper-repo\data\clean\outcome\Elections\raw_powiat")
output_path   = Path(r"C:\Users\jarem\OneDrive - London School of Economics\YEAR 2\1. Policy paper\policy-paper-repo\data\clean\outcome\Elections")

YEARS = [2001, 2005, 2007, 2011, 2015, 2019, 2023]

# ============================================================
# FILE SPECS — filenames and column names per year
# ============================================================
FILE_SPECS = {
    2001: {
        "file":       "sejm2001-lis-pow.xls",
        "format":     "xls",
        "teryt_col":  "TERYT",
        "powiat_col": "Powiat",
        "valid_col":  "Ważne",
    },
    2005: {
        "file":       "1456225675_36797.xls",   # _36797 = counts, _36798 = percentages
        "format":     "xls",
        "teryt_col":  "TERYT",
        "powiat_col": "Powiat",
        "valid_col":  "Głosy ważne",
    },
    2007: {
        "file":       "sejm2007-pow-listy.xls",
        "format":     "xls",
        "teryt_col":  "Kod pow.",
        "powiat_col": "Powiat",
        "valid_col":  "Ważne",
    },
    2011: {
        "file":       "2011-sejm-pow-listy.xls",
        "format":     "xls",
        "teryt_col":  "TERYT",
        "powiat_col": "Powiat",
        "valid_col":  "Głosy ważne",
    },
    2015: {
        "file":       "2015-gl-lis-pow.zip",    # counts, not proc
        "format":     "zip_xls",
        "teryt_col":  "TERYT",
        "powiat_col": "Powiat",
        "valid_col":  "Głosy ważne",
    },
    2019: {
        "file":       "wyniki_gl_na_listy_po_powiatach_sejm_csv.zip",
        "format":     "zip_csv",
        "teryt_col":  "Kod TERYT",
        "powiat_col": "Powiat",
        "valid_col":  "Liczba głosów ważnych oddanych łącznie na wszystkie listy kandydatów",
    },
    2023: {
        "file":       "wyniki_gl_na_listy_po_powiatach_sejm_csv.zip",
        "format":     "zip_csv",
        "teryt_col":  "Kod TERYT",
        "powiat_col": "Powiat",
        "valid_col":  "Liczba głosów ważnych oddanych łącznie na wszystkie listy kandydatów",
    },
}

# ============================================================
# PARTY MATCHING PATTERNS
# ============================================================
# Keywords that must ALL appear in the PKW column name to match a party.
# Case-insensitive. Based on actual column names inspected from your files.

PARTY_PATTERNS = {
    "PiS":          ["Prawo i Sprawiedliwo"],
    "Konfederacja": ["Konfederacja"],
    "KORWiN":       ["Korwin"],
    "LPR":          ["Liga Polskich Rodzin"],
    "SO":           ["Samoobrona"],
    "Kukiz '15":    ["Kukiz"],
    "RN":           ["Ruch Narodowy"],
    "SP":           ["Solidarna Polska"],
    "KNP":          ["Kongres Nowej Prawicy"],
    "ZChN":         ["Zjednoczenie Chrze"],
    "ROP":          ["Ruch Odbudowy Polski"],
}

# Metadata column keywords — these are NOT party vote columns
META_KEYWORDS = [
    "teryt", "kod", "powiat", "okręg", "nr okr", "nr\nokr",
    "województwo", "upr", "uprawn", "frekw", "odd.", "ważne",
    "głosy ważne", "głosy nieważne", "karty", "liczba", "komisj",
    "zaświadcz", "kopert", "pakiet", "w tym", "pełnomocn",
    "otrzymane", "wydane", "niewykorzys", "wyjęte", "nieważn",
    "wysłan", "zwrotn", "typ", "l. upr", "gł. odd"
]


# ============================================================
# LOAD POPULIST
# ============================================================
def load_populist():
    df = pd.read_csv(populist_path, sep=";", encoding="utf-8-sig")
    df = df[df["country_name"] == "Poland"].copy()
    print(f"PopuList: {len(df)} Polish parties loaded")
    return df

pop_pl = load_populist()


# ============================================================
# GET POPULIST TAGS FOR A YEAR
# ============================================================
def get_tags(year: int) -> dict:
    """
    Returns dict: party_short -> classification flags active in `year`.
    Uses PopuList start/end year logic and borderline flag.
    """
    tags = {}
    for _, row in pop_pl.iterrows():
        short = row["party_name_short"].strip()
        entry = {}
        for measure in ["populist", "farright", "farleft", "eurosceptic"]:
            val    = int(row[measure])
            start  = int(row[f"{measure}_start"])
            end    = int(row[f"{measure}_end"])
            bl     = int(row[f"{measure}_bl"])
            active = (val == 1) and (start <= year <= end)
            entry[measure]             = active           # incl. borderline
            entry[f"{measure}_strict"] = active and not bl  # excl. borderline
        tags[short] = entry
    return tags


# ============================================================
# MATCH COLUMN TO PARTY
# ============================================================
def match_column(col_name: str) -> list:
    col_lower = col_name.lower()
    return [p for p, kws in PARTY_PATTERNS.items()
            if all(kw.lower() in col_lower for kw in kws)]


# ============================================================
# LOAD RAW FILE
# ============================================================
def load_raw(year: int) -> pd.DataFrame:
    spec     = FILE_SPECS[year]
    filepath = raw_path / str(year) / spec["file"]

    if not filepath.exists():
        raise FileNotFoundError(
            f"File not found: {filepath}\n"
            f"Check the scraper downloaded the file correctly."
        )

    fmt = spec["format"]

    if fmt == "zip_csv":
        with zipfile.ZipFile(filepath) as zf:
            csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
            with zf.open(csv_name) as f:
                return pd.read_csv(f, sep=";", encoding="utf-8-sig",
                                   dtype=str, low_memory=False)

    elif fmt == "zip_xls":
        with zipfile.ZipFile(filepath) as zf:
            xls_name = [n for n in zf.namelist()
                        if n.endswith((".xls", ".xlsx"))][0]
            with zf.open(xls_name) as f:
                return pd.read_excel(f, dtype=str)

    elif fmt in ("xls", "xlsx"):
        return pd.read_excel(filepath, dtype=str)

    else:
        raise ValueError(f"Unknown format: {fmt}")


# ============================================================
# CLEAN TERYT
# ============================================================
def clean_teryt(series: pd.Series) -> pd.Series:
    """
    Standardise to 4-digit powiat TERYT.
    Filters out overseas rows, aggregate totals, and NaN values.
    """
    s = (series.astype(str)
               .str.strip()
               .str.replace(r"\.0$", "", regex=True)
               .str.zfill(4))
    # Valid powiat TERYT = exactly 4 digits
    return s.where(s.str.match(r"^\d{4}$"), other=pd.NA)


# ============================================================
# GET PARTY COLUMNS
# ============================================================
def get_party_columns(df: pd.DataFrame) -> dict:
    """
    Returns dict: col_name -> [matching party shorts].
    Skips metadata columns. Empty list = unmatched mainstream party.
    """
    result = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if any(meta in col_lower for meta in META_KEYWORDS):
            continue
        result[col] = match_column(col)
    return result


# ============================================================
# BUILD SINGLE YEAR PANEL
# ============================================================
def build_year(year: int, verbose: bool = True):
    spec = FILE_SPECS[year]
    df   = load_raw(year)
    tags = get_tags(year)

    # --- Validate key columns ---
    for col_key, col_name in [("teryt_col",  spec["teryt_col"]),
                               ("valid_col",  spec["valid_col"])]:
        if col_name not in df.columns:
            print(f"\n  ERROR {year}: column '{col_name}' not found")
            print(f"  Available columns: {df.columns.tolist()}")
            print(f"  Fix: update FILE_SPECS[{year}]['{col_key}']")
            return None

    # --- Clean TERYT, drop overseas/aggregate rows ---
    df["_teryt"] = clean_teryt(df[spec["teryt_col"]])
    df = df[df["_teryt"].notna()].copy()

    # --- Valid votes ---
    df["_valid"] = pd.to_numeric(
        df[spec["valid_col"]].astype(str).str.replace(",", ".").str.strip(),
        errors="coerce"
    )

    # --- Identify and convert party columns ---
    party_cols = get_party_columns(df)
    for col in party_cols:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", ".").str.strip(),
            errors="coerce"
        ).fillna(0)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  {year} — {len(df)} powiats")
        print(f"{'='*60}")
        matched   = {c: v for c, v in party_cols.items() if v}
        unmatched = [c for c, v in party_cols.items() if not v]
        print(f"  Matched party columns ({len(matched)}):")
        for col, parties in matched.items():
            print(f"    {col[:65]:65s} → {parties}")
        print(f"  Unmatched / mainstream ({len(unmatched)}):")
        for col in unmatched:
            print(f"    {col[:80]}")

    # --- Vote share computation ---
    def col_sum(tag: str, strict: bool = False) -> pd.Series:
        key  = f"{tag}_strict" if strict else tag
        cols = list(set(
            col for col, parties in party_cols.items()
            for p in parties
            if tags.get(p, {}).get(key, False)
        ))
        if not cols:
            return pd.Series(0.0, index=df.index)
        return df[cols].sum(axis=1)

    valid = df["_valid"].replace(0, np.nan)

    # Build panel
    powiat_name_col = spec["powiat_col"]
    panel = pd.DataFrame({
        "teryt_powiat":  df["_teryt"].values,
        "powiat_name":   df[powiat_name_col].values if powiat_name_col in df.columns else np.nan,
        "year":          year,
        "valid_votes":   valid.values,

        "votes_populist":          col_sum("populist",   strict=False).values,
        "votes_populist_strict":   col_sum("populist",   strict=True).values,
        "votes_farright":          col_sum("farright",   strict=False).values,
        "votes_farright_strict":   col_sum("farright",   strict=True).values,
        "votes_farleft":           col_sum("farleft",    strict=False).values,
        "votes_eurosceptic":       col_sum("eurosceptic",strict=False).values,
    })

    # Vote shares (proportions 0-1)
    for measure in ["populist", "populist_strict",
                    "farright", "farright_strict",
                    "farleft",  "eurosceptic"]:
        panel[f"share_{measure}"] = panel[f"votes_{measure}"] / valid.values

    # Union: populist OR far-right (no double-counting)
    pop_cols   = [c for c, ps in party_cols.items()
                  for p in ps if tags.get(p, {}).get("populist", False)]
    fr_cols    = [c for c, ps in party_cols.items()
                  for p in ps if tags.get(p, {}).get("farright", False)]
    union_cols = list(set(pop_cols) | set(fr_cols))
    panel["votes_populist_or_farright"] = (
        df[union_cols].sum(axis=1).values if union_cols else 0.0
    )
    panel["share_populist_or_farright"] = (
        panel["votes_populist_or_farright"] / valid.values
    )

    if verbose:
        print(f"\n  Mean vote shares:")
        print(f"    populist         (incl. borderline PiS): {panel['share_populist'].mean():.3f}")
        print(f"    populist_strict  (excl. borderline PiS): {panel['share_populist_strict'].mean():.3f}")
        print(f"    farright         (incl. borderline PiS): {panel['share_farright'].mean():.3f}")
        print(f"    farright_strict  (excl. borderline PiS): {panel['share_farright_strict'].mean():.3f}")
        print(f"    populist_or_farright:                    {panel['share_populist_or_farright'].mean():.3f}")

    return panel


# ============================================================
# BUILD CROSSWALK
# ============================================================
def build_crosswalk() -> pd.DataFrame:
    rows = []
    for year in YEARS:
        try:
            df   = load_raw(year)
            spec = FILE_SPECS[year]
            tags = get_tags(year)
            df["_teryt"] = clean_teryt(df[spec["teryt_col"]])
            df = df[df["_teryt"].notna()]
            for col, parties in get_party_columns(df).items():
                if parties:
                    for p in parties:
                        t = tags.get(p, {})
                        rows.append({
                            "year":         year,
                            "pkw_column":   col,
                            "party_short":  p,
                            "populist":     t.get("populist"),
                            "populist_bl":  t.get("populist") and not t.get("populist_strict"),
                            "farright":     t.get("farright"),
                            "farright_bl":  t.get("farright") and not t.get("farright_strict"),
                        })
                else:
                    rows.append({
                        "year": year, "pkw_column": col,
                        "party_short": "— mainstream —",
                        "populist": False, "populist_bl": False,
                        "farright": False, "farright_bl": False,
                    })
        except Exception as e:
            print(f"  Crosswalk skipped {year}: {e}")
    return pd.DataFrame(rows)


# ============================================================
# MAIN — build full panel
# ============================================================
if __name__ == "__main__":

    all_panels = []

    for year in YEARS:
        try:
            panel = build_year(year, verbose=True)
            if panel is not None:
                all_panels.append(panel)
        except FileNotFoundError as e:
            print(f"\nSKIPPED {year}: {e}")
        except Exception as e:
            print(f"\nERROR {year}: {e}")
            raise

    if not all_panels:
        print("\nNo panels built. Check raw_path points to raw_powiat folder.")
        print(f"Current raw_path: {raw_path}")
        raise SystemExit(1)

    panel = pd.concat(all_panels, ignore_index=True).sort_values(
        ["teryt_powiat", "year"]
    ).reset_index(drop=True)

    print(f"\n{'='*60}")
    print(f"  POWIAT PANEL COMPLETE")
    print(f"{'='*60}")
    print(f"  Rows:           {len(panel)}")
    print(f"  Unique powiats: {panel['teryt_powiat'].nunique()}")
    print(f"  Years:          {sorted(panel['year'].unique())}")

    # Summary table
    summary = panel.groupby("year").agg(
        n_powiats            = ("teryt_powiat", "nunique"),
        mean_populist        = ("share_populist", "mean"),
        mean_populist_strict = ("share_populist_strict", "mean"),
        mean_farright        = ("share_farright", "mean"),
        mean_farright_strict = ("share_farright_strict", "mean"),
        mean_pop_or_fr       = ("share_populist_or_farright", "mean"),
    ).round(3)

    print("\nNATIONAL MEAN VOTE SHARES BY YEAR (powiat level)")
    print("Incl. = includes borderline PiS  |  Strict = excludes borderline PiS")
    print(summary.to_string())

    # Crosswalk
    crosswalk = build_crosswalk()
    print("\nCROSSWALK (matched parties only):")
    print(
        crosswalk[~crosswalk["party_short"].str.startswith("—")]
        .sort_values(["year", "party_short"])
        .to_string(index=False)
    )

    # Save
    output_path.mkdir(parents=True, exist_ok=True)
    panel.to_csv(    output_path / "panel_populist_powiat.csv",      index=False, encoding="utf-8-sig")
    crosswalk.to_csv(output_path / "crosswalk_party_labels_powiat.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(  output_path / "summary_powiat_national.csv",    encoding="utf-8-sig")

    print(f"\nSaved to {output_path}:")
    print(f"  panel_populist_powiat.csv         ({len(panel)} rows)")
    print(f"  crosswalk_party_labels_powiat.csv")
    print(f"  summary_powiat_national.csv")