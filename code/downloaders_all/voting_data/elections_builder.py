# ============================================================
# CELL 1 — Paths and imports
# ============================================================
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path

populist_path = Path(r"C:\Users\jarem\OneDrive - London School of Economics\YEAR 2\1. Policy paper\policy-paper-repo\data\inputs\election_data\Populists_dataset\The Populist 3.0.csv")
raw_path      = Path(r"C:\Users\jarem\OneDrive - London School of Economics\YEAR 2\1. Policy paper\policy-paper-repo\data\clean\outcome\Elections\raw")
output_path   = Path(r"C:\Users\jarem\OneDrive - London School of Economics\YEAR 2\1. Policy paper\policy-paper-repo\data\clean\outcome\Elections")


# ============================================================
# CELL 2 — Load PopuList, filter Poland, build year-aware lookup
# ============================================================
pop_raw = pd.read_csv(populist_path, sep=";", encoding="utf-8-sig")
pop_pl  = pop_raw[pop_raw["country_name"] == "Poland"].copy()

print("Polish parties in PopuList:")
print(pop_pl[["party_name_short", "party_name_english",
              "populist", "populist_start", "populist_end", "populist_bl",
              "farright", "farright_start", "farright_end", "farright_bl"]].to_string(index=False))


# ============================================================
# CELL 3 — Party matching patterns
# ============================================================
# Each entry: party_short -> list of keywords that must ALL appear
# in the PKW column name (case-insensitive) to count as a match.
# Based on the actual column names we inspected above.

PARTY_PATTERNS = {
    "PiS":          ["Prawo i Sprawiedliwo"],          # covers "Prawo i Sprawiedliwość" across all years
    "Konfederacja": ["Konfederacja"],                   # 2019, 2023
    "KORWiN":       ["Korwin", "Korwin-Mikke","KORWiN"],         # 2011, 2015 — inside Konfederacja 2019+
    "LPR":          ["Liga Polskich Rodzin"],
    "SO":           ["Samoobrona"],                     # covers Samoobrona RP + Samoobrona Patriotyczna
    "Kukiz '15":    ["Kukiz"],
    "RN":           ["Ruch Narodowy"],
    "SP":           ["Solidarna Polska"],
    "KNP":          ["Kongres Nowej Prawicy"],
    "ZChN":         ["Zjednoczenie Chrze"],
    "ROP":          ["Ruch Odbudowy Polski"],
}

# Helper: given a PKW column name, return matching party shorts
def match_column(col_name: str) -> list[str]:
    col_lower = col_name.lower()
    matches = []
    for party, keywords in PARTY_PATTERNS.items():
        if all(kw.lower() in col_lower for kw in keywords):
            matches.append(party)
    return matches


# ============================================================
# CELL 4 — Build the PopuList tag lookup for a given election year
# ============================================================
def get_tags(year: int) -> dict:
    """
    Returns dict: party_short -> {populist, populist_strict,
                                  farright, farright_strict, ...}
    for parties active in `year` according to PopuList start/end dates.
    """
    tags = {}
    for _, row in pop_pl.iterrows():
        short = row["party_name_short"].strip()
        entry = {}
        for measure in ["populist", "farright", "farleft", "eurosceptic"]:
            val   = int(row[measure])
            start = int(row[f"{measure}_start"])
            end   = int(row[f"{measure}_end"])
            bl    = int(row[f"{measure}_bl"])
            active = (val == 1) and (start <= year <= end)
            entry[measure]              = active           # includes borderline
            entry[f"{measure}_strict"]  = active and not bl  # excludes borderline
        tags[short] = entry
    return tags


# ============================================================
# CELL 5 — File specs per year
# ============================================================
# Encoding all the structural differences we found in the inspection.

FILE_SPECS = {
    2023: {
        "file":      "wyniki_gl_na_listy_po_powiatach_sejm_csv.zip",
        "format":    "zip_csv",
        "teryt_col": "Kod TERYT",
        "powiat_col":"Powiat",
        "valid_col": "Liczba głosów ważnych oddanych łącznie na wszystkie listy kandydatów",
    },
    2019: {
        "file":      "wyniki_gl_na_listy_po_powiatach_sejm_csv.zip",
        "format":    "zip_csv",
        "teryt_col": "Kod TERYT",
        "powiat_col":"Powiat",
        "valid_col": "Liczba głosów ważnych oddanych łącznie na wszystkie listy kandydatów",
    },
    2015: {
        "file":      "2015-gl-lis-pow.zip",      # <-- counts, not proc
        "format":    "zip_csv",
        "teryt_col": "TERYT",                     # update after Cell 5
        "powiat_col":"Powiat",
        "valid_col": "Głosy ważne",               # update after Cell 5
    },
    2011: {
        "file":      "2011-sejm-pow-listy.xls",   # <-- corrected filename
        "format":    "xls",
        "teryt_col": "TERYT",
        "powiat_col":"Powiat",
        "valid_col": "Głosy ważne",
    },
    2007: {
        "file":      "sejm2007-pow-listy.xls",
        "format":    "xls",
        "teryt_col": "Kod pow.",
        "powiat_col":"Powiat",
        "valid_col": "Ważne",
    },
    2005: {
        "file":      "1456225675_36797.xls",
        "format":    "xls",
        "teryt_col": "TERYT",
        "powiat_col":"Powiat",
        "valid_col": "Głosy ważne",
    },
    2001: {
        "file":      "sejm2001-lis-pow.xls",      # <-- confirmed filename
        "format":    "xls",
        "teryt_col": "TERYT",
        "powiat_col":"Powiat",
        "valid_col": "Ważne",
    },
}

# FILE_SPECS_GMINA = {
#     2001: {
#         "file":         "sejm2001-lis-gm.xls",
#         "format":       "xls",
#         "teryt_col":    "TERYT",
#         "gmina_col":    "Gmina",
#         "valid_col":    "Ważne",
#         "note":         "5-digit TERYT, needs zero-padding to 6 digits"
#     },
#     2005: {
#         "file":         "1456225675_36795.xls",   # raw counts — NOT the % file
#         "format":       "xls",
#         "teryt_col":    "TERYT",
#         "gmina_col":    "Gmina",
#         "valid_col":    "Głosy ważne",
#         "note":         "Two files: _36795=counts, _36796=percentages. Use counts."
#     },
#     2007: {
#         "file":         "sejm2007-gm-listy.xls",
#         "format":       "xls",
#         "teryt_col":    "Kod gm.",               # different column name!
#         "gmina_col":    "Gmina",
#         "valid_col":    "Ważne",
#         "note":         "TERYT column is 'Kod gm.' not 'TERYT'"
#     },
#     2011: {
#         "file":         "2011-gl-lis-gm.xls",
#         "format":       "xls",
#         "teryt_col":    "TERYT",
#         "gmina_col":    "Gmina",
#         "valid_col":    "Głosy ważne",
#         "note":         ""
#     },
#     2015: {
#     "file":      "2015-gl-lis-gm.xls",
#     "format":    "xls",
#     "teryt_col": "TERYT",
#     "gmina_col": "Gmina",
#     "valid_col": "Głosy ważne",
#     "note":      "KORWiN and Kukiz'15 appear as separate columns"
#     },
#     2019: {
#         "file":         "wyniki_gl_na_listy_po_gminach_sejm_csv.zip",
#         "format":       "zip_csv",
#         "teryt_col":    "Kod TERYT",
#         "gmina_col":    "Gmina",
#         "valid_col":    "Liczba głosów ważnych oddanych łącznie na wszystkie listy kandydatów",
#         "note":         "First rows are overseas polling stations (no TERYT) — filtered out"
#     },
    
#     2023: {
#     "file":      "wyniki_gl_na_listy_po_gminach_sejm_csv.zip",
#     "format":    "zip_csv",
#     "teryt_col": "TERYT Gminy",         
#     "gmina_col": "Gmina",
#     "valid_col": "Liczba głosów ważnych oddanych łącznie na wszystkie listy kandydatów",
#     "note":      "Same structure as 2019"
#     }}


# ============================================================
# CELL 6 — Loader functions
# ============================================================
def load_raw(year: int) -> pd.DataFrame:
    spec     = FILE_SPECS[year]
    filepath = raw_path / str(year) / spec["file"]

    if spec["format"] == "zip_csv":
        with zipfile.ZipFile(filepath) as zf:
            csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(f, sep=";", encoding="utf-8", dtype=str, low_memory=False)
    else:
        df = pd.read_excel(filepath, dtype=str)

    return df


def clean_teryt(series: pd.Series, year: int) -> pd.Series:
    """Standardise TERYT to 7-digit string. Filter non-domestic rows."""
    s = series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    # Older files sometimes have 5 or 6 digits — pad to 7
    s = s.str.zfill(7)
    # Keep only rows that look like valid Polish TERYT (7 digits)
    valid_mask = s.str.match(r"^\d{7}$")
    return s.where(valid_mask, other=pd.NA)


def get_party_columns(df: pd.DataFrame) -> dict:
    """
    Find all party vote columns and match them to PopuList party shorts.
    Returns dict: col_name -> [party_short, ...]
    """
    # Party columns: everything that is NOT a known metadata column
    META = {
        "nr okr", "nr\nokr", "okręg", "teryt", "kod gm", "gmina", "powiat",
        "województwo", "typ", "l. upr", "uprawn", "frekw", "gł. odd", "odd.",
        "ważne", "głosy ważne", "głosy nieważne", "karty", "liczba",
        "komisj", "zaświadcz", "kopert", "pakiet", "w tym", "pełnomocn",
        "zaświadczenia"
    }
    result = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if any(meta in col_lower for meta in META):
            continue
        matches = match_column(col)
        result[col] = matches  # empty list = unmatched (mainstream party)
    return result


# ============================================================
# CELL 7 — Build single-year panel
# ============================================================
def build_year(year: int, verbose: bool = True) -> pd.DataFrame:
    spec = FILE_SPECS[year]
    df   = load_raw(year)
    tags = get_tags(year)

    # --- Standardise TERYT ---
    df["_teryt"] = clean_teryt(df[spec["teryt_col"]], year)
    df = df[df["_teryt"].notna()].copy()   # drop overseas/aggregate rows

    # --- Valid votes ---
    df["_valid"] = pd.to_numeric(
        df[spec["valid_col"]].astype(str).str.replace(",", ".").str.strip(),
        errors="coerce"
    )

    # --- Identify party columns ---
    party_cols = get_party_columns(df)

    # Convert all party columns to numeric
    for col in party_cols:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", ".").str.strip(),
            errors="coerce"
        ).fillna(0)

    if verbose:
        print(f"\n{'='*55}")
        print(f"  {year}")
        print(f"{'='*55}")
        print(f"  Rows (domestic gminas): {len(df)}")
        matched = {c: v for c, v in party_cols.items() if v}
        unmatched = [c for c, v in party_cols.items() if not v]
        print(f"  Matched party columns ({len(matched)}):")
        for col, parties in matched.items():
            print(f"    {col[:60]:60s} → {parties}")
        print(f"  Unmatched (mainstream, not in PopuList) ({len(unmatched)}):")
        for col in unmatched:
            print(f"    {col[:80]}")

    # --- Compute vote shares ---
    def col_sum_for_tag(tag: str, strict: bool = False) -> pd.Series:
        """Sum votes across all columns matching `tag` (incl. or excl. borderline)."""
        key = f"{tag}_strict" if strict else tag
        cols = [
            col for col, parties in party_cols.items()
            for p in parties
            if tags.get(p, {}).get(key, False)
        ]
        cols = list(set(cols))  # deduplicate if one column matched multiple parties
        if not cols:
            return pd.Series(0.0, index=df.index)
        return df[cols].sum(axis=1)

    valid = df["_valid"].replace(0, np.nan)

    panel = pd.DataFrame({
        "teryt":               df["_teryt"].values,
        "gmina_name":          df[spec["gmina_col"]].values,
        "year":                year,
        "valid_votes":         valid.values,

        # Populist (incl. borderline PiS)
        "votes_populist":      col_sum_for_tag("populist",    strict=False).values,
        # Populist strict (excl. borderline — PiS excluded 2005+)
        "votes_populist_strict": col_sum_for_tag("populist",  strict=True).values,

        # Far-right (incl. borderline PiS)
        "votes_farright":      col_sum_for_tag("farright",    strict=False).values,
        # Far-right strict
        "votes_farright_strict": col_sum_for_tag("farright",  strict=True).values,

        # Far-left
        "votes_farleft":       col_sum_for_tag("farleft",     strict=False).values,

        # Eurosceptic
        "votes_eurosceptic":   col_sum_for_tag("eurosceptic", strict=False).values,
    })

    # Vote SHARES (as proportions 0-1)
    for measure in ["populist", "populist_strict",
                    "farright", "farright_strict",
                    "farleft", "eurosceptic"]:
        panel[f"share_{measure}"] = panel[f"votes_{measure}"] / valid.values

    # Union: populist OR far-right (no double-counting)
    pop_cols = [
        col for col, parties in party_cols.items()
        for p in parties if tags.get(p, {}).get("populist", False)
    ]
    fr_cols = [
        col for col, parties in party_cols.items()
        for p in parties if tags.get(p, {}).get("farright", False)
    ]
    union_cols = list(set(pop_cols) | set(fr_cols))
    panel["votes_populist_or_farright"] = (
        df[union_cols].sum(axis=1).values if union_cols else 0.0
    )
    panel["share_populist_or_farright"] = panel["votes_populist_or_farright"] / valid.values

    # Summary
    if verbose:
        print(f"\n  National mean vote shares:")
        print(f"    share_populist         (incl. borderline PiS): {panel['share_populist'].mean():.3f}")
        print(f"    share_populist_strict  (excl. borderline PiS): {panel['share_populist_strict'].mean():.3f}")
        print(f"    share_farright         (incl. borderline PiS): {panel['share_farright'].mean():.3f}")
        print(f"    share_farright_strict  (excl. borderline PiS): {panel['share_farright_strict'].mean():.3f}")
        print(f"    share_populist_or_farright:                     {panel['share_populist_or_farright'].mean():.3f}")

    return panel


# ============================================================
# CELL 8 — Build full panel across all years
# ============================================================
YEARS = [2001, 2005, 2007, 2011, 2019, 2023]  # 2015 folder is empty

all_panels = []
for year in YEARS:
    try:
        panel = build_year(year, verbose=True)
        all_panels.append(panel)
    except FileNotFoundError as e:
        print(f"\nSKIPPED {year}: file not found — {e}")
    except Exception as e:
        print(f"\nERROR {year}: {e}")
        raise

panel = pd.concat(all_panels, ignore_index=True).sort_values(["teryt", "year"])

print(f"\n{'='*55}")
print(f"  PANEL COMPLETE")
print(f"{'='*55}")
print(f"  Total rows:     {len(panel)}")
print(f"  Unique gminas:  {panel['teryt'].nunique()}")
print(f"  Years:          {sorted(panel['year'].unique())}")
print(f"  Columns:        {panel.columns.tolist()}")


# ============================================================
# CELL 9 — Inspect crosswalk (ALWAYS CHECK THIS)
# ============================================================
# This shows exactly which PKW column was matched to which PopuList
# party in each year. Review carefully before trusting the shares.

crosswalk_rows = []
for year in YEARS:
    try:
        df_raw   = load_raw(year)
        spec     = FILE_SPECS[year]
        tags     = get_tags(year)
        df_raw["_teryt"] = clean_teryt(df_raw[spec["teryt_col"]], year)
        df_raw = df_raw[df_raw["_teryt"].notna()]
        party_cols = get_party_columns(df_raw)
        for col, parties in party_cols.items():
            if parties:
                for p in parties:
                    t = tags.get(p, {})
                    crosswalk_rows.append({
                        "year": year, "pkw_column": col, "party_short": p,
                        "populist": t.get("populist"), "populist_bl": t.get("populist") and not t.get("populist_strict"),
                        "farright": t.get("farright"), "farright_bl": t.get("farright") and not t.get("farright_strict"),
                    })
            else:
                crosswalk_rows.append({
                    "year": year, "pkw_column": col, "party_short": "— not in PopuList —",
                    "populist": False, "populist_bl": False,
                    "farright": False, "farright_bl": False,
                })
    except Exception:
        pass

crosswalk = pd.DataFrame(crosswalk_rows)
print("\nCROSSWALK — matched parties only:")
print(
    crosswalk[crosswalk["party_short"] != "— not in PopuList —"]
    .sort_values(["year", "party_short"])
    .to_string(index=False)
)


# ============================================================
# CELL 10 — Summary table by year
# ============================================================
summary = panel.groupby("year").agg(
    n_gminas             = ("teryt", "nunique"),
    mean_populist        = ("share_populist", "mean"),
    mean_populist_strict = ("share_populist_strict", "mean"),
    mean_farright        = ("share_farright", "mean"),
    mean_farright_strict = ("share_farright_strict", "mean"),
    mean_pop_or_fr       = ("share_populist_or_farright", "mean"),
).round(3)

print("\nNATIONAL MEAN VOTE SHARES BY YEAR")
print("(Incl. = includes borderline PiS;  Strict = excludes borderline PiS)")
print(summary.to_string())


# ============================================================
# CELL 11 — Save outputs
# ============================================================
output_path.mkdir(parents=True, exist_ok=True)

panel.to_csv(output_path / "panel_populist_gmina.csv", index=False, encoding="utf-8-sig")
crosswalk.to_csv(output_path / "crosswalk_party_labels.csv", index=False, encoding="utf-8-sig")
summary.to_csv(output_path / "summary_national_shares.csv", encoding="utf-8-sig")

print(f"\nSaved:")
print(f"  panel_populist_gmina.csv     — {len(panel)} rows")
print(f"  crosswalk_party_labels.csv   — review this!")
print(f"  summary_national_shares.csv")