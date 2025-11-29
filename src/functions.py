import re 
import pandas as pd
import unicodedata

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



def extract_woj_pow(location_str):
    """
    Extract WOJ and POW pairs from a location string.
    Handles format: "WOJ.: NAME, POW.: NAME | WOJ.: NAME, POW.: NAME"
    Returns a list of dicts with 'woj' and 'pow' keys.
    Example: "WOJ.: LUBUSKIE, POW.: Gorzów Wielkopolski" 
             -> [{'woj': 'LUBUSKIE', 'pow': 'Gorzów Wielkopolski'}]
    """
    if pd.isna(location_str):
        return []
    
    location_str = str(location_str).strip()
    pairs = []
    
    # Split by | to handle multiple locations
    locations = location_str.split('|')
    
    for loc in locations:
        loc = loc.strip()
        
        # Find WOJ value (text after "WOJ.:" up to comma)
        woj_match = re.search(r'WOJ\.\s*:\s*([^,]+)', loc, re.IGNORECASE)
        # Find POW value (text after "POW.:" to end of string or next |)
        pow_match = re.search(r'POW\.\s*:\s*([^|]+)', loc, re.IGNORECASE)
        
        woj = woj_match.group(1).strip() if woj_match else None
        pow = pow_match.group(1).strip() if pow_match else None
        
        if woj or pow:
            pairs.append({'woj': woj, 'pow': pow})
    
    return pairs

def split_funding_by_locations(df, location_col, funding_col):
    """Split funding equally across multiple locations."""
    df = df.copy()
    df['num_locations'] = df[location_col].apply(lambda x: len(extract_woj_pow(x)))
    df['funding_per_location'] = df.apply(
        lambda row: row[funding_col] / row['num_locations'] if row['num_locations'] > 0 else row[funding_col],
        axis=1
    )
    return df



_POLISH_VOIVODES = [
    "Dolnośląskie", "Kujawsko-pomorskie", "Lubelskie", "Lubuskie", "Łódzkie",
    "Małopolskie", "Mazowieckie", "Opolskie", "Podkarpackie", "Podlaskie",
    "Pomorskie", "Śląskie", "Świętokrzyskie", "Warmińsko-mazurskie",
    "Wielkopolskie", "Zachodniopomorskie"
]
def _norm_name(s):
    if not s:
        return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s

_POLISH_VOIVODES_N = set(_norm_name(v) for v in _POLISH_VOIVODES)

def is_polish_voivodeship(name):
    """Return True if name matches a Polish voivodeship (robust to case/diacritics)."""
    return _norm_name(name) in _POLISH_VOIVODES_N

def detect_poland_for_pairs(pairs, location_text=None):
    """
    Given a list of {'woj','pow'} pairs return:
      - is_poland: True if all pairs point to Poland (woj in Polish voivodeships)
      - countries_seen: set of detected countries ('Poland' and/or 'Other')
    Fallback: if pairs empty, inspect raw location_text for 'Poland'/'Polska'.
    """
    countries = set()
    if pairs:
        for p in pairs:
            woj = p.get("woj")
            if woj and is_polish_voivodeship(woj):
                countries.add("Poland")
            else:
                countries.add("Other")
    else:
        txt = (location_text or "") .lower()
        if "polska" in txt or "poland" in txt:
            countries.add("Poland")
        elif txt.strip() == "":
            countries.add("Unknown")
        else:
            countries.add("Other")
    is_poland = (len(countries) == 1 and "Poland" in countries)
    return is_poland, countries

def flag_projects_country(df, location_col="Project location", project_id_col=None):
    """
    Add row-level 'is_poland_location' (bool) and 'countries_seen' (str list),
    and project-level 'project_country_status' if project_id_col provided.

    project_country_status values: 'all_poland', 'crossborder', 'no_poland'
    """
    df = df.copy()
    is_poland_list = []
    countries_list = []
    for _, r in df.iterrows():
        loc = r.get(location_col, "")
        pairs = extract_woj_pow(loc)
        is_pol, countries = detect_poland_for_pairs(pairs, location_text=loc)
        is_poland_list.append(is_pol)
        countries_list.append(";".join(sorted(countries)))
    df["is_poland_location"] = is_poland_list
    df["countries_seen"] = countries_list

    if project_id_col and project_id_col in df.columns:
        # determine project-level status
        grp = df.groupby(project_id_col)["is_poland_location"]
        def _status(x):
            if x.all():
                return "all_poland"
            if x.any() and not x.all():
                return "crossborder"
            return "no_poland"
        proj_status = grp.apply(_status).rename("project_country_status").reset_index()
        df = df.merge(proj_status, on=project_id_col, how="left")
    return df

def unnest_locations(df, location_col, funding_col, project_id_col=None):
    """
    Return DataFrame with one row per WOJ-POW pair.
    - preserves all original columns by converting each row to dict()
    - replaces location_col with the specific "WOJ.: ..., POW.: ..." string for that row
    - adds 'woj', 'pow' and 'funding_split' columns
    - If a row has multiple locations, the original multi-location row is NOT kept.
    - If a row has exactly one location, it becomes a single standardized row.
    """
    rows = []
    for _, r in df.iterrows():
        pairs = extract_woj_pow(r.get(location_col))
        # robust parse of funding value
        total_raw = r.get(funding_col)
        total_num = 0.0
        if pd.notna(total_raw):
            try:
                if isinstance(total_raw, str):
                    s = total_raw.replace('\xa0', ' ').strip()
                    s = re.sub(r'[^\d\-\.,]', '', s)
                    s = s.replace(' ', '')
                    if ',' in s and '.' in s:
                        s = s.replace(',', '')
                    elif ',' in s and '.' not in s:
                        s = s.replace(',', '.')
                    total_num = float(s) if s not in ('', '.', '-') else 0.0
                else:
                    total_num = float(total_raw)
            except Exception:
                total_num = 0.0

        n = len(pairs)
        if n > 0:
            # compute split: if single location keep full amount; if multiple divide equally
            split = total_num if n == 1 else (total_num / n if n else total_num)
            for p in pairs:
                new = r.to_dict()            # preserve all original columns
                # set standardized location text for the single pair
                new[location_col] = f"WOJ.: {p.get('woj')}, POW.: {p.get('pow')}"
                new['woj'] = p.get('woj')
                new['pow'] = p.get('pow')
                new['funding_split'] = split
                rows.append(new)
            # important: do NOT append the original multi-location row
        else:
            # no parsed locations -> keep original row as-is but with funding_split parsed
            new = r.to_dict()
            new['woj'] = None
            new['pow'] = None
            new['funding_split'] = total_num
            rows.append(new)

    out = pd.DataFrame(rows)
    # keep original column order + added cols at the end if not present
    base_cols = list(df.columns)
    for extra in ['woj', 'pow', 'funding_split']:
        if extra not in base_cols:
            base_cols.append(extra)
    return out[base_cols]

def unnest_locations_with_gmina(df, location_col, funding_col, project_id_col=None):
    """
    Unnest locations where multiple locations are separated by '|' and components
    are labelled with WOJ / POW / GM (variants like 'WOJ.:', 'POW.:', 'GM.:' etc).
    Returns original rows exploded into one row per segment with columns:
    woj, pow, gmina, funding_split (funding divided equally across segments).
    """
    import re
    import pandas as pd

    def clean_component(s):
        if s is None or (isinstance(s, float) and pd.isna(s)):
            return None
        s = str(s).strip()
        # remove leading label variants (WOJ, POW, GM) with punctuation/space
        s = re.sub(r'^\s*(?:WOJ|POW|GM)(?:[\.\:]{1,2})?\s*[:\.\-]*\s*', '', s, flags=re.I)
        # remove trailing explanatory suffixes e.g. " - Gmina wiejska"
        s = re.sub(r'\s*-\s*Gmina.*$', '', s, flags=re.I)
        # trim leftover punctuation and whitespace
        s = s.strip(" .,:;–—-")
        return s if s != "" else None

    rows = []
    for _, row in df.iterrows():
        loc_raw = row.get(location_col)
        # parse funding numeric
        total_raw = row.get(funding_col)
        total_num = 0.0
        if pd.notna(total_raw):
            try:
                if isinstance(total_raw, str):
                    s = total_raw.replace('\xa0', ' ')
                    s = re.sub(r'[^\d\-\.,]', '', s)
                    s = s.replace(' ', '')
                    if ',' in s and '.' in s:
                        s = s.replace(',', '')
                    elif ',' in s and '.' not in s:
                        s = s.replace(',', '.')
                    total_num = float(s) if s not in ('', '.', '-') else 0.0
                else:
                    total_num = float(total_raw)
            except Exception:
                total_num = 0.0

        if pd.isna(loc_raw) or not isinstance(loc_raw, str) or loc_raw.strip() == "":
            new = row.to_dict()
            new.update({'woj': None, 'pow': None, 'gmina': None, 'funding_split': total_num})
            rows.append(new)
            continue

        # split by '|' into segments (user guaranteed this separator)
        segments = [seg.strip() for seg in loc_raw.split('|') if seg.strip()]
        nseg = max(1, len(segments))
        per_seg_fund = total_num / nseg if nseg > 0 else total_num

        for seg in segments:
            # split segment by commas (fields like "WOJ.: X, POW.: Y, GM.: Z")
            parts = [p.strip() for p in re.split(r',\s*', seg) if p.strip()]
            woj = None
            pow_ = None
            gmina = None
            for p in parts:
                pl = p.lower()
                if pl.startswith('woj') or re.match(r'^\s*woj[\.\:]', pl):
                    woj = clean_component(p)
                    continue
                if pl.startswith('pow') or re.match(r'^\s*pow[\.\:]', pl):
                    pow_ = clean_component(p)
                    continue
                if pl.startswith('gm') or re.match(r'^\s*gm[\.\:]', pl):
                    gmina = clean_component(p)
                    continue
                # fallback: if no explicit label but part looks like "POWNAME powiat" unlikely, skip

            # If some labels weren't captured but the segment contains them without comma separation,
            # try label searches across the whole segment (handles missing commas)
            if not (woj or pow_ or gmina):
                # search for labelled values anywhere
                m_woj = re.search(r'WOJ[^\wÀ-ž]*([^\|,;]+)', seg, flags=re.I)
                m_pow = re.search(r'POW[^\wÀ-ž]*([^\|,;]+)', seg, flags=re.I)
                m_gm = re.search(r'GM[^\wÀ-ž]*([^\|,;]+)', seg, flags=re.I)
                if m_woj:
                    woj = clean_component(m_woj.group(1))
                if m_pow:
                    pow_ = clean_component(m_pow.group(1))
                if m_gm:
                    gmina = clean_component(m_gm.group(1))

            new = row.to_dict()
            # build standardized Project location string for this segment
            parts_out = []
            if woj:
                parts_out.append(f"WOJ: {woj}")
            if pow_:
                parts_out.append(f"POW: {pow_}")
            if gmina:
                parts_out.append(f"GM: {gmina}")
            new_loc = ", ".join(parts_out) if parts_out else seg
            new[location_col] = new_loc
            new['woj'] = woj
            new['pow'] = pow_
            new['gmina'] = gmina
            new['funding_split'] = per_seg_fund
            rows.append(new)

    out = pd.DataFrame(rows)

    # ensure added cols exist in output (keep original columns order + extras at end)
    base_cols = list(df.columns)
    for extra in ['woj', 'pow', 'gmina', 'funding_split']:
        if extra not in base_cols:
            base_cols.append(extra)
    for c in base_cols:
        if c not in out.columns:
            out[c] = None

    return out[base_cols]



def add_iso3_column(df):
    """
    Add or infer iso3 column for location rows.
    - Preserve existing iso3 values if present.
    - Set 'POL' when woj matches a Polish voivodeship.
    - Also set 'POL' when pow or gmina are present (covers woj+gmina without pow).
    - If 'is_poland_location' column exists, use it as a fallback.
    """
    df = df.copy()
    has_flag = 'is_poland_location' in df.columns

    def _infer_iso(row):
        # keep existing explicit iso3
        existing = row.get('iso3', None)
        if pd.notna(existing) and existing not in (None, ''):
            return existing

        woj = row.get('woj')
        pow_ = row.get('pow')
        gmina = row.get('gmina')

        # if woj is a recognised Polish voivodeship -> POL
        if woj and is_polish_voivodeship(woj):
            return 'POL'

        # if woj missing but pow or gmina present -> assume POL
        if (pow_ and str(pow_).strip()) or (gmina and str(gmina).strip()):
            return 'POL'

        # fallback to project-level / row-level flag if available
        if has_flag and row.get('is_poland_location') is True:
            return 'POL'

        return None

    df['iso3'] = df.apply(_infer_iso, axis=1)
    return df
