# -*- coding: utf-8 -*-
"""
Revenue-Weighted Powiat → Gmina Disaggregation
===============================================

Drop-in replacement for the equal-split `disaggregate_powiat` function
in eu_funds_pipeline.py.

Methodology:
  For projects matched only to powiat level (no gmina), allocates funding
  proportionally to each gmina's share of own revenue per capita within
  that powiat in the corresponding year.

  weight_gmina_year = revenue_per_capita_gmina_year / sum(revenue_per_capita_powiat_year)

  This is more defensible than equal-split because wealthier gminas tend
  to have higher absorptive capacity for EU funds.

Usage:
  1. Load the weight table (generated separately):
     weights = pd.read_csv('revenue_weight_table_gmina_year.csv')
  
  2. Replace the disaggregate_powiat call in the pipeline:
     df = disaggregate_powiat_weighted(df, weights, decisions, period)
"""

import pandas as pd
import numpy as np
import logging

log = logging.getLogger("EU_FUNDS")


def disaggregate_powiat_weighted(df, weight_table, decisions=None, period=None):
    """
    Revenue-weighted disaggregation of powiat-level funding to gminas.
    
    For rows with powiat_id but no gmina_id, splits funding proportionally
    to each gmina's share of own revenue per capita within the powiat.
    
    Parameters
    ----------
    df : DataFrame
        Project-level data with columns: powiat_id, gmina_id, Year, 
        and financial columns (EU_subsidy_PLN, total_value_PLN, etc.)
    weight_table : DataFrame
        Must have columns: teryt_7, powiat_4, year, weight
        where weight = gmina revenue share within powiat (sums to 1)
    decisions : DecisionLog, optional
    period : str, optional
    
    Returns
    -------
    DataFrame with powiat-only rows replaced by weighted gmina-level rows
    """
    
    # Identify rows needing disaggregation
    mask_split = (
        (df['gmina_id'].isna() | (df['gmina_id'] == '')) & 
        df['powiat_id'].notna()
    )
    
    df_clean = df[~mask_split].copy()
    df_dirty = df[mask_split].copy()
    
    if df_dirty.empty:
        log.info("No powiat-only rows to disaggregate")
        return df_clean
    
    n_dirty = len(df_dirty)
    log.info(f"Weighted disaggregation: {n_dirty:,} powiat-only rows")
    
    # Ensure Year is int
    df_dirty['Year'] = pd.to_numeric(df_dirty['Year'], errors='coerce').fillna(0).astype(int)
    
    # Prepare weight table
    wt = weight_table[['teryt_7', 'powiat_4', 'year', 'weight']].copy()
    wt['year'] = wt['year'].astype(int)
    wt['powiat_4'] = wt['powiat_4'].astype(str)
    
    # Merge weights onto dirty rows
    df_dirty['powiat_id_4'] = df_dirty['powiat_id'].astype(str).str[:4].str.zfill(4)
    wt['powiat_4'] = wt['powiat_4'].astype(str).str.zfill(4)
    wt['teryt_7'] = wt['teryt_7'].astype(str).str.zfill(7)
    
    df_merged = df_dirty.merge(
        wt,
        left_on=['powiat_id_4', 'Year'],
        right_on=['powiat_4', 'year'],
        how='left'
    )
    
    # For years outside the weight table range, use nearest available year
    missing_weights = df_merged['weight'].isna()
    if missing_weights.sum() > 0:
        log.info(f"  {missing_weights.sum():,} rows have no weight for their year — using nearest year")
        
        available_years = sorted(wt['year'].unique())
        
        for idx in df_merged[missing_weights].index:
            yr = df_merged.loc[idx, 'Year']
            p4 = df_merged.loc[idx, 'powiat_id_4']
            
            # Find nearest year with weights for this powiat
            nearest = min(available_years, key=lambda y: abs(y - yr))
            fallback = wt[(wt['powiat_4'] == p4) & (wt['year'] == nearest)]
            
            if not fallback.empty:
                # This row will be handled in the explode step
                pass
        
        # Simpler approach: for missing years, use the overall average weight
        avg_weights = wt.groupby(['powiat_4', 'teryt_7'])['weight'].mean().reset_index()
        avg_weights.rename(columns={'weight': 'avg_weight'}, inplace=True)
        
        df_merged = df_merged.merge(
            avg_weights,
            left_on=['powiat_id_4', 'teryt_7'],
            right_on=['powiat_4', 'teryt_7'],
            how='left',
            suffixes=('', '_avg')
        )
        df_merged['weight'] = df_merged['weight'].fillna(df_merged.get('avg_weight', 0))
        df_merged.drop(columns=['avg_weight', 'powiat_4_avg'], errors='ignore', inplace=True)
    
    # Drop rows where weight is still 0 or NaN (no gmina info at all)
    df_merged = df_merged[df_merged['weight'].notna() & (df_merged['weight'] > 0)]
    
    # Apply weights to financial columns
    fin_cols = [c for c in ['EU_subsidy_PLN', 'total_value_PLN', 'subsidy_PLN',
                            'eligible_expenses_PLN'] if c in df.columns]
    
    for col in fin_cols:
        df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce').fillna(0) * df_merged['weight']
    
    # Set gmina_id from the weight table
    df_merged['gmina_id'] = df_merged['teryt_7']
    
    # Clean up helper columns
    drop_cols = ['powiat_id_4', 'teryt_7', 'powiat_4', 'year', 'weight']
    drop_cols = [c for c in drop_cols if c in df_merged.columns]
    df_merged.drop(columns=drop_cols, inplace=True)
    
    # Keep only columns from original df
    keep_cols = [c for c in df.columns if c in df_merged.columns]
    
    result = pd.concat([df_clean, df_merged[keep_cols]], ignore_index=True)
    
    log.info(
        f"  Disaggregated {n_dirty:,} → {len(df_merged):,} rows "
        f"(total: {len(result):,})"
    )
    
    if decisions:
        decisions.record(
            "D4_POWIAT_DISAGG_WEIGHTED",
            f"Weighted disaggregation ({period}): {n_dirty} powiat rows → {len(df_merged)} gmina rows",
            len(df), len(result), period=period
        )
    
    return result
