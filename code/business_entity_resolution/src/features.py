"""
Feature Engineering Module for Entity Resolution.
Generates rich similarity features for candidate pairs directly into pre-allocated numpy arrays.
"""
import math
from typing import Dict, List, Tuple
import numpy as np
import polars as pl
from rapidfuzz import fuzz
import Levenshtein

FEATURE_NAMES = [
    'name_exact_match',
    'name_levenshtein_ratio',
    'name_token_sort_ratio',
    'name_token_set_ratio',
    'name_len_diff',
    'addr_is_missing',
    'addr_exact_match',
    'addr_levenshtein_ratio',
    'addr_token_set_ratio',
    'addr_len_diff',
    'name_addr_sim_prod',
    'name_addr_sim_min',
    'name_addr_sim_max',
    'is_source2',
]

def compute_single_pair_into(out_row: np.ndarray, n1: str, a1: str, n2: str, a2: str, cand_id: str):
    """Computes similarity feature vector into pre-allocated numpy row array."""
    n1_clean = str(n1).lower()
    n2_clean = str(n2).lower()
    a1_clean = str(a1).lower()
    a2_clean = str(a2).lower()
    
    out_row[0] = 1.0 if n1_clean and n1_clean == n2_clean else 0.0
    out_row[1] = Levenshtein.ratio(n1_clean, n2_clean) if n1_clean and n2_clean else 0.0
    out_row[2] = fuzz.token_sort_ratio(n1_clean, n2_clean) / 100.0 if n1_clean and n2_clean else 0.0
    out_row[3] = fuzz.token_set_ratio(n1_clean, n2_clean) / 100.0 if n1_clean and n2_clean else 0.0
    out_row[4] = float(abs(len(n1_clean) - len(n2_clean)))
    
    a_missing = 1.0 if (not a1_clean or not a2_clean or a1_clean == 'nan' or a2_clean == 'nan') else 0.0
    out_row[5] = a_missing
    out_row[6] = 1.0 if (a1_clean and a1_clean == a2_clean) else 0.0
    out_row[7] = Levenshtein.ratio(a1_clean, a2_clean) if (a1_clean and a2_clean) else 0.0
    out_row[8] = fuzz.token_set_ratio(a1_clean, a2_clean) / 100.0 if (a1_clean and a2_clean) else 0.0
    out_row[9] = float(abs(len(a1_clean) - len(a2_clean)))
    
    n_set = out_row[3]
    a_set = out_row[8]
    out_row[10] = n_set * a_set
    out_row[11] = min(n_set, a_set)
    out_row[12] = max(n_set, a_set)
    out_row[13] = 1.0 if str(cand_id).startswith("S2-") else 0.0

def extract_features_parallel(cands_df: pl.DataFrame) -> np.ndarray:
    """
    Extracts features directly into a single pre-allocated float32 numpy array.
    Zero Python list overhead.
    """
    n_samples = len(cands_df)
    if n_samples == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        
    n1_arr = cands_df["name1"].to_list()
    a1_arr = cands_df["addr1"].to_list()
    n2_arr = cands_df["name2"].to_list()
    a2_arr = cands_df["addr2"].to_list()
    cand_ids = cands_df["cand_id"].to_list()
    
    # Pre-allocate numpy matrix
    X = np.zeros((n_samples, len(FEATURE_NAMES)), dtype=np.float32)
    
    for i in range(n_samples):
        compute_single_pair_into(X[i], n1_arr[i], a1_arr[i], n2_arr[i], a2_arr[i], cand_ids[i])
        
    return X
