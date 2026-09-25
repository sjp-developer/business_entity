"""
Data Loader Module for Entity Resolution pipeline using Polars.
Handles memory-efficient TSV scanning and leakage-free entity-level train/validation splitting.
"""
import os
from typing import Dict, List, Set, Tuple
import polars as pl
import numpy as np

from src.config import (
    TRAIN_S1_PATH, TRAIN_S2_PATH, TRAIN_S3_PATH, TRAIN_GT_PATH,
    TEST_S1_PATH, TEST_S2_PATH, TEST_S3_PATH, RANDOM_SEED
)

def scan_source_tsv(path: str) -> pl.LazyFrame:
    """Returns a LazyFrame scanner for a source TSV file."""
    return pl.scan_csv(path, separator="\t", has_header=True, schema_overrides={
        "entity_id": pl.Utf8, "business_name": pl.Utf8, "business_address": pl.Utf8, "country": pl.Utf8
    }).fill_null("")

def read_ground_truth(path: str) -> Dict[str, Set[str]]:
    """Reads ground truth TSV into a mapping {source1_entity_id: set_of_matched_ids}."""
    print(f"[DataLoader] Reading {os.path.basename(path)}...", flush=True)
    df = pl.read_csv(path, separator="\t", has_header=True, schema_overrides={
        "source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8
    }).fill_null("")
    
    gt_map = {}
    for row in df.iter_rows(named=True):
        s1_id = row["source1_entity_id"]
        m_str = row["matched_entity_ids"].strip()
        if m_str:
            m_set = {m.strip() for m in m_str.split(",") if m.strip()}
        else:
            m_set = set()
        gt_map[s1_id] = m_set
        
    return gt_map

def split_s1_train_val(l_s1: pl.LazyFrame, sample_val_size: int = 50000, sample_train_size: int = 150000) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    Splits Source 1 records into train and validation sets at the S1 entity level.
    """
    print(f"[DataLoader] Creating S1 Entity split: {sample_val_size} Val S1 entities, {sample_train_size} Train S1 entities...", flush=True)
    val_l_s1 = l_s1.slice(0, sample_val_size)
    train_l_s1 = l_s1.slice(sample_val_size, sample_train_size)
    return train_l_s1, val_l_s1
