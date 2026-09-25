"""
End-to-End Master Pipeline for Business Entity Resolution.
Orchestrates Data Loading -> Preprocessing -> Multi-Blocking -> Feature Engineering -> LightGBM Training -> F0.5 Optimization -> Test Inference -> Official Submission Validation.
"""
import os
import gc
import sys
import time
import numpy as np
import polars as pl
from typing import Dict, List, Set, Tuple

from src.config import (
    TRAIN_S1_PATH, TRAIN_S2_PATH, TRAIN_S3_PATH, TRAIN_GT_PATH,
    TEST_S1_PATH, TEST_S2_PATH, TEST_S3_PATH, LOG_PATH, EXPERIMENTS_DIR,
    RANDOM_SEED
)
from src.data_loader import scan_source_tsv, read_ground_truth, split_s1_train_val
from src.blocking import VectorizedBlocker
from src.features import extract_features_parallel, FEATURE_NAMES
from src.model import MatchingModel
from src.evaluation import evaluate_predictions
from src.submission import generate_submission_files, run_official_validator

def run_pipeline(sample_val_size: int = 50000, sample_train_size: int = 150000):
    """
    Executes end-to-end Entity Resolution pipeline.
    """
    print("==================================================", flush=True)
    print("AMAZON ML CHALLENGE 2026 - ENTITY RESOLUTION PIPELINE", flush=True)
    print("==================================================", flush=True)
    
    start_total = time.time()
    
    # 1. SCAN DATASETS (Lazy evaluation)
    print("\n--- PHASE 1: LAZY DATA SCANNING ---", flush=True)
    l_train_s1 = scan_source_tsv(TRAIN_S1_PATH)
    l_train_s2 = scan_source_tsv(TRAIN_S2_PATH)
    l_train_s3 = scan_source_tsv(TRAIN_S3_PATH)
    gt_map = read_ground_truth(TRAIN_GT_PATH)
    
    # 2. DATA SPLITTING (S1 Entity Level)
    print("\n--- PHASE 2: S1 ENTITY-LEVEL SPLITTING ---", flush=True)
    l_train_s1_sub, l_val_s1 = split_s1_train_val(l_train_s1, sample_val_size=sample_val_size, sample_train_size=sample_train_size)
    
    val_s1_df = l_val_s1.collect()
    train_s1_sub_df = l_train_s1_sub.collect()
    
    val_s1_ids = val_s1_df["entity_id"].to_list()
    train_s1_ids = train_s1_sub_df["entity_id"].to_list()
    
    val_gt_map = {sid: gt_map.get(sid, set()) for sid in val_s1_ids}
    train_gt_map = {sid: gt_map.get(sid, set()) for sid in train_s1_ids}
    
    val_countries = val_s1_df["country"].unique().to_list()
    train_countries = train_s1_sub_df["country"].unique().to_list()
    
    # 3. MULTI-BLOCKING
    print("\n--- PHASE 3: MULTI-BLOCKING CANDIDATE GENERATION ---", flush=True)
    blocker = VectorizedBlocker()
    
    print("[Blocking] Generating training candidate pairs...", flush=True)
    train_cands_df = blocker.generate_candidates(l_train_s1_sub, l_train_s2, l_train_s3, train_countries)
    
    print("[Blocking] Generating validation candidate pairs...", flush=True)
    val_cands_df = blocker.generate_candidates(l_val_s1, l_train_s2, l_train_s3, val_countries)
    
    # Measure Validation Candidate Recall
    val_recalled_pairs = 0
    val_total_gt_pairs = sum(len(s) for s in val_gt_map.values())
    
    cand_group = val_cands_df.group_by("s1_id").agg(pl.col("cand_id"))
    for row in cand_group.iter_rows():
        s1_i, c_list = row[0], row[1]
        true_set = val_gt_map.get(s1_i, set())
        val_recalled_pairs += len(true_set.intersection(set(c_list)))
        
    val_blocking_recall = (val_recalled_pairs / val_total_gt_pairs * 100) if val_total_gt_pairs > 0 else 0
    avg_cands_val = len(val_cands_df) / len(val_s1_ids)
    
    print(f"\n[Blocking Benchmark] Validation Candidate Recall: {val_blocking_recall:.2f}% ({val_recalled_pairs}/{val_total_gt_pairs})", flush=True)
    print(f"[Blocking Benchmark] Average Candidates per S1 Entity: {avg_cands_val:.2f}", flush=True)
    
    # 4. PAIR FEATURE EXTRACTION & HARD NEGATIVE MINING
    print("\n--- PHASE 4: FEATURE EXTRACTION & HARD NEGATIVE SAMPLING ---", flush=True)
    
    # Identify positive and negative candidate pairs
    train_s1_arr = train_cands_df["s1_id"].to_list()
    train_cand_arr = train_cands_df["cand_id"].to_list()
    y_raw = np.array([1 if cand_id in train_gt_map.get(s1_id, set()) else 0 for s1_id, cand_id in zip(train_s1_arr, train_cand_arr)], dtype=np.int32)
    
    pos_indices = np.where(y_raw == 1)[0]
    neg_indices = np.where(y_raw == 0)[0]
    
    np.random.seed(RANDOM_SEED)
    n_negs_to_sample = min(len(neg_indices), len(pos_indices) * 5)
    sampled_neg_indices = np.random.choice(neg_indices, size=n_negs_to_sample, replace=False)
    
    selected_indices = np.concatenate([pos_indices, sampled_neg_indices])
    np.random.shuffle(selected_indices)
    
    train_cands_sampled = train_cands_df[selected_indices]
    y_train = y_raw[selected_indices]
    
    print(f"[Features] Sampled {len(train_cands_sampled)} training pairs (Positives: {np.sum(y_train == 1)}, Hard Negatives: {np.sum(y_train == 0)}).", flush=True)
    
    # Materialize text DataFrames for Polars join
    s1_text_df = pl.concat([val_s1_df, train_s1_sub_df]).select(["entity_id", "business_name", "business_address"]).unique()
    s23_text_df = pl.concat([
        l_train_s2.select(["entity_id", "business_name", "business_address"]).collect(),
        l_train_s3.select(["entity_id", "business_name", "business_address"]).collect(),
    ])
    
    full_train_cands = train_cands_sampled.join(
        s1_text_df.select([pl.col("entity_id").alias("s1_id"), pl.col("business_name").alias("name1"), pl.col("business_address").alias("addr1")]), on="s1_id"
    ).join(
        s23_text_df.select([pl.col("entity_id").alias("cand_id"), pl.col("business_name").alias("name2"), pl.col("business_address").alias("addr2")]), on="cand_id"
    )
    
    X_train = extract_features_parallel(full_train_cands)
    
    del train_cands_df, train_cands_sampled, full_train_cands
    gc.collect()
    
    # 5. MODEL TRAINING
    print("\n--- PHASE 5: LIGHTGBM MODEL TRAINING ---", flush=True)
    model = MatchingModel()
    model.train(X_train, y_train)
    
    fi = model.get_feature_importances()
    print("\n[Features] Feature Importances:")
    sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
    for feat_name, imp in sorted_fi:
        print(f"  {feat_name:28s}: {imp}")
        
    del X_train, y_train
    gc.collect()
    
    # 6. VALIDATION & THRESHOLD OPTIMIZATION
    print("\n--- PHASE 6: VALIDATION & F0.5 THRESHOLD OPTIMIZATION ---", flush=True)
    full_val_cands = val_cands_df.join(
        s1_text_df.select([pl.col("entity_id").alias("s1_id"), pl.col("business_name").alias("name1"), pl.col("business_address").alias("addr1")]), on="s1_id"
    ).join(
        s23_text_df.select([pl.col("entity_id").alias("cand_id"), pl.col("business_name").alias("name2"), pl.col("business_address").alias("addr2")]), on="cand_id"
    )
    
    X_val = extract_features_parallel(full_val_cands)
    val_probs = model.predict_proba(X_val)
    
    val_cands_df = val_cands_df.with_columns(pl.Series("score", val_probs))
    val_grouped_scores = val_cands_df.group_by("s1_id").agg([pl.col("cand_id"), pl.col("score")])
    
    val_candidate_scores = {}
    for row in val_grouped_scores.iter_rows():
        s1_i, c_list, p_list = row[0], row[1], row[2]
        val_candidate_scores[s1_i] = list(zip(c_list, p_list))
        
    best_thresh = model.optimize_threshold(val_gt_map, val_candidate_scores)
    
    val_preds = {s1_id: {c_id for c_id, score in scored if score >= best_thresh} for s1_id, scored in val_candidate_scores.items()}
    val_results = evaluate_predictions(val_gt_map, val_preds)
    
    print("\n==================================================", flush=True)
    print("FINAL VALIDATION RESULTS:", flush=True)
    print(f"  Macro F0.5 Score   : {val_results['macro_f05']:.4f}", flush=True)
    print(f"  Macro Precision    : {val_results['macro_precision']:.4f}", flush=True)
    print(f"  Macro Recall       : {val_results['macro_recall']:.4f}", flush=True)
    print(f"  Singleton Accuracy : {val_results['singleton_accuracy']:.2f}%", flush=True)
    print(f"  False Positives    : {val_results['false_positives']}", flush=True)
    print(f"  False Negatives    : {val_results['false_negatives']}", flush=True)
    print("==================================================", flush=True)
    
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        if os.path.getsize(LOG_PATH) == 0:
            f.write("experiment_id,change,candidate_recall,candidate_count,validation_f05,precision,recall,notes\n")
        f.write(f"exp_{int(time.time())},polars_lightgbm_pipeline,{val_blocking_recall:.2f},{avg_cands_val:.2f},{val_results['macro_f05']:.4f},{val_results['macro_precision']:.4f},{val_results['macro_recall']:.4f},validated_on_{len(val_s1_ids)}_s1\n")
        
    del val_cands_df, full_val_cands, X_val, s1_text_df, s23_text_df
    gc.collect()
    
    # 7. TEST INFERENCE & OUTPUT GENERATION
    print("\n--- PHASE 7: FULL TEST INFERENCE ---", flush=True)
    l_test_s1 = scan_source_tsv(TEST_S1_PATH)
    l_test_s2 = scan_source_tsv(TEST_S2_PATH)
    l_test_s3 = scan_source_tsv(TEST_S3_PATH)
    
    test_s1_df = l_test_s1.collect()
    all_test_s1_ids = test_s1_df["entity_id"].to_list()
    test_countries = test_s1_df["country"].unique().to_list()
    
    print(f"[Inference] Generating test candidates across countries: {test_countries}...", flush=True)
    test_cands_df = blocker.generate_candidates(l_test_s1, l_test_s2, l_test_s3, test_countries)
    
    print("[DataLoader] Joining test text DataFrames...", flush=True)
    test_s23_text_df = pl.concat([
        l_test_s2.select(["entity_id", "business_name", "business_address"]).collect(),
        l_test_s3.select(["entity_id", "business_name", "business_address"]).collect(),
    ])
    
    full_test_cands = test_cands_df.join(
        test_s1_df.select([pl.col("entity_id").alias("s1_id"), pl.col("business_name").alias("name1"), pl.col("business_address").alias("addr1")]), on="s1_id"
    ).join(
        test_s23_text_df.select([pl.col("entity_id").alias("cand_id"), pl.col("business_name").alias("name2"), pl.col("business_address").alias("addr2")]), on="cand_id"
    )
    
    print(f"[Inference] Extracting features for {len(full_test_cands)} test candidate pairs...", flush=True)
    X_test = extract_features_parallel(full_test_cands)
    test_probs = model.predict_proba(X_test)
    
    test_cands_df = test_cands_df.with_columns(pl.Series("score", test_probs))
    
    test_cand_grouped = test_cands_df.group_by("s1_id").agg(pl.col("cand_id"))
    test_cands_map = {row[0]: set(row[1]) for row in test_cand_grouped.iter_rows()}
    
    test_matches_df = test_cands_df.filter(pl.col("score") >= best_thresh).group_by("s1_id").agg(pl.col("cand_id"))
    test_matches_map = {row[0]: set(row[1]) for row in test_matches_df.iter_rows()}
    
    # 8. SUBMISSION & OFFICIAL VALIDATOR
    print("\n--- PHASE 8: OFFICIAL SUBMISSION VALIDATION ---", flush=True)
    generate_submission_files(test_matches_map, test_cands_map, all_test_s1_ids)
    is_valid = run_official_validator(check_ids=False)
    
    print(f"\n==================================================", flush=True)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {time.time() - start_total:.2f} SECONDS!", flush=True)
    print(f"Validation Macro F0.5: {val_results['macro_f05']:.4f}")
    print(f"Official Validator Status: {'PASS' if is_valid else 'FAIL'}")
    print("==================================================", flush=True)
    
    return is_valid

if __name__ == "__main__":
    run_pipeline()
