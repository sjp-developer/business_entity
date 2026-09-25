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
    Executes end-to-end Entity Resolution pipeline with strict memory management.
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
    l_train_s1_sub, l_val_s1 = split_s1_train_val(l_train_s1, gt_map, sample_val_size=sample_val_size, sample_train_size=sample_train_size)
    
    val_s1_df = l_val_s1.collect()
    train_s1_sub_df = l_train_s1_sub.collect()
    
    val_s1_ids = val_s1_df["entity_id"].to_list()
    train_s1_ids = train_s1_sub_df["entity_id"].to_list()
    
    val_gt_map = {sid: gt_map.get(sid, set()) for sid in val_s1_ids if sid in gt_map}
    train_gt_map = {sid: gt_map.get(sid, set()) for sid in train_s1_ids if sid in gt_map}
    
    val_countries = val_s1_df["country"].unique().to_list()
    train_countries = train_s1_sub_df["country"].unique().to_list()
    
    # 3. MULTI-BLOCKING
    print("\n--- PHASE 3: MULTI-BLOCKING CANDIDATE GENERATION ---", flush=True)
    blocker = VectorizedBlocker()
    
    print("[Blocking] Generating training candidate pairs...", flush=True)
    train_cands_df = blocker.generate_candidates(l_train_s1_sub, l_train_s2, l_train_s3, train_countries)
    
    print("[Blocking] Generating validation candidate pairs...", flush=True)
    val_cands_df = blocker.generate_candidates(l_val_s1, l_train_s2, l_train_s3, val_countries)
    
    val_recalled_pairs = 0
    val_total_gt_pairs = sum(len(s) for s in val_gt_map.values())
    
    cand_group = val_cands_df.group_by("s1_id").agg(pl.col("cand_id"))
    for row in cand_group.iter_rows():
        s1_i, c_list = row[0], row[1]
        true_set = val_gt_map.get(s1_i, set())
        val_recalled_pairs += len(true_set.intersection(set(c_list)))
        
    val_blocking_recall = (val_recalled_pairs / val_total_gt_pairs * 100) if val_total_gt_pairs > 0 else 0
    avg_cands_val = len(val_cands_df) / max(1, len(val_s1_ids))
    
    print(f"\n[Blocking Benchmark] Validation Candidate Recall: {val_blocking_recall:.2f}% ({val_recalled_pairs}/{val_total_gt_pairs})", flush=True)
    print(f"[Blocking Benchmark] Average Candidates per S1 Entity: {avg_cands_val:.2f}", flush=True)
    
    # 4. PAIR FEATURE EXTRACTION & HARD NEGATIVE SAMPLING
    print("\n--- PHASE 4: FEATURE EXTRACTION & HARD NEGATIVE SAMPLING ---", flush=True)
    
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
    
    X_train = extract_features_parallel(train_cands_sampled)
    
    del train_cands_df, train_cands_sampled, train_s1_sub_df
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
    X_val = extract_features_parallel(val_cands_df)
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
        
    # Strictly release all training memory before loading test data
    del val_cands_df, X_val, val_s1_df, l_train_s1, l_train_s2, l_train_s3, gt_map
    gc.collect()
    
    # 7. TEST INFERENCE & OUTPUT GENERATION
    print("\n--- PHASE 7: COUNTRY-PARTITIONED STREAMING TEST INFERENCE ---", flush=True)
    l_test_s1 = scan_source_tsv(TEST_S1_PATH)
    l_test_s2 = scan_source_tsv(TEST_S2_PATH)
    l_test_s3 = scan_source_tsv(TEST_S3_PATH)
    
    all_test_s1_ids = l_test_s1.select("entity_id").collect()["entity_id"].to_list()
    test_countries = l_test_s1.select("country").unique().collect()["country"].to_list()
    
    test_cands_map = {}
    test_matches_map = {}
    
    chunk_size = 30000
    
    from src.blocking import get_lean_blocking_keys
    
    for c in test_countries:
        print(f"\n[Test Inference] Processing Country: '{c}'...", flush=True)
        s2_k = get_lean_blocking_keys(l_test_s2, c).collect()
        s3_k = get_lean_blocking_keys(l_test_s3, c).collect()
        s23_k = pl.concat([s2_k, s3_k])
        del s2_k, s3_k
        gc.collect()
        
        c_s1_lazy = l_test_s1.filter(pl.col("country") == c)
        n_c_s1 = c_s1_lazy.select(pl.len()).collect().item()
        print(f"  Country '{c}' total S1 entities: {n_c_s1}, S2/S3 candidates: {len(s23_k)}", flush=True)
        
        for offset in range(0, n_c_s1, chunk_size):
            s1_chunk_lazy = c_s1_lazy.slice(offset, chunk_size)
            s1_k_chunk = get_lean_blocking_keys(s1_chunk_lazy, c).collect()
            print(f"  [Chunk {offset//chunk_size + 1}] Processing entities {offset} to {min(offset+chunk_size, n_c_s1)}...", flush=True)
            
            chunk_cands_df = blocker.generate_candidates_from_keys(s1_k_chunk, s23_k)
            del s1_k_chunk
            gc.collect()
            
            if len(chunk_cands_df) == 0:
                continue
                
            X_chunk = extract_features_parallel(chunk_cands_df)
            chunk_probs = model.predict_proba(X_chunk)
            
            chunk_scored = chunk_cands_df.select(["s1_id", "cand_id"]).with_columns(pl.Series("score", chunk_probs))
            chunk_grouped = chunk_scored.group_by("s1_id").agg([pl.col("cand_id"), pl.col("score")])
            
            for row in chunk_grouped.iter_rows():
                s1_i, cand_ids, scores = row[0], row[1], row[2]
                test_cands_map[s1_i] = set(cand_ids)
                match_set = {cid for cid, sc in zip(cand_ids, scores) if sc >= best_thresh}
                if match_set:
                    test_matches_map[s1_i] = match_set
                    
            del chunk_cands_df, X_chunk, chunk_probs, chunk_scored, chunk_grouped
            gc.collect()
            
        del s23_k
        gc.collect()
    
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
