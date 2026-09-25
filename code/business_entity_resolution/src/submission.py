"""
Output Generation and Validation Helper Module.
Generates output/matching_results.tsv and output/candidate_pairs.tsv complying with official rules.
"""
import os
import subprocess
import sys
from typing import Dict, Set, List

from src.config import MATCHING_RESULTS_PATH, CANDIDATE_PAIRS_PATH, OUTPUT_DIR, VALIDATOR_SCRIPT, TEST_DIR

def write_tsv_output(file_path: str, header_col1: str, header_col2: str, data_map: Dict[str, Set[str]], all_s1_ids: List[str]):
    """
    Writes a tab-separated output file with comma-separated ID lists.
    Guarantees every S1 entity appears exactly once.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    print(f"[Submission] Writing {len(all_s1_ids)} rows to {os.path.basename(file_path)}...", flush=True)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"{header_col1}\t{header_col2}\n")
        for s1_id in all_s1_ids:
            target_set = data_map.get(s1_id, set())
            if target_set:
                ids_str = ",".join(sorted(list(target_set)))
            else:
                ids_str = ""
            f.write(f"{s1_id}\t{ids_str}\n")
            
    print(f"[Submission] Successfully wrote {os.path.basename(file_path)}.", flush=True)

def generate_submission_files(matching_map: Dict[str, Set[str]], candidate_map: Dict[str, Set[str]], all_s1_ids: List[str]):
    """
    Generates matching_results.tsv and candidate_pairs.tsv in output/ directory.
    """
    write_tsv_output(
        MATCHING_RESULTS_PATH,
        "source1_entity_id",
        "matched_entity_ids",
        matching_map,
        all_s1_ids
    )
    
    write_tsv_output(
        CANDIDATE_PAIRS_PATH,
        "source1_entity_id",
        "candidate_entity_ids",
        candidate_map,
        all_s1_ids
    )

def run_official_validator(check_ids: bool = False) -> bool:
    """
    Executes official validation script utils/validate_submission.py.
    Returns True if PASS (exit code 0), False otherwise.
    """
    print("\n[Submission] Running Official Submission Validator...", flush=True)
    cmd = [
        sys.executable,
        VALIDATOR_SCRIPT,
        "--matching", MATCHING_RESULTS_PATH,
        "--candidate", CANDIDATE_PAIRS_PATH,
        "--test-dir", TEST_DIR,
    ]
    if check_ids:
        cmd.append("--check-ids")
        
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)
        
    if res.returncode == 0:
        print("[Submission] OFFICIAL VALIDATION RESULT: PASS! Output files are 100% compliant and safe to submit.")
        return True
    else:
        print("[Submission] OFFICIAL VALIDATION RESULT: FAIL. Please inspect error messages above.")
        return False
