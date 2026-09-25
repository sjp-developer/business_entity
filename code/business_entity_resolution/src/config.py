"""
Configuration file for Amazon ML Challenge 2026 - Business Entity Resolution.
Contains all directory paths, hyperparameter settings, seeds, and feature flags.
"""
import os

# Base Directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

DATASET_DIR = os.path.join(BASE_DIR, "student_resource", "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

TRAIN_S1_PATH = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_S2_PATH = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_S3_PATH = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GT_PATH = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

TEST_S1_PATH = os.path.join(TEST_DIR, "test_source1.tsv")
TEST_S2_PATH = os.path.join(TEST_DIR, "test_source2.tsv")
TEST_S3_PATH = os.path.join(TEST_DIR, "test_source3.tsv")

# Output Paths
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MATCHING_RESULTS_PATH = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_PAIRS_PATH = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

# Official Validator Utility
VALIDATOR_SCRIPT = os.path.join(BASE_DIR, "student_resource", "utils", "validate_submission.py")

# Experiments & Logging
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")
LOG_PATH = os.path.join(EXPERIMENTS_DIR, "experiment_log.csv")

# Model & Random Seed
RANDOM_SEED = 42
N_JOBS = -1

# Legal Regex Pattern
LEGAL_RE = r"(?i)\b(inc|incorporated|corp|corporation|ltd|limited|pvt|private|llc|co|company|group|services|solutions|technologies|trust|associates|gmbh|sa|sarl|sas|eurl|spa|bv|nv|pte|plc)\b"
