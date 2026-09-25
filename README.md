# Amazon ML Challenge 2026 — Business Entity Resolution

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official Machine Learning solution for the **Amazon ML Challenge 2026 — Business Entity Resolution Problem**.

## Overview

In commercial platforms, business identity data arrives from multiple independent sources with partial, noisy, and unstandardized fragments. Source 1 is the deduplicated reference source. For every Source 1 record, this pipeline identifies all true matching records from Source 2 and Source 3.

The evaluation metric is **Macro-Averaged F0.5 Score** per Source 1 entity (weighting precision 2× over recall).

## Architecture

```
RAW DATA (S1, S2, S3)
   ↓
SMART NORMALIZATION (Name / Address / Country)
   ↓
COUNTRY-PARTITIONED MULTI-BLOCKING (Exact Name, Word Tokens, House No & Postal Keys)
   ↓
LIGHTWEIGHT CANDIDATE UNION & GROUP CAPPING
   ↓
HIGH-DIMENSIONAL PAIR FEATURE ENGINEERING (21 features: Levenshtein, Jaccard, RapidFuzz, Interaction)
   ↓
SUPERVISED LIGHTGBM MATCHING MODEL (Hard Negative Mining)
   ↓
S1-ENTITY MACRO F0.5 THRESHOLD OPTIMIZATION & SINGLETON HANDLING
   ↓
OUTPUT GENERATION (matching_results.tsv & candidate_pairs.tsv)
   ↓
OFFICIAL VALIDATOR VERIFICATION (utils/validate_submission.py)
```

## Repository Structure

```
.
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── config.py           # Master configuration & hyperparameters
│       │   ├── data_loader.py      # Fast Polars TSV loading & S1 entity-level split
│       │   ├── preprocessing.py    # Name, Address & Country smart normalization
│       │   ├── blocking.py         # Country-partitioned multi-key candidate generator
│       │   ├── features.py         # Parallel 21-feature similarity extractor
│       │   ├── model.py            # Supervised LightGBM classifier & threshold tuner
│       │   ├── evaluation.py       # Macro F0.5, precision, recall & singleton evaluator
│       │   ├── pipeline.py         # End-to-end execution pipeline
│       │   └── submission.py       # Submission generator & official validator invoker
│       ├── README.md               # Code documentation
│       └── requirements.txt        # Pinned dependencies
├── output/
│   ├── matching_results.tsv        # Final entity matches (Leaderboard scored)
│   └── candidate_pairs.tsv         # Final candidate pairs fed to model
├── student_resource/
│   ├── Documentation_template.md  # Official methodology report
│   ├── README.md                  # Challenge problem statement
│   └── utils/
│       └── validate_submission.py # Official submission validator
├── experiments/
│   └── experiment_log.csv         # Experiment tracking log
└── .gitignore
```

## Quick Start & Reproduction

### 1. Install Dependencies

```bash
pip install -r code/business_entity_resolution/requirements.txt
```

### 2. Run End-to-End Pipeline

```bash
cd code/business_entity_resolution
python -m src.pipeline
```

### 3. Validate Submission

```bash
python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```
