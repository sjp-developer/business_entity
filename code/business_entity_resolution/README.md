# Business Entity Resolution Pipeline — Amazon ML Challenge 2026

## Overview

This repository contains the competition-grade Multi-Stage Machine Learning Pipeline for the Amazon ML Challenge 2026 Business Entity Resolution problem.

The solution identifies matching records from Source 2 and Source 3 for every Source 1 entity using a precision-optimized candidate generation and LightGBM classification pipeline, specifically tuned to maximize Macro-Averaged F0.5.

## Architecture

```
RAW DATA (S1, S2, S3)
   ↓
SMART NORMALIZATION (Name / Address / Country)
   ↓
COUNTRY-PARTITIONED MULTI-BLOCKING (Exact Name, Word Tokens, House No / Postal Keys)
   ↓
CANDIDATE UNION & BUCKET CAPPING
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

## Directory Structure

```
code/business_entity_resolution/
├── src/
│   ├── config.py           # Master paths, parameters, random seeds
│   ├── data_loader.py      # Fast Polars TSV loading & S1 entity-level train/val split
│   ├── preprocessing.py    # Name, Address & Country smart normalization
│   ├── blocking.py         # Country-partitioned multi-key candidate generator
│   ├── features.py         # 21 high-precision pair similarity features
│   ├── model.py            # Supervised LightGBM classifier & threshold tuner
│   ├── evaluation.py       # Official macro F0.5, precision, recall & singleton metrics
│   ├── pipeline.py         # End-to-end execution pipeline
│   └── submission.py       # Output generator & validator invoker
├── README.md               # Pipeline documentation & reproduction guide
└── requirements.txt        # Pinned dependencies
```

## How to Reproduce

### 1. Environment Setup

Ensure Python 3.8+ is installed:

```bash
pip install -r code/business_entity_resolution/requirements.txt
```

### 2. Execution

Run the complete pipeline end-to-end:

```bash
python -m src.pipeline
```

This will automatically:
1. Load training and test datasets.
2. Generate candidate pairs via multi-blocking.
3. Compute 21 pairwise similarity features.
4. Train the LightGBM matching model on hard negative pairs.
5. Tune the decision threshold on validation data to maximize Macro F0.5.
6. Perform inference on the full test set.
7. Write `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
8. Execute the official `utils/validate_submission.py` validator script.

## Official Submission Validation

To manually run the official validator on generated output files:

```bash
python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```
