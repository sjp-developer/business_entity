# Amazon ML Challenge 2026: Business Entity Resolution Solution Document

**Challenge Name:** Amazon ML Challenge 2026 — Business Entity Resolution  
**Repository:** [https://github.com/sjp-developer/business_entity.git](https://github.com/sjp-developer/business_entity.git)  
**Task:** Identify all true matching records across Source 2 and Source 3 for each Source 1 business entity.  
**Primary Evaluation Metric:** Macro-Averaged F0.5 Score per Source 1 entity (Precision weighted 2× over Recall).  

---

## 1. Executive Summary

We developed an end-to-end, high-performance Machine Learning system for large-scale Business Entity Resolution across 12.5 million training records and 11.7 million test records. Our approach combines **country-partitioned vectorized multi-blocking** with **pre-allocated zero-copy NumPy feature engineering** and a **supervised LightGBM pairwise matching classifier**. 

Our key innovation is an empirically validated **country-partitioning invariance** (0.0000% cross-country matches across 700,000 ground-truth pairs), which reduces candidate search space and memory overhead by over 75% without any loss of recall. The system achieves a **Validation Macro F0.5 score of 0.7436 (74.36%)** with **82.22% Precision**, while strictly adhering to system memory limits and execution constraints.

---

## 2. Methodology

### 2.1 Problem Analysis & Exploratory Data Analysis (EDA)
- **Dataset Scale**: Analyzed 7 TSV tables comprising 10.3M Source 1 entities, 1.0M Source 2 entities, 1.1M Source 3 entities, and 700,000 true ground-truth match pairs.
- **Completeness**: Evaluated field missingness across all tables. `business_name` and `country` are 100% complete (0% missing), while `business_address` has ~1.2% missingness.
- **Noise Patterns**: Observed significant address variations (abbreviations like `St` vs `Street`, missing zip codes), OCR noise, character transpositions, and legal entity suffix noise (`LLC`, `Inc`, `Pvt Ltd`, `SARL`).
- **Country Partition Invariance**: Tested 700,000 true ground-truth matching pairs and confirmed that **0.0000% of matches occur across different countries**. This enabled us to partition blocking and inference strictly by country (`US`, `India`, `France`).

### 2.2 Solution Strategy
- **Approach Type**: Hybrid Multi-Blocking Candidate Generator + Supervised LightGBM Pairwise Classifier + Macro F0.5 Threshold Optimizer.
- **Core Innovations**:
  1. **Country-Partitioned Multi-Blocking**: Vectorized key generation using Polars lazy frames (`pl.scan_csv`).
  2. **Zero-Copy NumPy Similarity Extractor**: Direct computation into pre-allocated `float32` arrays, achieving 100,000 pairs/second without heap memory allocation overhead.
  3. **Pairwise Feature-Target Alignment**: Direct propagation of entity text fields inside candidate dataframes to prevent index misalignment during multi-thread joins.
  4. **Entity-Level Stratified Validation**: Leakage-free train/validation split at the Source 1 entity level.

---

## 3. Candidate Generation (Blocking)

To reduce the $O(N_1 \times (N_2 + N_3))$ comparison space of tens of trillions of pairs into a manageable candidate set, we engineered a multi-rule vectorized blocker in `src/blocking.py`:

- **Blocking Rules Implemented**:
  1. **Exact Normalized Name**: Case-folded, legal-suffix-stripped, and punctuation-cleaned name match.
  2. **Token 1 + House Number**: First name token ($\ge 3$ chars) combined with extracted street building number.
  3. **Token 1 + Postal Code**: First name token combined with 5/6 digit postal zip code.
  4. **Prefix-5 + House Number**: First 5 characters of normalized name + house number.
  5. **Capped Token 1 + Token 2**: First two name tokens with frequency capping ($\le 400$ occurrences).
  6. **Rare Token 1**: Distinctive first token with frequency capping ($\le 80$ occurrences).

- **Candidate Generation Metrics**:
  - **Validation Candidate Recall**: **71.91% – 72.03%**
  - **Average Candidates per S1 Entity**: **83.29 candidates**
  - **Execution Time**: Candidate generation across 12.5M records completes in **~25 seconds**.

---

## 4. Matching Model

### 4.1 Feature Engineering Matrix (14 Signals)
Each candidate pair $(e_1, e_2)$ is converted into a 14-dimensional feature vector in `src/features.py`:

| Feature Name | Feature Category | Description | Feature Importance |
| :--- | :--- | :--- | :--- |
| `name_levenshtein_ratio` | Name Similarity | Normalized Levenshtein edit similarity ratio | **1,453** |
| `addr_len_diff` | Address Distance | Absolute string length difference in address | **1,128** |
| `name_len_diff` | Name Distance | Absolute string length difference in name | **1,052** |
| `addr_token_set_ratio` | Address Similarity | RapidFuzz token set similarity ratio | **934** |
| `addr_levenshtein_ratio` | Address Similarity | Normalized Levenshtein edit ratio for addresses | **915** |
| `name_token_sort_ratio` | Name Similarity | RapidFuzz token sort similarity ratio | **914** |
| `name_token_set_ratio` | Name Similarity | RapidFuzz token set similarity ratio | **838** |
| `name_addr_sim_prod` | Interaction | Product of `name_token_set` and `addr_token_set` | **482** |
| `name_addr_sim_max` | Interaction | Maximum of name and address similarity ratios | **366** |
| `name_addr_sim_min` | Interaction | Minimum of name and address similarity ratios | **363** |
| `is_source2` | Provenance | Binary indicator (1 if Source 2, 0 if Source 3) | **348** |
| `addr_is_missing` | Missingness | Binary indicator if either address field is null | **139** |
| `addr_exact_match` | Address Match | Binary exact string equality flag for address | **41** |
| `name_exact_match` | Name Match | Binary exact string equality flag for name | **3** |

### 4.2 Model Architecture & Training
- **Model Type**: LightGBM Gradient Boosted Decision Trees (`LGBMClassifier`).
- **Hard Negative Sampling**: Sampled 5 hard negative candidate pairs for every positive ground-truth match pair (5:1 ratio), providing sharp decision boundary learning.
- **Hyperparameters**: `n_estimators=300`, `learning_rate=0.05`, `max_depth=6`, `num_leaves=31`.
- **Decision Threshold Optimization**: Performed grid search across threshold range $[0.20, 0.84]$ on validation set to directly maximize per-S1 Macro F0.5.

---

## 5. Results & Error Analysis

### 5.1 Final Validation Metrics

```
==================================================
FINAL VALIDATION RESULTS:
  Macro F0.5 Score   : 0.7436 (74.36%)
  Macro Precision    : 0.8222 (82.22%)
  Macro Recall       : 0.6177 (61.77%)
  Singleton Accuracy : 65.35%
  Optimal Threshold  : 0.84
==================================================
```

### 5.2 Error Analysis
- **False Positives (Precision Loss)**: Occur primarily when multiple branch locations of a single franchise (e.g., `"State Farm Insurance"`) share identical business names in the same city but differ only by suite/apartment numbers.
- **False Negatives (Recall Loss)**: Occur when business names undergo heavy acronym transformations (e.g., `"International Business Machines"` vs `"IBM"`), which are currently missed by token-level blocking rules.

---

## 6. Conclusion

We built a modular, fast, and compliant ML system for the Amazon ML Challenge 2026. By combining country-partitioned multi-blocking with zero-copy NumPy feature extraction and LightGBM classification, we achieved a **Macro F0.5 validation score of 0.7436** with **82.22% Precision**.

---

## Appendix A: Code Artefacts & Project Structure

The codebase is organized in `code/business_entity_resolution/`:

```
code/business_entity_resolution/
├── README.md                  # Comprehensive reproduction guide
├── requirements.txt           # Pinned Python dependencies
└── src/                       # Core ML source code
    ├── __init__.py            # Package initialization
    ├── config.py              # Master paths, parameters, seeds, regexes
    ├── data_loader.py         # Polars lazy frame readers & stratified S1 splitter
    ├── preprocessing.py       # Legal suffix stripping & address normalization
    ├── blocking.py            # Vectorized multi-blocking candidate generator
    ├── features.py            # Zero-copy float32 NumPy similarity matrix extractor
    ├── model.py               # LightGBM classifier & threshold optimizer
    ├── evaluation.py          # Macro F0.5 per-S1 evaluator
    ├── submission.py          # File generator & official validator runner
    └── pipeline.py            # Master end-to-end execution orchestrator
```

### Execution Command
To execute the complete end-to-end pipeline and reproduce all validation and submission outputs:

```bash
cd code/business_entity_resolution
python -m src.pipeline
```

---

## Appendix B: Output Specifications & Official Validator Compliance

The pipeline generates two output files in `output/` complying with official challenge requirements:

1. **`output/matching_results.tsv`**:
   - Format: `source1_entity_id \t matched_entity_ids` (comma-separated list of true predicted matches).
   - Guarantees every Source 1 entity in test set appears exactly once.
2. **`output/candidate_pairs.tsv`**:
   - Format: `source1_entity_id \t candidate_entity_ids` (comma-separated list of evaluated candidate IDs).
3. **Official Validation**:
   - Output files pass the official script `student_resource/utils/validate_submission.py` with status `PASS`.
