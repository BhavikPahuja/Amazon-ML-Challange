# Business Entity Resolution Solution

## Overview
This package implements an end-to-end Machine Learning pipeline for the Amazon ML Challenge 2026: Business Entity Resolution.
Given business records from 3 independent, noisy sources (Source 1 reference, Source 2, and Source 3), the solution determines which records refer to the same real-world business entity.

## Architecture
1. **Blocking & Candidate Generation (`src/blocking.py`)**:
   - **Country-Partitioned Inverted Indexing**: Businesses never cross country borders. Partitioning by country guarantees 0 cross-country leakage while drastically reducing candidate pairs.
   - **Multi-Index Name Keys**: Strips legal suffixes (`Inc`, `LLC`, `Corp`, `Pvt Ltd`, `SARL`, `SAS`), removes noise symbols (`<<`, `--`, `##`), generates token shingles, compacted brand names (`@primemoney` -> `primemoney`), and prefix keys.
   - **Address Component Keys**: Normalizes house/street numbers by stripping leading zeros (`0017560` -> `17560`), binds numbers to adjacent street words, and extracts postal codes/PIN codes.
   - **Reduction Ratio**: > 99.95% reduction with 95.58% recall ceiling.

2. **Pairwise Matching Model (`src/train_matching_model.py`, `src/train.py`)**:
   - **Feature Engineering**: 15 features across name similarity (Token Jaccard, Token Set Ratio, Token Sort Ratio, Levenshtein Ratio, Compact Name Ratio, Exact Match), address similarity (Token Jaccard, Ratio, House/Street Number overlap, PIN match), and source indicators.
   - **Classifier**: LightGBM (MIT License, < 8 Billion parameters) trained on stratified pairs with positive/negative mining.
   - **Metric Optimization**: Evaluated directly on Macro $F_{0.5}$, penalizing false merges 2x more than missed matches. Optimal decision threshold: $\tau = 0.70$.

3. **Memory-Efficient Production Inference (`src/pipeline.py`)**:
   - Streams test records country-by-country (France, India, US).
   - Generates candidates and scores batches in vectorized C++ LightGBM runtime.
   - Outputs `matching_results.tsv` and `candidate_pairs.tsv` conforming to all submission rules.

## Requirements
- Python 3.8+
- Pinned packages in `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```

## Reproduction Instructions
From the `student_resource` directory:

1. **(Optional) Re-train the LightGBM model**:
   ```bash
   python code/business_entity_resolution/src/train.py
   ```
   This generates `lgb_matcher.txt` inside `src/`.

2. **Run End-to-End Test Inference**:
   ```bash
   python code/business_entity_resolution/src/pipeline.py \
       --test-dir dataset/test \
       --output-dir output
   ```
   This generates:
   - `output/matching_results.tsv`
   - `output/candidate_pairs.tsv`

3. **Validate Submission Format**:
   ```bash
   python utils/validate_submission.py \
       --matching output/matching_results.tsv \
       --candidate output/candidate_pairs.tsv \
       --test-dir dataset/test
   ```
