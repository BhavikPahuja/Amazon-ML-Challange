"""Train production LightGBM matching model on stratified training set and export model."""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import gc
import numpy as np
import lightgbm as lgb
from collections import defaultdict

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from blocking import normalize_text, extract_tokens, extract_numbers, FastBlocker, STOP_WORDS, ADDR_STOP_WORDS
from train_matching_model import compute_features

MODEL_PATH = os.path.join(SRC_DIR, "lgb_matcher.txt")

def train_production_model(s1_path: str, s2_path: str, s3_path: str, gt_path: str, sample_size: int = 40000):
    print(f"1. Loading {sample_size} stratified S1 records from {s1_path}...")
    s1_records = {}
    us_count = 0
    in_count = 0
    target_each = sample_size // 2
    
    with open(s1_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                eid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                if country == "US" and us_count < target_each:
                    clean_n = normalize_text(name)
                    clean_a = normalize_text(addr)
                    n_toks = extract_tokens(clean_n, STOP_WORDS)
                    a_toks = extract_tokens(clean_a, ADDR_STOP_WORDS)
                    s1_records[eid] = {
                        'id': eid, 'clean_name': clean_n, 'clean_addr': clean_a, 'country': country,
                        'name_tokens': n_toks, 'compact_name': "".join(n_toks[:3]),
                        'addr_tokens': a_toks, 'numbers': extract_numbers(clean_a.split())
                    }
                    us_count += 1
                elif country == "India" and in_count < target_each:
                    clean_n = normalize_text(name)
                    clean_a = normalize_text(addr)
                    n_toks = extract_tokens(clean_n, STOP_WORDS)
                    a_toks = extract_tokens(clean_a, ADDR_STOP_WORDS)
                    s1_records[eid] = {
                        'id': eid, 'clean_name': clean_n, 'clean_addr': clean_a, 'country': country,
                        'name_tokens': n_toks, 'compact_name': "".join(n_toks[:3]),
                        'addr_tokens': a_toks, 'numbers': extract_numbers(clean_a.split())
                    }
                    in_count += 1
                if us_count >= target_each and in_count >= target_each:
                    break

    s1_ids = set(s1_records.keys())
    print(f"Loaded {len(s1_records)} S1 records ({us_count} US, {in_count} India).")

    print("2. Reading ground truth labels...")
    gt = {}
    needed_s2 = set()
    needed_s3 = set()
    with open(gt_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            s1_id = parts[0]
            if s1_id in s1_ids:
                if len(parts) > 1 and parts[1]:
                    matches = set(parts[1].split(","))
                    gt[s1_id] = matches
                    for m in matches:
                        if m.startswith("S2-"): needed_s2.add(m)
                        elif m.startswith("S3-"): needed_s3.add(m)
                else:
                    gt[s1_id] = set()

    print(f"Ground truth parsed: {len(gt)} S1 entities. Needed S2: {len(needed_s2)}, Needed S3: {len(needed_s3)}")

    print("3. Loading candidates pool...")
    candidates = {}
    
    # Load needed S2 + 100,000 distractors
    s2_distract = 100000
    with open(s2_path, "r", encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            parts = line.strip().split("\t")
            eid = parts[0]
            if eid in needed_s2:
                clean_n = normalize_text(parts[1] if len(parts) > 1 else '')
                clean_a = normalize_text(parts[2] if len(parts) > 2 else '')
                n_toks = extract_tokens(clean_n, STOP_WORDS)
                a_toks = extract_tokens(clean_a, ADDR_STOP_WORDS)
                candidates[eid] = {
                    'id': eid, 'clean_name': clean_n, 'clean_addr': clean_a,
                    'country': parts[3] if len(parts) > 3 else '',
                    'name_tokens': n_toks, 'compact_name': "".join(n_toks[:3]),
                    'addr_tokens': a_toks, 'numbers': extract_numbers(clean_a.split())
                }
            elif s2_distract > 0 and (i % 50 == 0):
                clean_n = normalize_text(parts[1] if len(parts) > 1 else '')
                clean_a = normalize_text(parts[2] if len(parts) > 2 else '')
                n_toks = extract_tokens(clean_n, STOP_WORDS)
                a_toks = extract_tokens(clean_a, ADDR_STOP_WORDS)
                candidates[eid] = {
                    'id': eid, 'clean_name': clean_n, 'clean_addr': clean_a,
                    'country': parts[3] if len(parts) > 3 else '',
                    'name_tokens': n_toks, 'compact_name': "".join(n_toks[:3]),
                    'addr_tokens': a_toks, 'numbers': extract_numbers(clean_a.split())
                }
                s2_distract -= 1

    # Load needed S3 + 100,000 distractors
    s3_distract = 100000
    with open(s3_path, "r", encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            parts = line.strip().split("\t")
            eid = parts[0]
            if eid in needed_s3:
                clean_n = normalize_text(parts[1] if len(parts) > 1 else '')
                clean_a = normalize_text(parts[2] if len(parts) > 2 else '')
                n_toks = extract_tokens(clean_n, STOP_WORDS)
                a_toks = extract_tokens(clean_a, ADDR_STOP_WORDS)
                candidates[eid] = {
                    'id': eid, 'clean_name': clean_n, 'clean_addr': clean_a,
                    'country': parts[3] if len(parts) > 3 else '',
                    'name_tokens': n_toks, 'compact_name': "".join(n_toks[:3]),
                    'addr_tokens': a_toks, 'numbers': extract_numbers(clean_a.split())
                }
            elif s3_distract > 0 and (i % 50 == 0):
                clean_n = normalize_text(parts[1] if len(parts) > 1 else '')
                clean_a = normalize_text(parts[2] if len(parts) > 2 else '')
                n_toks = extract_tokens(clean_n, STOP_WORDS)
                a_toks = extract_tokens(clean_a, ADDR_STOP_WORDS)
                candidates[eid] = {
                    'id': eid, 'clean_name': clean_n, 'clean_addr': clean_a,
                    'country': parts[3] if len(parts) > 3 else '',
                    'name_tokens': n_toks, 'compact_name': "".join(n_toks[:3]),
                    'addr_tokens': a_toks, 'numbers': extract_numbers(clean_a.split())
                }
                s3_distract -= 1

    print(f"Candidates loaded: {len(candidates)}")

    print("4. Building blocking index...")
    blocker = FastBlocker()
    blocker.index_candidates(candidates)

    print("5. Generating pairs and computing features...")
    X, y = [], []
    for s1_id, s1 in s1_records.items():
        true_set = gt[s1_id]
        cand_set = blocker.get_candidates(s1)
        for cid in cand_set:
            cand = candidates[cid]
            feats = compute_features(s1, cand)
            label = 1 if cid in true_set else 0
            X.append(feats)
            y.append(label)

    del candidates
    del s1_records
    gc.collect()

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"Total training pairs: {len(X)} | Positives: {y.sum()} ({y.mean()*100:.2f}%)")

    print("6. Fitting production LightGBM model...")
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.06,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)

    print(f"7. Saving model to {MODEL_PATH}...")
    model.booster_.save_model(MODEL_PATH)
    print("Model successfully trained and saved!")

if __name__ == "__main__":
    train_production_model(
        s1_path="student_resource/dataset/train/train_source1.tsv",
        s2_path="student_resource/dataset/train/train_source2.tsv",
        s3_path="student_resource/dataset/train/train_source3.tsv",
        gt_path="student_resource/dataset/train/train_ground_truth.tsv",
        sample_size=30000
    )
