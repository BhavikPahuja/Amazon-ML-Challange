"""Run fast batched end-to-end inference on validation set and evaluate."""

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
from metrics import evaluate_predictions

MODEL_PATH = os.path.join(SRC_DIR, "lgb_matcher.txt")

def run_fast_val_pipeline(threshold: float = 0.70):
    print("1. Loading Booster model...", flush=True)
    booster = lgb.Booster(model_file=MODEL_PATH)
    
    print("2. Reading S1 entities...", flush=True)
    s1_ordered_ids = []
    s1_dict = {}
    with open("student_resource/dataset/val/val_source1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            eid = parts[0]
            s1_ordered_ids.append(eid)
            clean_n = normalize_text(parts[1] if len(parts) > 1 else '')
            clean_a = normalize_text(parts[2] if len(parts) > 2 else '')
            n_toks = extract_tokens(clean_n, STOP_WORDS)
            a_toks = extract_tokens(clean_a, ADDR_STOP_WORDS)
            s1_dict[eid] = {
                'id': eid, 'clean_name': clean_n, 'clean_addr': clean_a,
                'country': parts[3] if len(parts) > 3 else '',
                'name_tokens': n_toks, 'compact_name': "".join(n_toks[:3]),
                'addr_tokens': a_toks, 'numbers': extract_numbers(clean_a.split())
            }

    print(f"Loaded {len(s1_ordered_ids)} validation S1 records.", flush=True)

    print("3. Reading candidate sources (val_source2, val_source3)...", flush=True)
    candidates = {}
    for s_file in ["val_source2.tsv", "val_source3.tsv"]:
        path = os.path.join("student_resource/dataset/val", s_file)
        with open(path, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.strip().split("\t")
                eid = parts[0]
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

    print(f"Loaded {len(candidates)} candidates.", flush=True)

    print("4. Indexing candidates with FastBlocker...", flush=True)
    blocker = FastBlocker()
    blocker.index_candidates(candidates)

    print("5. Generating candidates and predicting in batches...", flush=True)
    matching_results = {eid: [] for eid in s1_ordered_ids}
    candidate_pairs = {eid: [] for eid in s1_ordered_ids}
    
    BATCH_SIZE = 50000
    batch_s1_ids = []
    batch_cand_ids = []
    batch_features = []

    def flush_batch():
        if not batch_features:
            return
        X_batch = np.array(batch_features, dtype=np.float32)
        probs = booster.predict(X_batch)
        for i, p in enumerate(probs):
            if p >= threshold:
                matching_results[batch_s1_ids[i]].append(batch_cand_ids[i])
        batch_s1_ids.clear()
        batch_cand_ids.clear()
        batch_features.clear()

    for idx, eid in enumerate(s1_ordered_ids):
        s1 = s1_dict[eid]
        cand_set = blocker.get_candidates(s1)
        sorted_cands = sorted(list(cand_set))
        candidate_pairs[eid] = sorted_cands
        
        for cid in sorted_cands:
            cand = candidates[cid]
            feats = compute_features(s1, cand)
            batch_s1_ids.append(eid)
            batch_cand_ids.append(cid)
            batch_features.append(feats)
            
            if len(batch_features) >= BATCH_SIZE:
                flush_batch()
                
        if (idx + 1) % 5000 == 0:
            print(f"  Processed {idx + 1} / {len(s1_ordered_ids)} entities...", flush=True)

    flush_batch()
    print("Inference completed for all entities!", flush=True)

    print("6. Evaluating against Ground Truth...", flush=True)
    gt = {}
    with open("student_resource/dataset/val/val_ground_truth.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            s1_id = parts[0]
            if len(parts) > 1 and parts[1]:
                gt[s1_id] = set(parts[1].split(","))
            else:
                gt[s1_id] = set()

    pred_sets = {eid: set(ms) for eid, ms in matching_results.items()}
    score = evaluate_predictions(gt, pred_sets)
    print("==========================================", flush=True)
    print(f"Validation Macro F_0.5: {score['macro_f05']:.4f}", flush=True)
    print(f"Precision: {score['macro_precision']:.4f}, Recall: {score['macro_recall']:.4f}", flush=True)
    print(f"Singleton Accuracy: {score['singleton_acc']:.4f} ({score['singletons']} singletons)", flush=True)
    print("==========================================", flush=True)

if __name__ == "__main__":
    run_fast_val_pipeline(threshold=0.70)
