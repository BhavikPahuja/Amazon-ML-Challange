"""V2 Training: Train on realistic blocking conditions with full data.

Root causes of 0.6477:
1. Training on 40K entities with 50K distractors (toy)
2. Real test has 5M+ candidates per source = WAY more false positives
3. France never seen in training
4. Need to train with realistic neg:pos ratio from full blocking

Solution: 
- Use 100K+ S1 entities (stratified US + India)
- Use FULL S2/S3 source files for blocking (realistic neg ratio)
- Add more discriminative features
- Train with higher capacity model
- Use higher threshold
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import gc
import time
import random
import numpy as np
import lightgbm as lgb
from collections import defaultdict, Counter
from rapidfuzz import fuzz

random.seed(42)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from blocking import normalize_text, extract_tokens, extract_numbers, STOP_WORDS, ADDR_STOP_WORDS
from blocking import extract_name_keys, extract_address_keys

# ---------- IMPROVED FEATURE COMPUTATION ----------
def compute_features_v2(clean_n1, clean_a1, clean_n2, clean_a2, cand_id):
    """Enhanced 19-feature vector."""
    t1_name = [t for t in clean_n1.split() if t not in STOP_WORDS and len(t) > 1]
    t2_name = [t for t in clean_n2.split() if t not in STOP_WORDS and len(t) > 1]
    s1_set = set(t1_name)
    s2_set = set(t2_name)

    inter_name = len(s1_set & s2_set)
    union_name = len(s1_set | s2_set)
    name_jaccard = (inter_name / union_name) if union_name > 0 else 0.0
    name_ratio = fuzz.ratio(clean_n1, clean_n2) / 100.0
    name_token_set = fuzz.token_set_ratio(clean_n1, clean_n2) / 100.0
    name_token_sort = fuzz.token_sort_ratio(clean_n1, clean_n2) / 100.0
    name_exact = 1.0 if (clean_n1 and clean_n1 == clean_n2) else 0.0

    c1 = "".join(t1_name[:3])
    c2 = "".join(t2_name[:3])
    compact_ratio = fuzz.ratio(c1, c2) / 100.0 if (c1 and c2) else 0.0

    name_partial = fuzz.partial_ratio(clean_n1, clean_n2) / 100.0 if (clean_n1 and clean_n2) else 0.0
    
    len1 = max(len(clean_n1), 1)
    len2 = max(len(clean_n2), 1)
    name_len_ratio = min(len1, len2) / max(len1, len2)

    def char_trigrams(s):
        if len(s) < 3: return set()
        return {s[i:i+3] for i in range(len(s) - 2)}
    tg1 = char_trigrams(clean_n1)
    tg2 = char_trigrams(clean_n2)
    tg_inter = len(tg1 & tg2)
    tg_union = len(tg1 | tg2)
    name_trigram_jaccard = (tg_inter / tg_union) if tg_union > 0 else 0.0

    addr_empty = 1.0 if (not clean_a1 or not clean_a2) else 0.0
    if not addr_empty:
        t1_addr = [t for t in clean_a1.split() if t not in ADDR_STOP_WORDS and len(t) > 1]
        t2_addr = [t for t in clean_a2.split() if t not in ADDR_STOP_WORDS and len(t) > 1]
        s1_a_set = set(t1_addr)
        s2_a_set = set(t2_addr)
        inter_addr = len(s1_a_set & s2_a_set)
        union_addr = len(s1_a_set | s2_a_set)
        addr_jaccard = (inter_addr / union_addr) if union_addr > 0 else 0.0
        addr_ratio = fuzz.ratio(clean_a1, clean_a2) / 100.0
        addr_token_set = fuzz.token_set_ratio(clean_a1, clean_a2) / 100.0
        nums1 = {str(int(t)) for t in clean_a1.split() if t.isdigit() and len(t) >= 2}
        nums2 = {str(int(t)) for t in clean_a2.split() if t.isdigit() and len(t) >= 2}
        num_inter = len(nums1 & nums2)
        num_union = len(nums1 | nums2)
        num_jaccard = (num_inter / num_union) if num_union > 0 else (1.0 if not nums1 and not nums2 else 0.0)
        has_num_match = 1.0 if num_inter > 0 else 0.0
        addr_partial = fuzz.partial_ratio(clean_a1, clean_a2) / 100.0
    else:
        addr_jaccard = addr_ratio = addr_token_set = 0.0
        inter_addr = 0
        num_jaccard = has_num_match = addr_partial = 0.0

    cand_src = 2.0 if cand_id.startswith('S2-') else 3.0

    return [
        name_jaccard, name_ratio, name_token_set, name_token_sort,
        compact_ratio, name_exact, inter_name, name_partial,
        name_len_ratio, name_trigram_jaccard,
        addr_empty, addr_jaccard, addr_ratio, addr_token_set,
        inter_addr, num_jaccard, has_num_match, addr_partial,
        cand_src
    ]

FEATURE_NAMES_V2 = [
    'name_jaccard', 'name_ratio', 'name_token_set', 'name_token_sort',
    'compact_ratio', 'name_exact', 'inter_name', 'name_partial',
    'name_len_ratio', 'name_trigram_jaccard',
    'addr_empty', 'addr_jaccard', 'addr_ratio', 'addr_token_set',
    'inter_addr', 'num_jaccard', 'has_num_match', 'addr_partial',
    'cand_src'
]

def main():
    t0 = time.time()
    train_dir = "student_resource/dataset/train"
    
    s1_file = os.path.join(train_dir, "train_source1.tsv")
    s2_file = os.path.join(train_dir, "train_source2.tsv")
    s3_file = os.path.join(train_dir, "train_source3.tsv")
    gt_file = os.path.join(train_dir, "train_ground_truth.tsv")
    
    SAMPLE_SIZE = 100000  # 100K S1 entities

    # ---- STEP 1: Sample S1 entities ----
    print(f"[{time.strftime('%X')}] Step 1: Sampling {SAMPLE_SIZE} S1 entities...", flush=True)
    
    # First count
    all_s1_by_country = defaultdict(list)
    with open(s1_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                all_s1_by_country[parts[3]].append(parts[0])
    
    for c, ids in all_s1_by_country.items():
        print(f"  {c}: {len(ids):,} total", flush=True)
    
    # Proportional sample
    total = sum(len(v) for v in all_s1_by_country.values())
    sampled_ids = set()
    for c, ids in all_s1_by_country.items():
        budget = int(SAMPLE_SIZE * len(ids) / total)
        random.shuffle(ids)
        sampled_ids.update(ids[:budget])
    
    print(f"  Sampled: {len(sampled_ids):,} S1 entities", flush=True)
    
    # ---- STEP 2: Load sampled S1 data ----
    print(f"[{time.strftime('%X')}] Step 2: Loading sampled S1 data...", flush=True)
    s1_data = {}  # eid -> (clean_n, clean_a, country)
    with open(s1_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if parts[0] in sampled_ids:
                clean_n = normalize_text(parts[1] if len(parts) > 1 else '')
                clean_a = normalize_text(parts[2] if len(parts) > 2 else '')
                country = parts[3] if len(parts) > 3 else ''
                s1_data[parts[0]] = (clean_n, clean_a, country)
    
    print(f"  Loaded {len(s1_data):,} S1 records", flush=True)

    # ---- STEP 3: Load GT ----
    print(f"[{time.strftime('%X')}] Step 3: Loading ground truth...", flush=True)
    gt = {}
    needed_cand_ids = set()
    with open(gt_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if parts[0] in s1_data:
                if len(parts) > 1 and parts[1]:
                    matches = set(parts[1].split(","))
                    gt[parts[0]] = matches
                    needed_cand_ids.update(matches)
                else:
                    gt[parts[0]] = set()
    
    print(f"  GT loaded: {len(gt)} entities, {len(needed_cand_ids):,} needed cand IDs", flush=True)

    # ---- STEP 4: Process country-by-country ----
    # This is the KEY change: use full S2/S3 files for realistic blocking
    countries = sorted(set(v[2] for v in s1_data.values()))
    print(f"[{time.strftime('%X')}] Step 4: Processing by country: {countries}", flush=True)
    
    all_X = []
    all_y = []
    total_positives_found = 0
    total_positives_missed = 0
    
    for country in countries:
        s1_eids = [eid for eid, (cn, ca, c) in s1_data.items() if c == country]
        print(f"\n[{time.strftime('%X')}] === Country: {country} ({len(s1_eids):,} S1) ===", flush=True)
        
        # Build inverted index from FULL S2/S3 for this country
        index = defaultdict(list)
        cand_data = {}
        
        for src_path in [s2_file, s3_file]:
            src_name = os.path.basename(src_path)
            count = 0
            with open(src_path, "r", encoding="utf-8") as f:
                next(f)
                for line in f:
                    parts = line.strip().split("\t")
                    c = parts[3] if len(parts) > 3 else ''
                    if c != country:
                        continue
                    eid = parts[0]
                    clean_n = normalize_text(parts[1] if len(parts) > 1 else '')
                    clean_a = normalize_text(parts[2] if len(parts) > 2 else '')
                    cand_data[eid] = (clean_n, clean_a)
                    
                    for k in extract_name_keys(clean_n):
                        index[k].append(eid)
                    for k in extract_address_keys(clean_a):
                        index[k].append(eid)
                    count += 1
            print(f"  Indexed {count:,} from {src_name}", flush=True)
        
        print(f"  Total candidates: {len(cand_data):,}, Index keys: {len(index):,}", flush=True)
        
        # Generate pairs using blocking
        country_X = []
        country_y = []
        pos_found = 0
        pos_missed = 0
        
        for idx, eid in enumerate(s1_eids):
            cn1, ca1, _ = s1_data[eid]
            true_matches = gt.get(eid, set())
            
            # Block
            cand_set = set()
            for k in extract_name_keys(cn1):
                b = index.get(k, [])
                if len(b) <= 100:
                    cand_set.update(b)
            for k in extract_address_keys(ca1):
                b = index.get(k, [])
                max_b = 40 if k[0] == 'addr_tok' else 80
                if len(b) <= max_b:
                    cand_set.update(b)
            
            # Track blocking recall
            for tp_id in true_matches:
                if tp_id in cand_data:
                    if tp_id in cand_set:
                        pos_found += 1
                    else:
                        pos_missed += 1
                        cand_set.add(tp_id)  # Include anyway for training
            
            # Extract features for ALL blocked candidates
            for cid in cand_set:
                if cid not in cand_data:
                    continue
                cn2, ca2 = cand_data[cid]
                feats = compute_features_v2(cn1, ca1, cn2, ca2, cid)
                label = 1 if cid in true_matches else 0
                country_X.append(feats)
                country_y.append(label)
            
            if (idx + 1) % 10000 == 0:
                print(f"  Progress: {idx+1}/{len(s1_eids)} entities, {len(country_X)} pairs...", flush=True)
        
        all_X.extend(country_X)
        all_y.extend(country_y)
        total_positives_found += pos_found
        total_positives_missed += pos_missed
        
        cy = np.array(country_y)
        print(f"  Pairs: {len(country_X):,} (pos: {int(cy.sum()):,}, neg: {len(cy)-int(cy.sum()):,})", flush=True)
        print(f"  Blocking recall: {pos_found}/{pos_found+pos_missed}", flush=True)
        
        # Free memory
        del cand_data, index, country_X, country_y
        gc.collect()
    
    # ---- STEP 5: Train model ----
    print(f"\n[{time.strftime('%X')}] Step 5: Preparing training data...", flush=True)
    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int32)
    del all_X, all_y
    gc.collect()
    
    pos_count = int(y.sum())
    neg_count = len(y) - pos_count
    print(f"  Total pairs: {len(X):,}", flush=True)
    print(f"  Positives: {pos_count:,} ({pos_count/len(y)*100:.3f}%)", flush=True)
    print(f"  Negatives: {neg_count:,}", flush=True)
    print(f"  Overall blocking recall: {total_positives_found}/{total_positives_found+total_positives_missed}", flush=True)
    
    scale_pos = neg_count / max(pos_count, 1)
    print(f"  scale_pos_weight: {scale_pos:.2f}", flush=True)
    
    print(f"\n[{time.strftime('%X')}] Step 6: Training LightGBM...", flush=True)
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=100,
        scale_pos_weight=scale_pos,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(X, y, feature_name=FEATURE_NAMES_V2)
    
    # Feature importance
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\n  Feature Importances:", flush=True)
    for i in sorted_idx:
        print(f"    {FEATURE_NAMES_V2[i]:25s}: {importances[i]}", flush=True)
    
    # ---- STEP 7: Threshold search ----
    print(f"\n[{time.strftime('%X')}] Step 7: Threshold search...", flush=True)
    probs = model.predict_proba(X)[:, 1]
    
    for threshold in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        preds = (probs >= threshold).astype(int)
        tp = int(((preds == 1) & (y == 1)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        fn = int(((preds == 0) & (y == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f05 = 1.25 * prec * rec / (0.25 * prec + rec) if (prec + rec) > 0 else 0
        print(f"  τ={threshold:.2f}: Prec={prec:.4f} Rec={rec:.4f} F0.5={f05:.4f} (TP={tp:,} FP={fp:,} FN={fn:,})", flush=True)
    
    # Save model
    model_path = os.path.join(SRC_DIR, "lgb_matcher_v2.txt")
    model.booster_.save_model(model_path)
    print(f"\n[{time.strftime('%X')}] Model saved to {model_path}", flush=True)
    print(f"Total training time: {(time.time()-t0)/60:.1f} minutes", flush=True)

if __name__ == "__main__":
    main()
