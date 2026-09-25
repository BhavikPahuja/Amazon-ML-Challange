"""Feature extraction, training, and threshold optimization for matching model."""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import re
import unicodedata
import numpy as np
import lightgbm as lgb
from collections import defaultdict
from rapidfuzz import fuzz

from metrics import evaluate_predictions

STOP_WORDS = {
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited',
    'pvt', 'private', 'co', 'company', 'the', 'and', 'of', 'in', 'at',
    'group', 'services', 'service', 'solutions', 'solution', 'enterprises',
    'enterprise', 'center', 'centre', 'associates', 'holdings', 'holding',
    'consulting', 'management', 'international', 'systems', 'tech', 'technologies',
    'sarl', 'sasu', 'eurl', 'gmbh', 'sa', 'sas', 'sci', 'ste', 'societe',
    'pvtltd', 'india', 'us', 'usa', 'france', 'llp', 'pc', 'foundation'
}

ADDR_STOP_WORDS = {
    'near', 'opp', 'opposite', 'road', 'rd', 'street', 'st', 'lane', 'ln',
    'drive', 'dr', 'avenue', 'ave', 'boulevard', 'blvd', 'floor', 'fl',
    'unit', 'suite', 'ste', 'building', 'bldg', 'block', 'blk', 'house',
    'flat', 'plot', 'no', 'hno', 'sector', 'sec', 'nagar', 'colony',
    'california', 'texas', 'florida', 'new', 'york', 'null', 'nan'
}

def normalize_text(text: str) -> str:
    if not text or text == 'nan':
        return ""
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'https?://|www\.', '', text)
    text = re.sub(r'\.(com|in|org|net|co|fr)\b', '', text)
    text = re.sub(r'^[<\-#@\s]+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return " ".join(text.split())

def extract_tokens(text: str, stop: set):
    return [t for t in text.split() if t not in stop and len(t) > 1]

def extract_numbers(tokens):
    nums = set()
    for t in tokens:
        if t.isdigit() and len(t) >= 2:
            nums.add(str(int(t)))
    return nums

def compute_features(s1: dict, cand: dict) -> list:
    n1 = s1['clean_name']
    n2 = cand['clean_name']
    a1 = s1['clean_addr']
    a2 = cand['clean_addr']
    
    # 1. Name features
    t1_name = s1['name_tokens']
    t2_name = cand['name_tokens']
    s1_set = set(t1_name)
    s2_set = set(t2_name)
    
    inter_name = len(s1_set & s2_set)
    union_name = len(s1_set | s2_set)
    name_jaccard = (inter_name / union_name) if union_name > 0 else 0.0
    
    name_ratio = fuzz.ratio(n1, n2) / 100.0
    name_token_set = fuzz.token_set_ratio(n1, n2) / 100.0
    name_token_sort = fuzz.token_sort_ratio(n1, n2) / 100.0
    name_exact = 1.0 if (n1 and n1 == n2) else 0.0
    
    # Compacted name similarity (e.g. primemoney vs primemoney)
    c1 = s1['compact_name']
    c2 = cand['compact_name']
    compact_ratio = fuzz.ratio(c1, c2) / 100.0 if (c1 and c2) else 0.0
    
    # 2. Address features
    addr_empty = 1.0 if (not a1 or not a2) else 0.0
    if not addr_empty:
        t1_addr = s1['addr_tokens']
        t2_addr = cand['addr_tokens']
        s1_a_set = set(t1_addr)
        s2_a_set = set(t2_addr)
        
        inter_addr = len(s1_a_set & s2_a_set)
        union_addr = len(s1_a_set | s2_a_set)
        addr_jaccard = (inter_addr / union_addr) if union_addr > 0 else 0.0
        
        addr_ratio = fuzz.ratio(a1, a2) / 100.0
        addr_token_set = fuzz.token_set_ratio(a1, a2) / 100.0
        
        # Number matching
        nums1 = s1['numbers']
        nums2 = cand['numbers']
        num_inter = len(nums1 & nums2)
        num_union = len(nums1 | nums2)
        num_jaccard = (num_inter / num_union) if num_union > 0 else (1.0 if not nums1 and not nums2 else 0.0)
        has_num_match = 1.0 if num_inter > 0 else 0.0
    else:
        addr_jaccard = 0.0
        addr_ratio = 0.0
        addr_token_set = 0.0
        inter_addr = 0
        num_jaccard = 0.0
        has_num_match = 0.0
        
    cand_src = 2.0 if cand['id'].startswith('S2-') else 3.0
    
    return [
        name_jaccard,
        name_ratio,
        name_token_set,
        name_token_sort,
        compact_ratio,
        name_exact,
        inter_name,
        addr_empty,
        addr_jaccard,
        addr_ratio,
        addr_token_set,
        inter_addr,
        num_jaccard,
        has_num_match,
        cand_src
    ]

FEATURE_NAMES = [
    'name_jaccard', 'name_ratio', 'name_token_set', 'name_token_sort',
    'compact_ratio', 'name_exact', 'inter_name', 'addr_empty',
    'addr_jaccard', 'addr_ratio', 'addr_token_set', 'inter_addr',
    'num_jaccard', 'has_num_match', 'cand_src'
]

print("1. Loading validation set...")
s1_records = {}
with open("student_resource/dataset/val/val_source1.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        clean_n = normalize_text(parts[1] if len(parts) > 1 else '')
        clean_a = normalize_text(parts[2] if len(parts) > 2 else '')
        n_toks = extract_tokens(clean_n, STOP_WORDS)
        a_toks = extract_tokens(clean_a, ADDR_STOP_WORDS)
        s1_records[parts[0]] = {
            'id': parts[0],
            'clean_name': clean_n,
            'clean_addr': clean_a,
            'country': parts[3] if len(parts) > 3 else '',
            'name_tokens': n_toks,
            'compact_name': "".join(n_toks[:3]),
            'addr_tokens': a_toks,
            'numbers': extract_numbers(clean_a.split())
        }

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

candidates = {}
for src_file in ["val_source2.tsv", "val_source3.tsv"]:
    path = os.path.join("student_resource/dataset/val", src_file)
    with open(path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            clean_n = normalize_text(parts[1] if len(parts) > 1 else '')
            clean_a = normalize_text(parts[2] if len(parts) > 2 else '')
            n_toks = extract_tokens(clean_n, STOP_WORDS)
            a_toks = extract_tokens(clean_a, ADDR_STOP_WORDS)
            candidates[parts[0]] = {
                'id': parts[0],
                'clean_name': clean_n,
                'clean_addr': clean_a,
                'country': parts[3] if len(parts) > 3 else '',
                'name_tokens': n_toks,
                'compact_name': "".join(n_toks[:3]),
                'addr_tokens': a_toks,
                'numbers': extract_numbers(clean_a.split())
            }

print(f"Loaded {len(s1_records)} S1 and {len(candidates)} candidates.")

# Split S1 into Train (14,000) and Val (6,000)
all_s1_ids = list(s1_records.keys())
train_s1_ids = set(all_s1_ids[:14000])
val_s1_ids = set(all_s1_ids[14000:])
print(f"Train S1 split: {len(train_s1_ids)}, Val S1 split: {len(val_s1_ids)}")

# Build blocking index
print("2. Building blocking index...")
from blocking import FastBlocker

blocker = FastBlocker()
blocker.index_candidates(candidates)

print("3. Generating candidate pairs and extracting features for training...")
X_train, y_train = [], []

for s1_id in train_s1_ids:
    s1 = s1_records[s1_id]
    true_matches = gt[s1_id]
    cand_set = blocker.get_candidates(s1)
            
    for cid in cand_set:
        cand = candidates[cid]
        feats = compute_features(s1, cand)
        is_match = 1 if cid in true_matches else 0
        X_train.append(feats)
        y_train.append(is_match)

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int32)
print(f"Training dataset: {len(X_train)} candidate pairs. Positives: {y_train.sum()} ({y_train.mean()*100:.2f}%)")

# Train LightGBM model
print("4. Training LightGBM classifier...")
model = lgb.LGBMClassifier(
    n_estimators=150,
    learning_rate=0.08,
    num_leaves=31,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate on Val S1 split
print("5. Generating candidates & predicting for Validation S1 split...")
val_pair_records = []
val_s1_cand_map = defaultdict(list)

for s1_id in val_s1_ids:
    s1 = s1_records[s1_id]
    cand_set = blocker.get_candidates(s1)
            
    for cid in cand_set:
        cand = candidates[cid]
        feats = compute_features(s1, cand)
        val_pair_records.append((s1_id, cid, feats))

print(f"Validation pairs: {len(val_pair_records)}")
X_val = np.array([r[2] for r in val_pair_records], dtype=np.float32)
preds_prob = model.predict_proba(X_val)[:, 1]

# Organize by S1
s1_scored_cands = defaultdict(list)
for i, (s1_id, cid, _) in enumerate(val_pair_records):
    s1_scored_cands[s1_id].append((cid, preds_prob[i]))

val_gt = {s1_id: gt[s1_id] for s1_id in val_s1_ids}

print("6. Optimizing threshold for Macro F_0.5...")
best_threshold = 0.5
best_f05 = 0.0
best_results = None

for threshold in [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]:
    predictions = {}
    for s1_id in val_s1_ids:
        matches = [cid for cid, p in s1_scored_cands[s1_id] if p >= threshold]
        predictions[s1_id] = set(matches)
        
    res = evaluate_predictions(val_gt, predictions)
    f05 = res['macro_f05']
    print(f"  Threshold {threshold:.2f} -> Macro F_0.5: {f05:.4f} | Prec: {res['macro_precision']:.4f} | Rec: {res['macro_recall']:.4f} | Singleton Acc: {res['singleton_acc']:.4f}")
    if f05 > best_f05:
        best_f05 = f05
        best_threshold = threshold
        best_results = res

print(f"\n==========================================")
print(f"Best Validation Macro F_0.5: {best_f05:.4f} at Threshold: {best_threshold:.2f}")
print(f"Precision: {best_results['macro_precision']:.4f}, Recall: {best_results['macro_recall']:.4f}")
print(f"==========================================")
