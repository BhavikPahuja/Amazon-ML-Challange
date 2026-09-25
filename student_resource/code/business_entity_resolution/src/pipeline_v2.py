"""V2 Production Pipeline: Improved inference with better features and threshold.

Key changes from v1:
1. Uses 19-feature v2 feature vector (same as training)
2. Higher default threshold (0.85)
3. Tighter blocking bucket limits
4. Memory-efficient country-partitioned processing
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import gc
import time
import argparse
import numpy as np
import lightgbm as lgb
from collections import defaultdict
from rapidfuzz import fuzz

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from blocking import (
    normalize_text, extract_tokens, extract_numbers,
    extract_name_keys, extract_address_keys,
    STOP_WORDS, ADDR_STOP_WORDS
)

DEFAULT_MODEL_PATH = os.path.join(SRC_DIR, "lgb_matcher_v2.txt")

def compute_features_v2(clean_n1, clean_a1, clean_n2, clean_a2, cand_id):
    """Enhanced 19-feature vector matching train_v2."""
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


def run_pipeline_v2(
    test_dir: str,
    output_dir: str,
    model_path: str = DEFAULT_MODEL_PATH,
    threshold: float = 0.85,
    batch_size: int = 50000
):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")

    print(f"[{time.strftime('%X')}] Loading Booster model from {model_path}...", flush=True)
    booster = lgb.Booster(model_file=model_path)

    s1_file = os.path.join(test_dir, "test_source1.tsv")
    print(f"[{time.strftime('%X')}] Reading S1 test entities...", flush=True)

    s1_ordered_ids = []
    s1_data = {}
    s1_ids_by_country = defaultdict(list)

    with open(s1_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            eid = parts[0]
            s1_ordered_ids.append(eid)
            country = parts[3] if len(parts) > 3 else ''
            clean_n = normalize_text(parts[1] if len(parts) > 1 else '')
            clean_a = normalize_text(parts[2] if len(parts) > 2 else '')
            s1_data[eid] = (clean_n, clean_a)
            s1_ids_by_country[country].append(eid)

    print(f"[{time.strftime('%X')}] Total S1 entities: {len(s1_ordered_ids):,}", flush=True)
    for c, lst in s1_ids_by_country.items():
        print(f"  Country '{c}': {len(lst):,} entities", flush=True)

    s2_file = os.path.join(test_dir, "test_source2.tsv")
    s3_file = os.path.join(test_dir, "test_source3.tsv")

    matching_map = {}
    candidate_map = {}

    countries = sorted(list(s1_ids_by_country.keys()))
    for country in countries:
        s1_eids = s1_ids_by_country[country]
        print(f"\n[{time.strftime('%X')}] === Processing Country: {country} ({len(s1_eids):,} S1 entities) ===", flush=True)

        index = defaultdict(list)
        cand_data = {}

        for src_path in [s2_file, s3_file]:
            src_name = os.path.basename(src_path)
            print(f"[{time.strftime('%X')}] Indexing {src_name} for '{country}'...", flush=True)
            with open(src_path, "r", encoding="utf-8") as f:
                next(f)
                for line in f:
                    parts = line.strip().split("\t")
                    c = parts[3] if len(parts) > 3 else ''
                    if c == country:
                        eid = parts[0]
                        clean_n = normalize_text(parts[1] if len(parts) > 1 else '')
                        clean_a = normalize_text(parts[2] if len(parts) > 2 else '')
                        cand_data[eid] = (clean_n, clean_a)
                        for k in extract_name_keys(clean_n):
                            index[k].append(eid)
                        for k in extract_address_keys(clean_a):
                            index[k].append(eid)

        print(f"[{time.strftime('%X')}] Loaded {len(cand_data):,} candidates for '{country}'. Index: {len(index):,} keys.", flush=True)

        print(f"[{time.strftime('%X')}] Generating candidates and running batched inference...", flush=True)
        country_matches = {eid: [] for eid in s1_eids}
        country_cands = {eid: [] for eid in s1_eids}

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
                    country_matches[batch_s1_ids[i]].append(batch_cand_ids[i])
            batch_s1_ids.clear()
            batch_cand_ids.clear()
            batch_features.clear()

        for i, eid in enumerate(s1_eids):
            clean_n1, clean_a1 = s1_data[eid]
            cand_set = set()

            for k in extract_name_keys(clean_n1):
                b = index.get(k, [])
                if len(b) <= 100:
                    cand_set.update(b)

            for k in extract_address_keys(clean_a1):
                b = index.get(k, [])
                max_b = 40 if k[0] == 'addr_tok' else 80
                if len(b) <= max_b:
                    cand_set.update(b)

            sorted_cands = sorted(list(cand_set))
            country_cands[eid] = sorted_cands

            for cid in sorted_cands:
                clean_n2, clean_a2 = cand_data[cid]
                feats = compute_features_v2(clean_n1, clean_a1, clean_n2, clean_a2, cid)
                batch_s1_ids.append(eid)
                batch_cand_ids.append(cid)
                batch_features.append(feats)

                if len(batch_features) >= batch_size:
                    flush_batch()

            if (i + 1) % 50000 == 0 or (i + 1) == len(s1_eids):
                print(f"  Progress: {i + 1:,} / {len(s1_eids):,} entities processed...", flush=True)

        flush_batch()

        matching_map.update(country_matches)
        candidate_map.update(country_cands)

        del cand_data, index, country_matches, country_cands
        gc.collect()

    print(f"\n[{time.strftime('%X')}] Writing output files...", flush=True)

    with open(matching_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for eid in s1_ordered_ids:
            ms = matching_map.get(eid, [])
            f.write(f"{eid}\t{','.join(ms)}\n")
    print(f"[{time.strftime('%X')}] Saved {matching_path} ({len(s1_ordered_ids):,} rows)", flush=True)

    with open(candidate_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for eid in s1_ordered_ids:
            cs = candidate_map.get(eid, [])
            f.write(f"{eid}\t{','.join(cs)}\n")
    print(f"[{time.strftime('%X')}] Saved {candidate_path} ({len(s1_ordered_ids):,} rows)", flush=True)

    # Stats
    match_counts = [len(matching_map.get(eid, [])) for eid in s1_ordered_ids]
    singletons = match_counts.count(0)
    matched = len(match_counts) - singletons
    total_matches = sum(match_counts)
    print(f"\n  Singletons: {singletons:,} ({singletons/len(match_counts)*100:.1f}%)", flush=True)
    print(f"  Matched entities: {matched:,} ({matched/len(match_counts)*100:.1f}%)", flush=True)
    print(f"  Total match links: {total_matches:,}", flush=True)
    if matched > 0:
        non_zero = [c for c in match_counts if c > 0]
        print(f"  Mean matches (non-zero): {sum(non_zero)/len(non_zero):.2f}", flush=True)
        print(f"  Max matches: {max(non_zero)}", flush=True)

    elapsed = time.time() - start_time
    print(f"\n[{time.strftime('%X')}] Pipeline V2 finished in {elapsed/60:.2f} minutes!", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run V2 Business Entity Resolution Pipeline")
    parser.add_argument("--test-dir", default="student_resource/dataset/test")
    parser.add_argument("--output-dir", default="student_resource/output")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    run_pipeline_v2(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        threshold=args.threshold
    )
