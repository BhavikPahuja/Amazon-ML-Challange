"""Further boost candidate recall using alphanumeric address codes and character 3-grams."""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import re
import unicodedata
from collections import defaultdict

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
    'delhi', 'mumbai', 'bangalore', 'bengaluru', 'chennai', 'hyderabad', 'kolkata',
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

def extract_name_keys(name: str):
    norm = normalize_text(name)
    tokens = [t for t in norm.split() if t not in STOP_WORDS and len(t) > 1]
    keys = []
    for t in tokens[:4]:
        keys.append(('tok', t))
    compact = "".join(tokens[:3])
    if len(compact) >= 5:
        keys.append(('compact', compact))
    clean_all = norm.replace(" ", "")
    if len(clean_all) >= 4:
        keys.append(('pref4', clean_all[:4]))
    return keys

def extract_address_keys(addr: str):
    norm = normalize_text(addr)
    tokens = norm.split()
    keys = []
    
    # 1. Normalized numbers
    for i, tok in enumerate(tokens):
        if tok.isdigit() and len(tok) >= 2:
            norm_num = str(int(tok))
            adj_words = []
            if i + 1 < len(tokens) and not tokens[i+1].isdigit() and len(tokens[i+1]) > 2:
                adj_words.append(tokens[i+1])
            if i - 1 >= 0 and not tokens[i-1].isdigit() and len(tokens[i-1]) > 2:
                adj_words.append(tokens[i-1])
            for w in adj_words:
                if w not in ADDR_STOP_WORDS:
                    keys.append(('num_word', norm_num, w))
            if len(norm_num) >= 5:
                keys.append(('pin', norm_num))
                
    # 2. Distinctive address tokens (e.g. rare colony/locality names)
    rare_words = [t for t in tokens if not t.isdigit() and len(t) >= 4 and t not in ADDR_STOP_WORDS]
    for w in rare_words[:3]:
        keys.append(('addr_tok', w))
        
    return keys

# Load validation
print("Loading validation dataset...")
s1_records = {}
with open("student_resource/dataset/val/val_source1.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        s1_records[parts[0]] = {
            'id': parts[0],
            'name': parts[1] if len(parts) > 1 else '',
            'addr': parts[2] if len(parts) > 2 else '',
            'country': parts[3] if len(parts) > 3 else ''
        }

gt = {}
total_true_matches = 0
with open("student_resource/dataset/val/val_ground_truth.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        s1_id = parts[0]
        if len(parts) > 1 and parts[1]:
            matches = set(parts[1].split(","))
            gt[s1_id] = matches
            total_true_matches += len(matches)
        else:
            gt[s1_id] = set()

candidates = {}
for src_file in ["val_source2.tsv", "val_source3.tsv"]:
    path = os.path.join("student_resource/dataset/val", src_file)
    with open(path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            candidates[parts[0]] = {
                'id': parts[0],
                'name': parts[1] if len(parts) > 1 else '',
                'addr': parts[2] if len(parts) > 2 else '',
                'country': parts[3] if len(parts) > 3 else ''
            }

print(f"Loaded {len(candidates)} candidates. Building indices...")
index = defaultdict(list)

for cid, cand in candidates.items():
    c = cand['country']
    for k in extract_name_keys(cand['name']):
        index[(c, k)].append(cid)
    for k in extract_address_keys(cand['addr']):
        index[(c, k)].append(cid)

print("Querying S1 records...")
found_matches = 0
total_candidates_generated = 0

for s1_id, s1_rec in s1_records.items():
    true_set = gt[s1_id]
    c = s1_rec['country']
    cand_set = set()
    
    # Query name keys
    for k in extract_name_keys(s1_rec['name']):
        bucket = index.get((c, k), [])
        if len(bucket) <= 150:
            cand_set.update(bucket)
            
    # Query address keys
    for k in extract_address_keys(s1_rec['addr']):
        bucket = index.get((c, k), [])
        # Stricter limit on raw address words, looser on num_word / pin
        max_b = 50 if k[0] == 'addr_tok' else 100
        if len(bucket) <= max_b:
            cand_set.update(bucket)
            
    total_candidates_generated += len(cand_set)
    if true_set:
        found_matches += len(true_set & cand_set)

recall = (found_matches / total_true_matches) * 100 if total_true_matches > 0 else 0
avg_cands = total_candidates_generated / len(s1_records)
print(f"Blocking v3 Results:")
print(f"  Total True Matches: {total_true_matches}")
print(f"  Matches Captured in Candidate Set: {found_matches}")
print(f"  Candidate Set Recall Ceiling: {recall:.2f}%")
print(f"  Average Candidates per S1 Entity: {avg_cands:.1f}")
