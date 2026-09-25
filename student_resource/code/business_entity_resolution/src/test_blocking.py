"""Test and optimize blocking/candidate generation strategies on validation set."""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import re
from collections import defaultdict

# Common business stop words & legal suffixes across US, India, France
STOP_WORDS = {
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited',
    'pvt', 'private', 'co', 'company', 'the', 'and', 'of', 'in', 'at',
    'group', 'services', 'service', 'solutions', 'solution', 'enterprises',
    'enterprise', 'center', 'centre', 'associates', 'holdings', 'holding',
    'consulting', 'management', 'international', 'systems', 'tech', 'technologies',
    'sarl', 'sasu', 'eurl', 'gmbh', 'sa', 'sas', 'sci', 'ste', 'societe',
    'pvtltd', 'india', 'us', 'usa', 'france', 'llp', 'pc', 'foundation'
}

def clean_text(text: str) -> str:
    if not text or text == 'nan':
        return ""
    # Remove leading noise symbols like <<, --, ##
    text = re.sub(r'^[<\-#\s]+', '', text)
    # Lowercase and replace non-alphanumeric with space
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return " ".join(text.split())

def extract_name_tokens(name: str):
    clean = clean_text(name)
    tokens = [t for t in clean.split() if t not in STOP_WORDS and len(t) > 1]
    return tokens

def extract_address_components(addr: str):
    clean = clean_text(addr)
    tokens = clean.split()
    # Extract numbers (house/building/PIN codes)
    numbers = [t for t in tokens if t.isdigit() and len(t) >= 2]
    # Significant words
    words = [t for t in tokens if not t.isdigit() and len(t) > 2]
    return numbers, words

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

print(f"Loaded {len(s1_records)} S1 records. Total true matches to retrieve: {total_true_matches}")

# Load S2 and S3 candidates
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

print(f"Loaded {len(candidates)} S2/S3 candidate records.")

# Index candidates by Country + Blocking Keys
print("Building multi-index blocking tables on candidates...")
# 1. Name tokens index: (country, token) -> list of candidate IDs
name_token_index = defaultdict(list)
# 2. Name prefix 4-char index: (country, prefix) -> list of candidate IDs
name_prefix_index = defaultdict(list)
# 3. Address number + first word index: (country, num, first_word) -> list of candidate IDs
addr_index = defaultdict(list)

for cid, cand in candidates.items():
    c = cand['country']
    # Name tokens
    ntoks = extract_name_tokens(cand['name'])
    for tok in ntoks[:4]: # top 4 significant tokens
        name_token_index[(c, tok)].append(cid)
    
    # Name prefix
    clean_n = clean_text(cand['name'])
    if len(clean_n) >= 4:
        pref = clean_n[:4]
        name_prefix_index[(c, pref)].append(cid)
        
    # Address tokens
    nums, awords = extract_address_components(cand['addr'])
    if nums and awords:
        addr_index[(c, nums[0], awords[0])].append(cid)

print("Blocking index built. Querying S1 records...")
found_matches = 0
total_candidates_generated = 0

for s1_id, s1_rec in s1_records.items():
    true_set = gt[s1_id]
    c = s1_rec['country']
    cand_set = set()
    
    # 1. Query name tokens
    ntoks = extract_name_tokens(s1_rec['name'])
    for tok in ntoks[:3]:
        bucket = name_token_index.get((c, tok), [])
        if len(bucket) <= 100:  # Skip huge buckets (too generic)
            cand_set.update(bucket)
            
    # 2. Query name prefix
    clean_n = clean_text(s1_rec['name'])
    if len(clean_n) >= 4:
        bucket = name_prefix_index.get((c, clean_n[:4]), [])
        if len(bucket) <= 50:
            cand_set.update(bucket)
            
    # 3. Query address number + word
    nums, awords = extract_address_components(s1_rec['addr'])
    if nums and awords:
        bucket = addr_index.get((c, nums[0], awords[0]), [])
        if len(bucket) <= 50:
            cand_set.update(bucket)
            
    total_candidates_generated += len(cand_set)
    if true_set:
        found = len(true_set & cand_set)
        found_matches += found

recall = (found_matches / total_true_matches) * 100 if total_true_matches > 0 else 0
avg_cands = total_candidates_generated / len(s1_records)
print(f"Results:")
print(f"  Total True Matches: {total_true_matches}")
print(f"  Matches Captured in Candidate Set: {found_matches}")
print(f"  Candidate Set Recall Ceiling: {recall:.2f}%")
print(f"  Average Candidates per S1 Entity: {avg_cands:.1f}")
