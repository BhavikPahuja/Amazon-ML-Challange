"""Diagnose missed matches from blocking to improve recall."""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import re
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

def clean_text(text: str) -> str:
    if not text or text == 'nan':
        return ""
    text = re.sub(r'^[<\-#\s]+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return " ".join(text.split())

def extract_name_tokens(name: str):
    clean = clean_text(name)
    tokens = [t for t in clean.split() if t not in STOP_WORDS and len(t) > 1]
    return tokens

def extract_address_components(addr: str):
    clean = clean_text(addr)
    tokens = clean.split()
    numbers = [t for t in tokens if t.isdigit() and len(t) >= 2]
    words = [t for t in tokens if not t.isdigit() and len(t) > 2]
    return numbers, words

# Load S1 records
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
with open("student_resource/dataset/val/val_ground_truth.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        s1_id = parts[0]
        if len(parts) > 1 and parts[1]:
            gt[s1_id] = set(parts[1].split(","))
        else:
            gt[s1_id] = set()

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

name_token_index = defaultdict(list)
name_prefix_index = defaultdict(list)
addr_index = defaultdict(list)

for cid, cand in candidates.items():
    c = cand['country']
    ntoks = extract_name_tokens(cand['name'])
    for tok in ntoks[:4]:
        name_token_index[(c, tok)].append(cid)
    
    clean_n = clean_text(cand['name'])
    if len(clean_n) >= 4:
        pref = clean_n[:4]
        name_prefix_index[(c, pref)].append(cid)
        
    nums, awords = extract_address_components(cand['addr'])
    if nums and awords:
        addr_index[(c, nums[0], awords[0])].append(cid)

missed_samples = []

for s1_id, s1_rec in s1_records.items():
    true_set = gt[s1_id]
    if not true_set:
        continue
    c = s1_rec['country']
    cand_set = set()
    
    ntoks = extract_name_tokens(s1_rec['name'])
    for tok in ntoks[:3]:
        bucket = name_token_index.get((c, tok), [])
        if len(bucket) <= 100:
            cand_set.update(bucket)
            
    clean_n = clean_text(s1_rec['name'])
    if len(clean_n) >= 4:
        bucket = name_prefix_index.get((c, clean_n[:4]), [])
        if len(bucket) <= 50:
            cand_set.update(bucket)
            
    nums, awords = extract_address_components(s1_rec['addr'])
    if nums and awords:
        bucket = addr_index.get((c, nums[0], awords[0]), [])
        if len(bucket) <= 50:
            cand_set.update(bucket)
            
    missed = true_set - cand_set
    for m in missed:
        cand_rec = candidates.get(m)
        if cand_rec:
            missed_samples.append((s1_rec, cand_rec))
            if len(missed_samples) >= 8:
                break
    if len(missed_samples) >= 8:
        break

print("=== 8 Sample Missed Matches ===")
for s1_rec, cand in missed_samples:
    print("-" * 70)
    print(f"S1: [{s1_rec['country']}] {s1_rec['id']}")
    print(f"    Name: {s1_rec['name']}")
    print(f"    Addr: {s1_rec['addr']}")
    print(f"Cand: [{cand['country']}] {cand['id']}")
    print(f"    Name: {cand['name']}")
    print(f"    Addr: {cand['addr']}")
