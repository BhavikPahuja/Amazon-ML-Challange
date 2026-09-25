"""Create a stratified validation split from the training dataset.

Selects 10,000 US and 10,000 India Source 1 entities (total 20,000 S1 entities),
collects their ground truth matches, and extracts all corresponding S2/S3 entities
plus negative distractor entities to form a realistic validation benchmark.
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import random
import pandas as pd

random.seed(42)

VAL_DIR = "student_resource/dataset/val"
os.makedirs(VAL_DIR, exist_ok=True)

print("1. Reading S1 entities...")
us_s1 = []
india_s1 = []

with open("student_resource/dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
    header = f.readline().strip()
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 4:
            s1_id, name, addr, country = parts[0], parts[1], parts[2], parts[3]
            if country == "US" and len(us_s1) < 10000:
                us_s1.append(line)
            elif country == "India" and len(india_s1) < 10000:
                india_s1.append(line)
            if len(us_s1) >= 10000 and len(india_s1) >= 10000:
                break

val_s1_lines = us_s1 + india_s1
val_s1_ids = {line.split("\t")[0] for line in val_s1_lines}
print(f"Selected {len(val_s1_ids)} S1 validation entities (10k US, 10k India).")

with open(os.path.join(VAL_DIR, "val_source1.tsv"), "w", encoding="utf-8") as f:
    f.write(header + "\n")
    for line in val_s1_lines:
        f.write(line)

print("2. Reading Ground Truth for validation S1 entities...")
val_gt_lines = []
val_s2_needed = set()
val_s3_needed = set()

with open("student_resource/dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
    gt_header = f.readline().strip()
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 1 and parts[0] in val_s1_ids:
            val_gt_lines.append(line)
            if len(parts) > 1 and parts[1]:
                for m in parts[1].split(","):
                    if m.startswith("S2-"):
                        val_s2_needed.add(m)
                    elif m.startswith("S3-"):
                        val_s3_needed.add(m)

print(f"Found GT for {len(val_gt_lines)} S1 entities.")
print(f"True positive S2 IDs needed: {len(val_s2_needed)}, S3 IDs needed: {len(val_s3_needed)}")

with open(os.path.join(VAL_DIR, "val_ground_truth.tsv"), "w", encoding="utf-8") as f:
    f.write(gt_header + "\n")
    for line in val_gt_lines:
        f.write(line)

# Now collect S2 and S3: all needed true positives + 50,000 distractor records from S2 and S3
print("3. Collecting S2 records (true positives + distractors)...")
val_s2_lines = []
distractor_s2_budget = 50000

with open("student_resource/dataset/train/train_source2.tsv", "r", encoding="utf-8") as f:
    s2_header = f.readline().strip()
    for i, line in enumerate(f):
        eid = line.split("\t", 1)[0]
        if eid in val_s2_needed:
            val_s2_lines.append(line)
        elif distractor_s2_budget > 0 and (i % 80 == 0):
            val_s2_lines.append(line)
            distractor_s2_budget -= 1

with open(os.path.join(VAL_DIR, "val_source2.tsv"), "w", encoding="utf-8") as f:
    f.write(s2_header + "\n")
    for line in val_s2_lines:
        f.write(line)
print(f"Saved {len(val_s2_lines)} S2 validation records.")

print("4. Collecting S3 records (true positives + distractors)...")
val_s3_lines = []
distractor_s3_budget = 50000

with open("student_resource/dataset/train/train_source3.tsv", "r", encoding="utf-8") as f:
    s3_header = f.readline().strip()
    for i, line in enumerate(f):
        eid = line.split("\t", 1)[0]
        if eid in val_s3_needed:
            val_s3_lines.append(line)
        elif distractor_s3_budget > 0 and (i % 80 == 0):
            val_s3_lines.append(line)
            distractor_s3_budget -= 1

with open(os.path.join(VAL_DIR, "val_source3.tsv"), "w", encoding="utf-8") as f:
    f.write(s3_header + "\n")
    for line in val_s3_lines:
        f.write(line)
print(f"Saved {len(val_s3_lines)} S3 validation records.")
print("Validation split successfully created in:", VAL_DIR)
