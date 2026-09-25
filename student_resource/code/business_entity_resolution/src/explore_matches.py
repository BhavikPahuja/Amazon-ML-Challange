import sys
sys.stdout.reconfigure(encoding='utf-8')

# Read first 5 matches from GT
gt_entries = []
s1_needed = set()
s2_needed = set()
s3_needed = set()

with open('student_resource/dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2 and parts[1]:
            s1_id = parts[0]
            ms = parts[1].split(',')
            gt_entries.append((s1_id, ms))
            s1_needed.add(s1_id)
            for m in ms:
                if m.startswith('S2-'): s2_needed.add(m)
                elif m.startswith('S3-'): s3_needed.add(m)
            if len(gt_entries) == 5:
                break

print(f"Targeting {len(gt_entries)} GT groups. S1 needed: {len(s1_needed)}, S2 needed: {len(s2_needed)}, S3 needed: {len(s3_needed)}")

s1_data = {}
with open('student_resource/dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        parts = line.strip().split('\t')
        if parts[0] in s1_needed:
            s1_data[parts[0]] = parts
            if len(s1_data) == len(s1_needed):
                break

s2_data = {}
with open('student_resource/dataset/train/train_source2.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        parts = line.strip().split('\t')
        if parts[0] in s2_needed:
            s2_data[parts[0]] = parts
            if len(s2_data) == len(s2_needed):
                break

s3_data = {}
with open('student_resource/dataset/train/train_source3.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        parts = line.strip().split('\t')
        if parts[0] in s3_needed:
            s3_data[parts[0]] = parts
            if len(s3_data) == len(s3_needed):
                break

for s1_id, ms in gt_entries:
    rec1 = s1_data.get(s1_id, [s1_id, '', '', ''])
    name1 = rec1[1] if len(rec1) > 1 else ''
    addr1 = rec1[2] if len(rec1) > 2 else ''
    c1 = rec1[3] if len(rec1) > 3 else ''
    print('='*75)
    print(f"S1: [{c1}] ID: {s1_id}")
    print(f"    Name: {name1}")
    print(f"    Addr: {addr1}")
    print("  Matches:")
    for m in ms:
        rec = s2_data.get(m) or s3_data.get(m)
        if rec:
            src = m[:2]
            mn = rec[1] if len(rec) > 1 else ''
            ma = rec[2] if len(rec) > 2 else ''
            print(f"    -> [{src}] ID: {m}")
            print(f"       Name: {mn}")
            print(f"       Addr: {ma}")
        else:
            print(f"    -> [{m[:2]}] ID: {m} (not found)")
