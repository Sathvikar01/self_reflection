"""Phase 1: Merge and deduplicate all datasets into unified benchmark."""
import json
import hashlib

# Load all datasets
with open('data/datasets/strategyqa_full.json') as f:
    sq_full = json.load(f)
with open('data/datasets/expanded_problems.json') as f:
    expanded = json.load(f)
with open('data/datasets/complex_reasoning.json') as f:
    complex_r = json.load(f)

def make_key(q):
    return hashlib.md5(q['question'].lower().strip().encode()).hexdigest()

# Deduplicate across all sources
seen = {}
merged = []
sources = []

for q in sq_full:
    key = make_key(q)
    if key not in seen:
        item = {
            'id': f'merged_{len(merged)}',
            'question': q['question'],
            'answer': q['answer'].lower().strip() if isinstance(q['answer'], str) else q['answer'],
            'source': 'strategyqa_full',
            'type': 'yesno' if str(q.get('answer','')).lower() in ('yes','no') else 'other'
        }
        merged.append(item)
        seen[key] = len(merged) - 1

for q in expanded:
    key = make_key(q)
    if key not in seen:
        ans = str(q.get('answer','')).lower().strip()
        item = {
            'id': f'merged_{len(merged)}',
            'question': q['question'],
            'answer': ans,
            'source': 'expanded_problems',
            'type': 'yesno' if ans in ('yes','no') else ans
        }
        merged.append(item)
        seen[key] = len(merged) - 1

for q in complex_r:
    key = make_key(q)
    if key not in seen:
        item = {
            'id': f'merged_{len(merged)}',
            'question': q['question'],
            'answer': q['answer'].lower().strip(),
            'source': 'complex_reasoning',
            'type': 'yesno'
        }
        merged.append(item)
        seen[key] = len(merged) - 1

# Stats
yesno = [m for m in merged if m['type'] == 'yesno']
other = [m for m in merged if m['type'] != 'yesno']
print(f"Total merged: {len(merged)}")
print(f"  Yes/No: {len(yesno)}")
print(f"  Other: {len(other)}")
print(f"  (Other includes: {set(m['type'] for m in other)})")

# Save full merged
with open('data/datasets/benchmark_dataset.json', 'w') as f:
    json.dump(merged, f, indent=2)

# Save yes/no subset
with open('data/datasets/benchmark_yesno.json', 'w') as f:
    json.dump(yesno, f, indent=2)

print(f"\nSaved benchmark_dataset.json ({len(merged)} questions)")
print(f"Saved benchmark_yesno.json ({len(yesno)} yes/no questions)")

# Category breakdown for yes/no
cats = {}
for q in yesno:
    src = q['source']
    cats[src] = cats.get(src, 0) + 1
print(f"By source: {cats}")