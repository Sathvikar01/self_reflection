"""Final comprehensive analysis across all available data."""
import json, os
from scipy.stats import chi2

def load(method, prefix='v7'):
    path = f'benchmark_results/{prefix}_{method}.json'
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def accuracy(data):
    if not data: return 0, 0
    correct = sum(1 for r in data if r.get('answer','') == r.get('correct',''))
    return correct, len(data)

def mcnemar(b, c):
    if b + c == 0: return 1.0
    chi2_stat = (abs(b - c) - 1)**2 / (b + c)
    return max(0, min(1, 1.0 - chi2.cdf(chi2_stat, 1)))

print("=" * 70)
print("COMPREHENSIVE BENCHMARK RESULTS")
print("=" * 70)

# 405B results (complete)
print("\n--- 405B (Llama-3.1-405B-Instruct) ---")
for method in ['zeroshot', 'cot', 'rag']:
    data = load(method, 'v4') or load(method, 'v3')
    if data:
        c, n = accuracy(data)
        print(f"  {method:12s}: {c}/{n} = {100*c/n:.1f}%")

# 70B results
print("\n--- 70B (Llama-3.3-70b-Instruct) ---")
for method in ['zeroshot', 'cot', 'rag', 'sc_only', 'kbsc']:
    data = load(method, 'v7')
    if data:
        c, n = accuracy(data)
        print(f"  {method:12s}: {c}/{n} = {100*c/n:.1f}%")

# 675B results
print("\n--- 675B (Mistral-Large-3-675B) ---")
for method in ['zeroshot', 'cot']:
    data = load(method, 'v6')
    if data:
        c, n = accuracy(data)
        print(f"  {method:12s}: {c}/{n} = {100*c/n:.1f}%")

# Best available comparison
print("\n" + "=" * 70)
print("BEST AVAILABLE COMPARISON (all methods)")
print("=" * 70)

methods = {
    'Zero-Shot': ('v4', 'zeroshot', 'v3'),
    'CoT': ('v4', 'cot', None),
    'RAG': ('v4', 'rag', None),
    'SC-only': ('v7', 'sc_only', None),
    'KB+SC': ('v7', 'kbsc', None),
}

results = {}
for name, (prefix, method, fallback) in methods.items():
    data = load(method, prefix)
    if not data and fallback:
        data = load(method, fallback)
    if data:
        c, n = accuracy(data)
        results[name] = {'correct': c, 'total': n, 'accuracy': c/n, 'data': data}

print(f"\n{'Method':<15} {'Acc':>8} {'Correct':>10} {'N':>5}")
print("-" * 40)
for name, r in results.items():
    print(f"{name:<15} {r['accuracy']:>7.1%} {r['correct']:>9}/{r['total']:<4} {r['total']:>5}")

# McNemar's test (if KB+SC available)
if 'KB+SC' in results and 'Zero-Shot' in results:
    print("\n" + "=" * 70)
    print("McNEMAR'S TEST (vs KB+SC)")
    print("=" * 70)
    kbsc = results['KB+SC']['data']
    zs = results['Zero-Shot']['data']
    
    # Find common questions
    kbsc_ids = {r['id']: r for r in kbsc}
    zs_ids = {r['id']: r for r in zs}
    common = set(kbsc_ids.keys()) & set(zs_ids.keys())
    
    b = c = 0
    for qid in common:
        k_ok = kbsc_ids[qid]['answer'] == kbsc_ids[qid]['correct']
        z_ok = zs_ids[qid]['answer'] == zs_ids[qid]['correct']
        if not z_ok and k_ok: b += 1
        elif z_ok and not k_ok: c += 1
    
    p = mcnemar(b, c)
    print(f"  Common questions: {len(common)}")
    print(f"  ZS wrong, KB+SC right: {b}")
    print(f"  ZS right, KB+SC wrong: {c}")
    print(f"  McNemar p-value: {p:.4f}")
    print(f"  Significant at 0.05: {'YES' if p < 0.05 else 'NO'}")

print("\nDone.")