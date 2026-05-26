"""Aggregate v2 benchmark results (220 questions)"""
import json
from scipy.stats import chi2

METHODS = ['zeroshot', 'cot', 'sc_only', 'rag', 'kbsc']
LABELS = {
    'zeroshot': 'Zero-Shot',
    'cot': 'CoT (Wei 2022)',
    'sc_only': 'SC-only (Wang 2022)',
    'rag': 'RAG (Lewis 2020)',
    'kbsc': 'KB+SC+Step1/2 (Ours)'
}

def load_v2(name):
    try:
        with open(f'benchmark_results/v2_{name}_s0_e220.json') as f:
            return json.load(f)
    except:
        return None

def mcnemar(b, c):
    if b + c == 0:
        return 1.0
    chi2_stat = (abs(b - c) - 1)**2 / (b + c) if b + c > 0 else 0
    p = 1.0 - chi2.cdf(chi2_stat, 1)
    return max(0, min(1, p))

results = {}
for m in METHODS:
    d = load_v2(m)
    if d:
        results[m] = d
        correct = sum(1 for r in d if r['answer'] == r['correct'])
        print(f"{m}: {correct}/{len(d)} = {100*correct/len(d):.1f}%")
    else:
        print(f"{m}: NO DATA")

if len(results) < 5:
    missing = [m for m in METHODS if m not in results]
    print(f"\nMissing: {missing}")
    sys.exit(1)

n = len(results[METHODS[0]])
correct_answers = [results['zeroshot'][i]['correct'] for i in range(n)]

print(f"\n{'='*70}")
print(f"COMPREHENSIVE RESULTS: {n} questions")
print(f"{'='*70}")
print(f"\n{'Method':<25} {'Accuracy':>10} {'Correct':>8} {'vs ZS':>10}")
print("-" * 60)

zs_correct = sum(1 for i in range(n) if results['zeroshot'][i]['answer'] == correct_answers[i])
zs_acc = zs_correct / n

for method in METHODS:
    correct = sum(1 for i in range(n) if results[method][i]['answer'] == correct_answers[i])
    acc = correct / n
    rel = (acc - zs_acc) / zs_acc if zs_acc > 0 else 0
    label = LABELS.get(method, method)
    print(f"{label:<25} {acc:>10.1%} {correct:>7}/{n} {rel:>+9.1%}")

print(f"\n{'='*70}")
print("STATISTICAL SIGNIFICANCE (McNemar's test vs KB+SC)")
print(f"{'='*70}")
our = results['kbsc']
print(f"{'Method':<30} {'b':>5} {'c':>5} {'p-value':>10} {'Sig?':>8}")
print("-" * 60)
for method in METHODS:
    b = c = 0
    for i in range(n):
        gt = correct_answers[i]
        m_ok = results[method][i]['answer'] == gt
        o_ok = our[i]['answer'] == gt
        if not m_ok and o_ok: b += 1
        elif m_ok and not o_ok: c += 1
    p = mcnemar(b, c)
    sig = "YES **" if p < 0.01 else "yes *" if p < 0.05 else "no"
    label = LABELS.get(method, method)
    print(f"{label:<30} {b:>5} {c:>5} {p:>10.4f} {sig:>8}")

print(f"\n{'='*70}")
print("IMPROVEMENT vs ZERO-SHOT")
print(f"{'='*70}")
for method in METHODS:
    improved = regressed = 0
    for i in range(n):
        gt = correct_answers[i]
        m_ok = results[method][i]['answer'] == gt
        zs_ok = results['zeroshot'][i]['answer'] == gt
        if m_ok and not zs_ok: improved += 1
        elif not m_ok and zs_ok: regressed += 1
    net = improved - regressed
    label = LABELS.get(method, method)
    print(f"{label:<25} improved={improved:>3} regressed={regressed:>3} net={net:>+3}")

# Save summary
summary = {'n': n, 'methods': {}}
for m in METHODS:
    correct = sum(1 for i in range(n) if results[m][i]['answer'] == correct_answers[i])
    summary['methods'][m] = {'accuracy': correct/n, 'correct': correct, 'total': n}
with open('benchmark_results/v2_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)