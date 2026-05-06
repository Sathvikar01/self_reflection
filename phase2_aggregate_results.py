"""Aggregate results from all 5 methods and compute statistics."""
import json
import sys
from collections import defaultdict

METHODS = ['zeroshot', 'cot', 'sc_only', 'rag', 'kb_sc']
METHOD_LABELS = {
    'zeroshot': 'Zero-Shot',
    'cot': 'CoT (Wei 2022)',
    'sc_only': 'SC-only (Wang 2022)',
    'rag': 'RAG (Lewis 2020)',
    'kb_sc': 'KB+SC+Step1/2 (Ours)'
}

def load_results(method):
    # Try multiple ranges
    for end in [220, 50, 100]:
        path = f'benchmark_results/comp_{method}_s0_e{end}.json'
        try:
            with open(path) as f:
                return json.load(f), end
        except:
            continue
    return None, None

def mcnemar_table(results1, results2, correct_answers):
    """Build McNemar discordant table."""
    b = 0  # method1 wrong, method2 right
    c = 0  # method1 right, method2 wrong
    for r1, r2, gt in zip(results1, results2, correct_answers):
        m1_correct = (r1['answer'] == gt)
        m2_correct = (r2['answer'] == gt)
        if not m1_correct and m2_correct:
            b += 1
        elif m1_correct and not m2_correct:
            c += 1
    return b, c

def mcnemar_p(b, c):
    """Two-sided McNemar's test p-value using chi-squared approximation."""
    if b + c == 0:
        return 1.0
    chi2 = abs(b - c)**2 / (b + c)
    # chi-squared with 1 df, two-sided
    import math
    # Use normal approximation for 2x2 McNemar
    n = b + c
    if n == 0:
        return 1.0
    # continuity correction
    chi2_corr = (abs(b - c) - 1)**2 / n if n > 0 else 0
    # p-value from chi-squared distribution with 1 df
    try:
        from scipy.stats import chi2
        p = 1.0 - chi2.cdf(chi2_corr, 1)
        return p
    except:
        # Approximate using normal
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(b - c) / (2 * n)**0.5)))
        return max(0, min(1, p))

def main():
    end = 220
    all_results = {}

    for method in METHODS:
        data, end = load_results(method)
        if data:
            all_results[method] = data
            print(f"{method}: {len(data)} results loaded (range 0-{end})")
        else:
            print(f"{method}: NO DATA")

    if len(all_results) < 5:
        print("\nSome methods missing data.")
        missing = [m for m in METHODS if m not in all_results]
        print(f"Missing: {missing}")
        return

    n = len(all_results[METHODS[0]])
    for m in METHODS:
        assert len(all_results[m]) == n, f"Length mismatch: {m}"

    correct_answers = [all_results['zeroshot'][i]['correct'] for i in range(n)]

    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE RESULTS: {n} questions")
    print(f"{'='*70}")

    # Accuracy table
    print(f"\n{'Method':<25} {'Accuracy':>10} {'Correct':>8} {'Rel. to ZS':>10}")
    print("-" * 60)
    zs_correct = sum(1 for i in range(n) if all_results['zeroshot'][i]['answer'] == correct_answers[i])
    zs_acc = zs_correct / n

    for method in METHODS:
        correct = sum(1 for i in range(n) if all_results[method][i]['answer'] == correct_answers[i])
        acc = correct / n
        rel = (acc - zs_acc) / zs_acc if zs_acc > 0 else 0
        label = METHOD_LABELS.get(method, method)
        print(f"{label:<25} {acc:>10.1%} {correct:>7}/{n} {rel:>+9.1%}")

    # McNemar's test: KB+SC vs all others
    print(f"\n{'='*70}")
    print("STATISTICAL SIGNIFICANCE (McNemar's test, vs KB+SC)")
    print(f"{'='*70}")
    print(f"{'Comparison':<30} {'b':>5} {'c':>5} {'p-value':>10} {'Significant':>12}")
    print("-" * 70)

    our_results = all_results['kb_sc']
    for method in ['zeroshot', 'cot', 'sc_only', 'rag']:
        m_results = all_results[method]
        b, c = mcnemar_table(m_results, our_results, correct_answers)
        p = mcnemar_p(b, c)
        sig = "YES **" if p < 0.01 else "yes *" if p < 0.05 else "no"
        label = METHOD_LABELS.get(method, method)
        print(f"{label:<30} {b:>5} {c:>5} {p:>10.4f} {sig:>12}")

    # Improvement/regression counts
    print(f"\n{'='*70}")
    print("IMPROVEMENT vs ZERO-SHOT")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'Improved':>10} {'Regressed':>10} {'Net':>8}")
    print("-" * 60)
    for method in METHODS:
        improved = 0
        regressed = 0
        for i in range(n):
            gt = correct_answers[i]
            m_correct = all_results[method][i]['answer'] == gt
            zs_correct_flag = all_results['zeroshot'][i]['answer'] == gt
            if m_correct and not zs_correct_flag:
                improved += 1
            elif not m_correct and zs_correct_flag:
                regressed += 1
        net = improved - regressed
        print(f"{METHOD_LABELS.get(method,method):<25} {improved:>10} {regressed:>10} {net:>+8}")

    # Save summary
    summary = {
        'n_questions': n,
        'method_accuracies': {},
        'method_correct': {}
    }
    for method in METHODS:
        correct = sum(1 for i in range(n) if all_results[method][i]['answer'] == correct_answers[i])
        summary['method_accuracies'][method] = correct / n
        summary['method_correct'][method] = correct

    with open('benchmark_results/comprehensive_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary to benchmark_results/comprehensive_summary.json")

if __name__ == "__main__":
    main()