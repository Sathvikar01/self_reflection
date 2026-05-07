"""
Comprehensive Benchmark Results Analyzer

Generates detailed results table with:
- Accuracy (baseline and SR)
- Tokens used (total, avg)
- Latency
- KB relevance
- McNemar's test
- Overfitting metrics
"""

import os
import sys
import json
from pathlib import Path

def analyze_benchmark_results(result_path: str):
    """Analyze a benchmark result file and generate a detailed table."""

    with open(result_path, 'r') as f:
        data = json.load(f)

    n_problems = data.get('n_problems', 0)
    pipelines = data.get('pipelines', {})

    baseline_data = pipelines.get('Baseline', {})
    sr_data = pipelines.get('Self-Reflection', {})

    baseline_results = baseline_data.get('results', [])
    sr_results = sr_data.get('results', [])

    print("=" * 140)
    print("COMPREHENSIVE BENCHMARK RESULTS")
    print("=" * 140)

    # Summary metrics
    print(f"\n{'METRIC':<40} {'BASELINE':<25} {'SELF-REFLECTION':<25}")
    print("-" * 90)
    print(f"{'Accuracy':<40} {baseline_data.get('accuracy', 0):.1%} {sr_data.get('accuracy', 0):.1%}")
    print(f"{'Correct/Total':<40} {baseline_data.get('correct', 0)}/{baseline_data.get('total', 0)} {sr_data.get('correct', 0)}/{sr_data.get('total', 0)}")
    print(f"{'Total Tokens':<40} {baseline_data.get('total_tokens', 0):,} {sr_data.get('total_tokens', 0):,}")
    print(f"{'Avg Tokens/Question':<40} {baseline_data.get('avg_tokens', 0):.0f} {sr_data.get('avg_tokens', 0):.0f}")
    print(f"{'Avg Latency (s)':<40} {baseline_data.get('avg_latency', 0):.2f} {sr_data.get('avg_latency', 0):.2f}")

    # Token overhead
    token_overhead = (sr_data.get('total_tokens', 0) / baseline_data.get('total_tokens', 1) - 1) * 100
    print(f"{'Token Overhead (%)':<40} {'--'} {token_overhead:+.1f}%")

    # McNemar's test
    n_01 = 0  # baseline wrong, SR right
    n_10 = 0  # baseline right, SR wrong
    kb_relevant_correct = 0
    kb_relevant_total = 0
    kb_unseen_correct = 0
    kb_unseen_total = 0

    for b, s in zip(baseline_results, sr_results):
        if not b.get('correct') and s.get('correct'):
            n_01 += 1
        elif b.get('correct') and not s.get('correct'):
            n_10 += 1

        kb_flag = s.get('metadata', {}).get('kb_relevant', False)
        if kb_flag:
            kb_relevant_total += 1
            if s.get('correct'):
                kb_relevant_correct += 1
        else:
            kb_unseen_total += 1
            if s.get('correct'):
                kb_unseen_correct += 1

    print(f"\n{'='*90}")
    print("OVERFITTING & DISCRIMINATIVE ANALYSIS")
    print(f"{'='*90}")
    print(f"{'Metric':<40} {'Value':<30}")
    print("-" * 70)
    print(f"{'Baseline wrong, SR right (improvement)':<40} {n_01}")
    print(f"{'Baseline right, SR wrong (regression)':<40} {n_10}")
    print(f"{'Net improvement':<40} {n_01 - n_10:+d}")
    print(f"{'KB-Relevant accuracy (SR)':<40} {kb_relevant_correct/kb_relevant_total:.1%} ({kb_relevant_correct}/{kb_relevant_total})" if kb_relevant_total > 0 else f"{'KB-Relevant accuracy (SR)':<40} N/A")
    print(f"{'KB-Unseen accuracy (SR)':<40} {kb_unseen_correct/kb_unseen_total:.1%} ({kb_unseen_correct}/{kb_unseen_total})" if kb_unseen_total > 0 else f"{'KB-Unseen accuracy (SR)':<40} N/A")

    if kb_relevant_total > 0 and kb_unseen_total > 0:
        kb_rel_acc = kb_relevant_correct / kb_relevant_total
        kb_unseen_acc = kb_unseen_correct / kb_unseen_total
        if kb_unseen_acc > 0:
            ratio = kb_rel_acc / kb_unseen_acc
            print(f"{'KB-Rel/KB-Unseen accuracy ratio':<40} {ratio:.2f}x")
            if ratio > 1.2:
                print(f"{'WARNING: Possible KB overfitting!':<40} SR performs much better on KB-relevant questions")

    # Per-question detailed table
    print(f"\n{'='*140}")
    print("PER-QUESTION DETAILED RESULTS")
    print(f"{'='*140}")
    print(f"{'ID':<10} {'Question':<45} {'GT':<4} {'BL':<4} {'SR':<4} {'Tokens':<8} {'Lat(ms)':<10} {'Method':<18} {'KB?':<4} {'Conf':<5}")
    print("-" * 140)

    for b, s in zip(baseline_results, sr_results):
        pid = b.get('problem_id', '')[:8]
        question = b.get('problem', '')[:43]
        gt = b.get('ground_truth', '')[:3]
        bl = b.get('answer', '')[:3]
        sr_ans = s.get('answer', '')[:3]
        tokens = s.get('total_tokens', 0)
        latency = s.get('latency_seconds', 0) * 1000  # ms
        method = s.get('metadata', {}).get('method', '')[:16]
        kb = 'Y' if s.get('metadata', {}).get('kb_relevant', False) else 'N'
        conf = s.get('confidence', 0)

        bl_correct = 'Y' if b.get('correct') else 'N'
        sr_correct = 'Y' if s.get('correct') else 'N'

        print(f"{pid:<10} {question:<45} {gt:<4} {bl_correct:<4} {sr_correct:<4} {tokens:<8.0f} {latency:<10.0f} {method:<18} {kb:<4} {conf:<5.2f}")

    print(f"\n{'='*90}")
    print("LEGEND")
    print(f"{'='*90}")
    print("GT = Ground Truth | BL = Baseline | SR = Self-Reflection")
    print("Y/N = Correct (Y) or Wrong (N)")
    print("Tokens = Total tokens used | Lat(ms) = Latency in milliseconds")
    print("Method = How SR reached answer (unanimous_vote, majority_vote, low_agreement)")
    print("KB? = Knowledge Base was relevant (Y/N)")
    print("Conf = Confidence level (0-1)")


if __name__ == "__main__":
    import sys
    result_path = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results/iter6_benchmark_1777432488.json"
    analyze_benchmark_results(result_path)