"""Quick 5-question benchmark with detailed results."""
import os
import sys
import json
import time

sys.path.insert(0, os.getcwd())
os.environ['NVIDIA_API_KEY'] = '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'

from anti_overfitting_pipeline import (
    NVIDIANIMClient, BaselinePipeline, AntiOverfittingSelfReflectionPipeline,
    mcnemar_test
)

API_KEY = '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'

# Load dataset
with open("data/datasets/strategyqa_full.json", 'r', encoding='utf-8') as f:
    problems = json.load(f)

test_problems = problems[:10]  # First 10

print("=" * 100)
print("QUICK BENCHMARK - 10 Questions with Anti-Overfitting Pipeline")
print("=" * 100)

client = NVIDIANIMClient(api_key=API_KEY)
baseline = BaselinePipeline(client)
sr = AntiOverfittingSelfReflectionPipeline(client)

baseline_results = []
sr_results = []

print("\nRunning baseline...")
for p in test_problems:
    r = baseline.solve(p['question'], p['id'], p['answer'])
    baseline_results.append(r)
    status = "CORRECT" if r.correct else "WRONG"
    print(f"  {r.problem_id}: {status} - {r.answer}")
    time.sleep(0.3)

print("\nRunning Self-Reflection...")
for p in test_problems:
    r = sr.solve(p['question'], p['id'], p['answer'])
    sr_results.append(r)
    kb = 'Y' if r.metadata.get('kb_relevant', False) else 'N'
    method = r.metadata.get('method', '')[:15]
    status = "CORRECT" if r.correct else "WRONG"
    print(f"  {r.problem_id}: {status} - {r.answer} | Votes:{r.metadata.get('vote_counts', {})} | KB:{kb} | {method}")
    time.sleep(0.3)

client.close()

# Calculate metrics
print("\n" + "=" * 100)
print("RESULTS TABLE")
print("=" * 100)

b_correct = sum(1 for r in baseline_results if r.correct)
s_correct = sum(1 for r in sr_results if r.correct)
b_acc = b_correct / len(baseline_results)
s_acc = s_correct / len(sr_results)
b_tokens = sum(r.total_tokens for r in baseline_results)
s_tokens = sum(r.total_tokens for r in sr_results)
b_latency = sum(r.latency_seconds for r in baseline_results)
s_latency = sum(r.latency_seconds for r in sr_results)

print(f"\n{'METRIC':<35} {'BASELINE':<20} {'SELF-REFLECTION':<20}")
print("-" * 75)
print(f"{'Accuracy':<35} {b_acc:.1%} {s_acc:.1%}")
print(f"{'Correct/Total':<35} {b_correct}/{len(baseline_results)} {s_correct}/{len(sr_results)}")
print(f"{'Total Tokens':<35} {b_tokens:,} {s_tokens:,}")
print(f"{'Avg Tokens/Question':<35} {b_tokens/len(baseline_results):.0f} {s_tokens/len(sr_results):.0f}")
print(f"{'Total Latency (s)':<35} {b_latency:.1f} {s_latency:.1f}")
print(f"{'Avg Latency (s)':<35} {b_latency/len(baseline_results):.2f} {s_latency/len(sr_results):.2f}")
print(f"{'Token Overhead':<35} {'--'} {(s_tokens/b_tokens - 1)*100:+.1f}%")

p_val, n_01, n_10 = mcnemar_test(baseline_results, sr_results)
print(f"\n{'McNemar p-value':<35} {p_val:.4f}")
print(f"{'Baseline wrong, SR right':<35} {n_01}")
print(f"{'Baseline right, SR wrong':<35} {n_10}")

# KB relevance analysis
kb_rel = sum(1 for r in sr_results if r.metadata.get('kb_relevant', False))
print(f"\n{'KB-Relevant questions':<35} {kb_rel}/{len(sr_results)}")

# Per-question table
print("\n" + "=" * 120)
print("PER-QUESTION DETAILS")
print("=" * 120)
print(f"{'ID':<10} {'Question':<45} {'GT':<4} {'BL':<4} {'SR':<4} {'Tokens':<8} {'Lat(ms)':<10} {'Method':<15} {'KB':<4}")
print("-" * 120)

for b, s in zip(baseline_results, sr_results):
    pid = b.problem_id[:8]
    q = b.problem[:43]
    gt = b.ground_truth[:3]
    bl_c = 'Y' if b.correct else 'N'
    sr_c = 'Y' if s.correct else 'N'
    tokens = s.total_tokens
    lat = s.latency_seconds * 1000
    method = s.metadata.get('method', '')[:13]
    kb = 'Y' if s.metadata.get('kb_relevant', False) else 'N'
    print(f"{pid:<10} {q:<45} {gt:<4} {bl_c:<4} {sr_c:<4} {tokens:<8.0f} {lat:<10.0f} {method:<15} {kb:<4}")

print("\n" + "=" * 100)
print("DONE")
print("=" * 100)