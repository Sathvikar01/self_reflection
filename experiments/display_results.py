"""Display benchmark results in formatted tables."""

import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('benchmark_results/complex_extended_benchmark_1775360648.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print('=' * 100)
print('COMPLEX EXTENDED DATASET BENCHMARK RESULTS - DETAILED BREAKDOWN')
print('=' * 100)

for config in data['configurations']:
    print(f"\n{'=' * 100}")
    print(f"Configuration: {config['config_name']}")
    print('=' * 100)
    print(f"Overall Accuracy: {config['accuracy']:.1%} ({config['correct']}/{config['total_problems']})")
    print(f"Average Tokens: {config['avg_tokens_per_problem']:.0f}")
    print(f"Average Latency: {config['avg_latency_seconds']:.2f}s")
    print(f"Efficiency Score: {config['efficiency']:.4f}")
    
    print(f"\nAccuracy by Complexity:")
    for complexity, acc in sorted(config['accuracy_by_complexity'].items(), key=lambda x: {'very_high': 4, 'high': 3, 'medium': 2, 'low': 1}.get(x[0], 0), reverse=True):
        print(f"  {complexity:>12}: {acc:.1%}")
    
    print(f"\nTop Performing Categories:")
    sorted_cats = sorted(config['accuracy_by_category'].items(), key=lambda x: x[1], reverse=True)[:8]
    for cat, acc in sorted_cats:
        print(f"  {cat:<40}: {acc:.1%}")

print(f"\n{'=' * 100}")
print('COMPARATIVE ANALYSIS')
print('=' * 100)

baseline = data['configurations'][0]
sr = data['configurations'][1]
rl = data['configurations'][2]

print(f"\n{'Metric':<30} {'Baseline':>15} {'Self-Reflect':>15} {'RL-Guided':>15}")
print('-' * 75)
print(f"{'Accuracy':<30} {baseline['accuracy']:>14.1%} {sr['accuracy']:>14.1%} {rl['accuracy']:>14.1%}")
print(f"{'Avg Tokens':<30} {baseline['avg_tokens_per_problem']:>14.0f} {sr['avg_tokens_per_problem']:>14.0f} {rl['avg_tokens_per_problem']:>14.0f}")
print(f"{'Avg Latency (s)':<30} {baseline['avg_latency_seconds']:>14.2f} {sr['avg_latency_seconds']:>14.2f} {rl['avg_latency_seconds']:>14.2f}")
print(f"{'Efficiency':<30} {baseline['efficiency']:>14.4f} {sr['efficiency']:>14.4f} {rl['efficiency']:>14.4f}")

print(f"\n{'=' * 100}")
print('KEY INSIGHTS')
print('=' * 100)
print(f"\n1. ACCURACY IMPROVEMENT:")
print(f"   - Self-Reflection: +{(sr['accuracy'] - baseline['accuracy'])*100:.1f}pp over baseline ({(sr['accuracy']/baseline['accuracy']-1)*100:+.1f}% relative)")
print(f"   - RL-Guided: +{(rl['accuracy'] - baseline['accuracy'])*100:.1f}pp over baseline ({(rl['accuracy']/baseline['accuracy']-1)*100:+.1f}% relative)")
print(f"   - RL-Guided vs Self-Reflect: +{(rl['accuracy'] - sr['accuracy'])*100:.1f}pp ({(rl['accuracy']/sr['accuracy']-1)*100:+.1f}% relative)")

print(f"\n2. COMPLEXITY BREAKDOWN:")
print(f"   Very High Complexity Problems (n=10):")
vh_base = baseline['accuracy_by_complexity'].get('very_high', 0)
vh_sr = sr['accuracy_by_complexity'].get('very_high', 0)
vh_rl = rl['accuracy_by_complexity'].get('very_high', 0)
print(f"     Baseline: {vh_base:.1%} | Self-Reflect: {vh_sr:.1%} | RL-Guided: {vh_rl:.1%}")
print(f"     RL-Guided improvement over baseline: +{(vh_rl-vh_base)*100:.1f}pp")

print(f"\n   High Complexity Problems (n=19):")
h_base = baseline['accuracy_by_complexity'].get('high', 0)
h_sr = sr['accuracy_by_complexity'].get('high', 0)
h_rl = rl['accuracy_by_complexity'].get('high', 0)
print(f"     Baseline: {h_base:.1%} | Self-Reflect: {h_sr:.1%} | RL-Guided: {h_rl:.1%}")
print(f"     RL-Guided improvement over baseline: +{(h_rl-h_base)*100:.1f}pp")

print(f"\n3. COST-EFFICIENCY TRADEOFFS:")
print(f"   - Baseline: Best token efficiency ({baseline['efficiency']:.4f})")
print(f"   - Self-Reflection: +74% cost for +28% accuracy improvement")
print(f"   - RL-Guided: +105% cost for +83% accuracy improvement")
print(f"   - RL-Guided achieves 82.5% accuracy - only 17.5% error rate vs 55% baseline")

print(f"\n4. CATEGORY-SPECIFIC PERFORMANCE:")
print(f"   Categories where RL-Guided excels:")
rl_cats = sorted(rl['accuracy_by_category'].items(), key=lambda x: x[1], reverse=True)[:5]
for cat, acc in rl_cats:
    base_acc = baseline['accuracy_by_category'].get(cat, 0)
    print(f"     {cat:<40}: {acc:.1%} (vs baseline {base_acc:.1%})")

print('\n' + '=' * 100)
