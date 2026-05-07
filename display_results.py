import json
import os

results_dir = 'benchmark_results'
files = [f for f in os.listdir(results_dir) if f.startswith('real_llm_all_40_')]
latest_file = max([os.path.join(results_dir, f) for f in files], key=os.path.getctime)

with open(latest_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print('='*120)
print('COMPLETE BENCHMARK RESULTS')
print('='*120)
print('Total Problems:', data['total_problems'])
print()

for name, pd in data['pipelines'].items():
    print('='*120)
    print('PIPELINE:', name)
    print('Accuracy:', '{:.1%}'.format(pd['accuracy']))
    print('Total Results:', len(pd['results']))
    print()
    
    header = 'Problem ID'.ljust(20) + 'Correct'.ljust(10) + 'Tokens'.ljust(10) + 'Latency'.ljust(12) + 'Category'.ljust(25)
    print(header)
    print('-'*120)
    
    for r in pd['results']:
        row = str(r['problem_id']).ljust(20) + str(r['correct']).ljust(10) + str(r['tokens']).ljust(10) + '{:.2f}'.format(r['latency']).ljust(12) + str(r['category']).ljust(25)
        print(row)
    
    print()
    correct_count = sum(1 for r in pd['results'] if r.get('correct', False))
    total_tokens = sum(r.get('tokens', 0) for r in pd['results'])
    avg_latency = sum(r.get('latency', 0) for r in pd['results']) / len(pd['results'])
    
    print('Summary:')
    print('  Correct:', correct_count, '/40')
    print('  Total Tokens:', '{:,}'.format(total_tokens))
    print('  Avg Latency:', '{:.2f}'.format(avg_latency), 's')
    print()

print('='*120)
print('OVERALL COMPARISON')
print('='*120)
print()
print('Pipeline'.ljust(30) + 'Accuracy'.rjust(10) + 'Correct'.rjust(10) + 'Avg Tokens'.rjust(12) + 'Avg Latency'.rjust(12))
print('-'*120)

for name, pd in data['pipelines'].items():
    acc = '{:.1%}'.format(pd['accuracy'])
    correct = sum(1 for r in pd['results'] if r.get('correct', False))
    avg_tok = sum(r.get('tokens', 0) for r in pd['results']) / len(pd['results'])
    avg_lat = sum(r.get('latency', 0) for r in pd['results']) / len(pd['results'])
    print(name.ljust(30) + acc.rjust(10) + str(correct).rjust(10) + '{:.0f}'.format(avg_tok).rjust(12) + '{:.2f}s'.format(avg_lat).rjust(12))

print()
print('='*120)
