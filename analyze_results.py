import json

r1 = json.load(open('benchmark_results/real_benchmark_1777102041.json'))
r2 = json.load(open('benchmark_results/real_benchmark_1777098379.json'))

def extract_results(data, pipeline_name):
    results = {}
    for r in data['pipelines'][pipeline_name]['results']:
        results[r['problem_id']] = {
            'answer': r['answer'],
            'correct': r['correct'],
            'ground_truth': r['ground_truth'],
            'switched': r.get('metadata', {}).get('switched', None)
        }
    return results

b1 = extract_results(r1, 'Baseline')
s1 = extract_results(r1, 'Self-Reflection')
b2 = extract_results(r2, 'Baseline')
s2 = extract_results(r2, 'Self-Reflection')

print('=== RUN 1 (sq_001 - sq_028) ===')
print('Baseline correct:', sum(1 for v in b1.values() if v['correct']), '/', len(b1))
print('SR correct:', sum(1 for v in s1.values() if v['correct']), '/', len(s1))

print('\nREGRESSIONS (baseline right, SR wrong):')
for pid in sorted(b1.keys()):
    if b1[pid]['correct'] and not s1[pid]['correct']:
        gt = b1[pid]['ground_truth']
        ba = b1[pid]['answer']
        sa = s1[pid]['answer']
        sw = s1[pid]['switched']
        print(f'  {pid}: baseline={ba}(correct), SR={sa}(wrong), gt={gt}, switched={sw}')

print('\nRECOVERIES (baseline wrong, SR right):')
for pid in sorted(b1.keys()):
    if not b1[pid]['correct'] and s1[pid]['correct']:
        gt = b1[pid]['ground_truth']
        ba = b1[pid]['answer']
        sa = s1[pid]['answer']
        sw = s1[pid]['switched']
        print(f'  {pid}: baseline={ba}(wrong), SR={sa}(correct), gt={gt}, switched={sw}')

print('\n=== RUN 2 (sq_029 - sq_056) ===')
print('Baseline correct:', sum(1 for v in b2.values() if v['correct']), '/', len(b2))
print('SR correct:', sum(1 for v in s2.values() if v['correct']), '/', len(s2))

print('\nREGRESSIONS (baseline right, SR wrong):')
for pid in sorted(b2.keys()):
    if b2[pid]['correct'] and not s2[pid]['correct']:
        gt = b2[pid]['ground_truth']
        ba = b2[pid]['answer']
        sa = s2[pid]['answer']
        sw = s2[pid]['switched']
        print(f'  {pid}: baseline={ba}(correct), SR={sa}(wrong), gt={gt}, switched={sw}')

print('\nRECOVERIES (baseline wrong, SR right):')
for pid in sorted(b2.keys()):
    if not b2[pid]['correct'] and s2[pid]['correct']:
        gt = b2[pid]['ground_truth']
        ba = b2[pid]['answer']
        sa = s2[pid]['answer']
        sw = s2[pid]['switched']
        print(f'  {pid}: baseline={ba}(wrong), SR={sa}(correct), gt={gt}, switched={sw}')

# Count switches
print('\n=== SWITCH ANALYSIS ===')
for label, s in [('Run1', s1), ('Run2', s2)]:
    switched = sum(1 for v in s.values() if v['switched'] == True)
    not_switched = sum(1 for v in s.values() if v['switched'] == False)
    none_sw = sum(1 for v in s.values() if v['switched'] is None)
    print(f'{label}: switched={switched}, not_switched={not_switched}, none={none_sw}')
    
    # Of those that switched, how many were correct?
    switch_correct = sum(1 for v in s.values() if v['switched'] == True and v['correct'])
    switch_wrong = sum(1 for v in s.values() if v['switched'] == True and not v['correct'])
    print(f'  Switches: correct={switch_correct}, wrong={switch_wrong}')
