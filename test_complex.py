"""Test SR on complex_reasoning with proper max_tokens and fixed answer extraction."""
import json, time, requests, re

MODEL = "meta/llama-3.1-405b-instruct"
API_KEY = "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"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

with open('data/datasets/complex_reasoning.json') as f:
    questions = json.load(f)

print(f"Loaded {len(questions)} questions", flush=True)

def call_model(messages, max_tokens=256):
    resp = requests.post(
        'https://integrate.api.nvidia.com/v1/chat/completions',
        headers=HEADERS,
        json={"model": MODEL, "messages": messages, "temperature": 0.3, "max_tokens": max_tokens},
        timeout=180
    )
    if resp.status_code == 200:
        return resp.json()['choices'][0]['message']['content']
    raise Exception(f"API error {resp.status_code}: {resp.text}")

def extract_answer(text):
    """Extract YES/NO from anywhere in text - check last 200 chars for answer."""
    t = text.lower().strip()
    if t.startswith('yes'):
        return 'yes'
    if t.startswith('no'):
        return 'no'
    last200 = t[-200:]
    if re.search(r'\byes\b', last200):
        return 'yes'
    if re.search(r'\bno\b', last200):
        return 'no'
    return 'unknown'

results = []
DELAY = 2.0

for i, q in enumerate(questions):
    print(f"\n[{i+1}/{len(questions)}] {q['category']}/{q['difficulty']}: {q['question'][:55]}...", flush=True)

    time.sleep(DELAY)
    t0 = time.time()
    baseline_resp = call_model([{"role": "user", "content": f"Question: {q['question']}\nAnswer with just YES or NO."}], max_tokens=128)
    baseline_ans = extract_answer(baseline_resp)
    print(f"  Baseline: {baseline_ans} (GT={q['answer']}) [{time.time()-t0:.0f}s]", flush=True)

    time.sleep(DELAY)
    t0 = time.time()
    kb_resp = call_model([{"role": "user", "content": f"List 2-4 key facts needed to answer: {q['question']}"}], max_tokens=128)
    time.sleep(DELAY)
    sr_resp = call_model([{"role": "user", "content": f"Facts: {kb_resp}\nQuestion: {q['question']}\nReason step by step, then answer YES or NO."}], max_tokens=1024)
    sr_ans = extract_answer(sr_resp)
    print(f"  SR:       {sr_ans} (GT={q['answer']}) [{time.time()-t0:.0f}s]", flush=True)

    results.append({
        "id": q['id'],
        "category": q['category'],
        "difficulty": q['difficulty'],
        "baseline": baseline_ans,
        "sr": sr_ans,
        "correct": q['answer']
    })

    with open('benchmark_results/complex_benchmark_partial.json', 'w') as f:
        json.dump(results, f, indent=2)

print("\n" + "="*60, flush=True)
b_ok = sum(1 for r in results if r['baseline'] == r['correct'])
s_ok = sum(1 for r in results if r['sr'] == r['correct'])
print(f"Baseline: {b_ok}/{len(results)} = {100*b_ok/len(results):.1f}%", flush=True)
print(f"SR:       {s_ok}/{len(results)} = {100*s_ok/len(results):.1f}%", flush=True)
impr = sum(1 for r in results if r['sr'] == r['correct'] and r['baseline'] != r['correct'])
regr = sum(1 for r in results if r['baseline'] == r['correct'] and r['sr'] != r['correct'])
print(f"Improved: {impr}, Regressed: {regr}", flush=True)
for r in results:
    tag = "OK" if r['sr'] == r['correct'] else "WRONG"
    print(f"  {tag} {r['id']} [{r['category']}]: B={r['baseline']} SR={r['sr']} GT={r['correct']}", flush=True)