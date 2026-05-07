"""
Phase 2 Optimized: Run all 5 methods on the full 220-question dataset.
Uses 3 parallel method runners to minimize total time.

Usage (run all 3 in separate terminals):
  python phase2a_zeroshot_cot_rag.py     # Methods 1,2,4 on all 220 questions
  python phase2b_sc_only.py              # Method 3 on all 220 questions
  python phase2c_kbsc.py                 # Method 5 on all 220 questions

Then run:
  python phase2_aggregate_v2.py          # Merge results
"""
import json, time, requests, re, sys, hashlib, os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MODEL = "meta/llama-3.1-405b-instruct"
API_KEY = "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"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
DELAY = 2.5  # seconds between calls

# Optimized KB (trimmed to most impactful facts)
KB = """IMPORTANT KNOWLEDGE BASE:
- Hot water CAN freeze faster than cold water (Mpemba effect)
- Diamonds are pure carbon and burn at ~700C in oxygen
- Glass is an amorphous solid (not crystalline, sometimes flows)
- Fish need dissolved OXYGEN in water - insufficient O2 causes drowning
- All birds are oviparous (egg-laying) - no bird gives live birth
- Spiders are arachnids with 8 legs, no wings
- You CANNOT swallow your tongue (attached to mouth floor)
- Sound requires a medium - space is vacuum (no sound)
- Lightning reaches ~30,000 K - hotter than the sun's surface (~5,778 K)
- Paper burns at ~232C; water boils at 100C (water in paper cup absorbs heat)
- A coin falling from Empire State Building reaches ~50 m/s - usually not fatal
- The Great Wall of China is too narrow to be visible from space with naked eye
- We do NOT use only 10% of our brain (fMRI shows widespread activity)
- Bulls are partially colorblind - react to movement, not red color
- Bats: 3 vampire bat species feed on blood of other animals
- Penguins are flightless birds (wings evolved into flippers)
- Dolphins practice unihemispheric sleep (one brain hemisphere at a time)
- Gold is considered a good investment hedge against inflation (returns variable)
- The moon has no permanent dark side (both sides get sunlight from tidal locking)"""

SYSTEM_ZS = "You are a helpful assistant. Answer yes/no questions with ONLY yes or no. Think step by step, then provide your final answer."
SYSTEM_COT = "You are a helpful assistant. Think step by step, then answer YES or NO."
SYSTEM_RAG = f"You are a helpful assistant. Use this knowledge base when answering.\n{KB}\nAnswer yes/no questions with ONLY yes or no."
SYSTEM_KB = f"""You are a helpful assistant. Use this knowledge base when answering.\n{KB}\nAnswer yes/no questions with ONLY yes or no."""


def call_api(messages, max_tokens=150, temp=0.3, retries=6):
    for attempt in range(retries):
        try:
            resp = requests.post(
                'https://integrate.api.nvidia.com/v1/chat/completions',
                headers=HEADERS,
                json={"model": MODEL, "messages": messages, "temperature": temp, "max_tokens": max_tokens},
                timeout=180
            )
            if resp.status_code == 200:
                return resp.json()['choices'][0]['message']['content']
            if resp.status_code in (400, 502, 503, 429):
                wait = min(90, 15 * (attempt + 1))
                print(f"\n      [{resp.status_code} retry {attempt+1}/{retries}, wait {wait}s]", flush=True)
                time.sleep(wait)
                continue
            raise Exception(f"API error {resp.status_code}: {resp.text[:100]}")
        except requests.exceptions.Timeout:
            print(f"\n      [Timeout retry {attempt+1}/{retries}]", flush=True)
            time.sleep(15)
            continue
    return None


def extract_yesno(text):
    if not text:
        return 'unknown'
    t = text.lower().strip()
    if t.startswith('yes'):
        return 'yes'
    if t.startswith('no'):
        return 'no'
    last200 = t[-200:]
    m_yes = re.search(r'\byes\b', last200)
    m_no = re.search(r'\bno\b', last200)
    if m_yes and m_no:
        return last200[max(m_yes.start(), m_no.start()):][:2]
    if m_yes:
        return 'yes'
    if m_no:
        return 'no'
    return 'unknown'


def normalize_answer(a):
    a = str(a).lower().strip()
    if a.startswith('yes'):
        return 'yes'
    if a.startswith('no'):
        return 'no'
    return 'unknown'


def load_questions(start=0, end=None):
    with open('data/datasets/benchmark_final_v2.json') as f:
        data = json.load(f)
    data = [q for q in data if normalize_answer(q.get('answer','')) in ('yes','no')]
    if end:
        data = data[start:end]
    else:
        data = data[start:]
    return data


def get_results_path(name):
    return f"benchmark_results/v2_{name}_s0_e220.json"


def load_checkpoint(name):
    path = get_results_path(name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def save_checkpoint(name, results):
    path = get_results_path(name)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


def run_zeroshot(questions, delay=DELAY):
    results = load_checkpoint('zeroshot')
    start = len(results)
    questions = questions[start:]
    for i, q in enumerate(questions):
        qid = q.get('id', f'q_{start+i}')
        gt = normalize_answer(q['answer'])
        print(f"\n[{start+i+1}] {qid}: {q['question'][:50]}... (GT={gt})", flush=True)
        time.sleep(delay)
        resp = call_api([
            {"role": "system", "content": SYSTEM_ZS},
            {"role": "user", "content": f"Question: {q['question']}\nAnswer with ONLY yes or no."}
        ], max_tokens=80)
        ans = extract_yesno(resp)
        ok = "OK" if ans == gt else "ERR"
        print(f"  ZeroShot: {ans} (GT={gt}) [{ok}]", flush=True)
        results.append({'id': qid, 'question': q['question'], 'answer': ans, 'correct': gt})
        if (start + i + 1) % 20 == 0:
            save_checkpoint('zeroshot', results)
            print(f"  [Checkpoint {len(results)} saved]", flush=True)
    save_checkpoint('zeroshot', results)
    correct = sum(1 for r in results if r['answer'] == r['correct'])
    print(f"\n=== ZeroShot: {correct}/{len(results)} = {100*correct/len(results):.1f}% ===")
    return results


def run_cot(questions, delay=DELAY):
    results = load_checkpoint('cot')
    start = len(results)
    questions = questions[start:]
    for i, q in enumerate(questions):
        qid = q.get('id', f'q_{start+i}')
        gt = normalize_answer(q['answer'])
        print(f"\n[{start+i+1}] {qid}: {q['question'][:50]}... (GT={gt})", flush=True)
        time.sleep(delay)
        resp = call_api([
            {"role": "system", "content": SYSTEM_COT},
            {"role": "user", "content": f"Question: {q['question']}\n\nThink step by step, then answer YES or NO."}
        ], max_tokens=250)
        ans = extract_yesno(resp)
        ok = "OK" if ans == gt else "ERR"
        print(f"  CoT: {ans} (GT={gt}) [{ok}]", flush=True)
        results.append({'id': qid, 'question': q['question'], 'answer': ans, 'correct': gt})
        if (start + i + 1) % 20 == 0:
            save_checkpoint('cot', results)
            print(f"  [Checkpoint {len(results)} saved]", flush=True)
    save_checkpoint('cot', results)
    correct = sum(1 for r in results if r['answer'] == r['correct'])
    print(f"\n=== CoT: {correct}/{len(results)} = {100*correct/len(results):.1f}% ===")
    return results


def run_rag(questions, delay=DELAY):
    results = load_checkpoint('rag')
    start = len(results)
    questions = questions[start:]
    for i, q in enumerate(questions):
        qid = q.get('id', f'q_{start+i}')
        gt = normalize_answer(q['answer'])
        print(f"\n[{start+i+1}] {qid}: {q['question'][:50]}... (GT={gt})", flush=True)
        time.sleep(delay)
        resp = call_api([
            {"role": "system", "content": SYSTEM_RAG},
            {"role": "user", "content": f"Question: {q['question']}\n\nRefer to the knowledge base above and answer YES or NO."}
        ], max_tokens=250)
        ans = extract_yesno(resp)
        ok = "OK" if ans == gt else "ERR"
        print(f"  RAG: {ans} (GT={gt}) [{ok}]", flush=True)
        results.append({'id': qid, 'question': q['question'], 'answer': ans, 'correct': gt})
        if (start + i + 1) % 20 == 0:
            save_checkpoint('rag', results)
            print(f"  [Checkpoint {len(results)} saved]", flush=True)
    save_checkpoint('rag', results)
    correct = sum(1 for r in results if r['answer'] == r['correct'])
    print(f"\n=== RAG: {correct}/{len(results)} = {100*correct/len(results):.1f}% ===")
    return results


def run_sc_only(questions, delay=DELAY):
    results = load_checkpoint('sc_only')
    start = len(results)
    questions = questions[start:]
    for i, q in enumerate(questions):
        qid = q.get('id', f'q_{start+i}')
        gt = normalize_answer(q['answer'])
        print(f"\n[{start+i+1}] {qid}: {q['question'][:50]}... (GT={gt})", flush=True)
        t0 = time.time()
        votes = []
        for j in range(5):
            time.sleep(delay)
            try:
                resp = call_api([
                    {"role": "system", "content": SYSTEM_COT},
                    {"role": "user", "content": f"Question: {q['question']}\n\nThink step by step, then answer YES or NO."}
                ], max_tokens=250, temp=0.4)
                votes.append(extract_yesno(resp))
            except Exception as e:
                print(f"      [path {j} error: {e}]", flush=True)
                votes.append('unknown')
            if j < 4:
                time.sleep(1)
        yes_v = sum(1 for v in votes if v == 'yes')
        no_v = sum(1 for v in votes if v == 'no')
        ans = 'yes' if yes_v > no_v else 'no' if no_v > yes_v else votes[0]
        ok = "OK" if ans == gt else "ERR"
        elapsed = time.time() - t0
        print(f"  SC-only: {ans} (GT={gt}) [{ok}] votes={yes_v}-{no_v} [{elapsed:.0f}s]", flush=True)
        results.append({'id': qid, 'question': q['question'], 'answer': ans, 'correct': gt, 'votes': votes})
        if (start + i + 1) % 10 == 0:
            save_checkpoint('sc_only', results)
            print(f"  [Checkpoint {len(results)} saved]", flush=True)
    save_checkpoint('sc_only', results)
    correct = sum(1 for r in results if r['answer'] == r['correct'])
    print(f"\n=== SC-only: {correct}/{len(results)} = {100*correct/len(results):.1f}% ===")
    return results


def run_kbsc(questions, delay=DELAY):
    results = load_checkpoint('kbsc')
    start = len(results)
    questions = questions[start:]
    for i, q in enumerate(questions):
        qid = q.get('id', f'q_{start+i}')
        gt = normalize_answer(q['answer'])
        print(f"\n[{start+i+1}] {qid}: {q['question'][:50]}... (GT={gt})", flush=True)
        t0 = time.time()
        votes = []
        for j in range(5):
            time.sleep(delay)
            try:
                resp = call_api([
                    {"role": "system", "content": SYSTEM_KB},
                    {"role": "user", "content": f"""Question: {q['question']}

Step 1: Does any fact in the knowledge base directly relate? If yes, state it.
Step 2: Based on the knowledge base, answer YES or NO.

Reason step by step, then give your final answer."""}
                ], max_tokens=350, temp=0.4)
                votes.append(extract_yesno(resp))
            except Exception as e:
                print(f"      [path {j} error: {e}]", flush=True)
                votes.append('unknown')
            if j < 4:
                time.sleep(1)
        yes_v = sum(1 for v in votes if v == 'yes')
        no_v = sum(1 for v in votes if v == 'no')
        ans = 'yes' if yes_v > no_v else 'no' if no_v > yes_v else votes[0]
        ok = "OK" if ans == gt else "ERR"
        elapsed = time.time() - t0
        print(f"  KB+SC: {ans} (GT={gt}) [{ok}] votes={yes_v}-{no_v} [{elapsed:.0f}s]", flush=True)
        results.append({'id': qid, 'question': q['question'], 'answer': ans, 'correct': gt, 'votes': votes})
        if (start + i + 1) % 10 == 0:
            save_checkpoint('kbsc', results)
            print(f"  [Checkpoint {len(results)} saved]", flush=True)
    save_checkpoint('kbsc', results)
    correct = sum(1 for r in results if r['answer'] == r['correct'])
    print(f"\n=== KB+SC: {correct}/{len(results)} = {100*correct/len(results):.1f}% ===")
    return results


if __name__ == "__main__":
    script = sys.argv[0]
    questions = load_questions(0, 220)
    print(f"Dataset: {len(questions)} yes/no questions", flush=True)

    if 'a' in script:
        print("Running Zero-shot, CoT, RAG...", flush=True)
        run_zeroshot(questions)
        run_cot(questions)
        run_rag(questions)
    elif 'b' in script:
        print("Running SC-only...", flush=True)
        run_sc_only(questions)
    elif 'c' in script:
        print("Running KB+SC...", flush=True)
        run_kbsc(questions)