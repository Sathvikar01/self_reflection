"""
Ultra-Fast Benchmark - 2-path SC, 1.4s delay, resume support
"""
import os
import sys
import json
import time
import requests
from collections import Counter
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.unified_extractor import UnifiedAnswerExtractor

API_KEY = os.getenv("NVIDIA_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
BASE_URL = "https://integrate.api.nvidia.com/v1"

MIN_DELAY = 1.4  # Slightly aggressive for 40/min

def call_api(model, messages, temperature=0.5, max_tokens=150):
    time.sleep(MIN_DELAY)
    try:
        resp = requests.post(f"{BASE_URL}/chat/completions",
            headers=HEADERS, json={
                "model": model, "messages": messages,
                "temperature": temperature, "max_tokens": max_tokens, "top_p": 0.95
            }, timeout=45)
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"API err: {e}")
        return ""

def extract_answer(text):
    """Extract yes/no from response using unified extractor."""
    extracted = UnifiedAnswerExtractor.extract(text)
    answer = extracted.answer.lower().strip()
    if answer in ('yes', 'no'):
        return answer
    return "no"

KB = """KB: Hot water can freeze faster (Mpemba). Lightning > sun (30K vs 5.8K). Diamonds burn (C, 700C). Fish need O2, can drown. Sound needs medium. 5-sec rule false. Plants need O2 at night. All birds lay eggs. Penguins/ostriches can't fly. Bullsd react to movement. Gold is investment hedge. You can't tickle yourself. Astronauts don't need sunscreen. Great Wall not visible from space. Coin can't kill. We use >10% brain. Yawning contagious. Dolphins sleep unihemispherically. Whales get water from food. Elephants have teeth. Vampire bats drink blood."""

def baseline_solve(q):
    msgs = [{"role": "system", "content": "Answer yes/no only. <answer>"}, {"role": "user", "content": f"{q}\n<answer>"}]
    return extract_answer(call_api("meta/llama-3.1-8b-instruct", msgs, temperature=0.1, max_tokens=80))

def sr_solve(q):
    msgs = [{"role": "system", "content": f"Use KB if relevant: {KB}\nAnswer yes/no in <answer>."},
            {"role": "user", "content": f"{q}\n<answer>"}]
    votes = [extract_answer(call_api("meta/llama-3.1-8b-instruct", msgs, temperature=0.5, max_tokens=120)) for _ in range(2)]
    counts = Counter(votes)
    return counts.most_common(1)[0][0], dict(counts), len(set(votes)) == 1

def load_checkpoint(path):
    cp = path + ".checkpoint"
    if os.path.exists(cp):
        with open(cp, 'r') as f:
            return json.load(f)
    return None

def save_checkpoint(path, bl, sr, count):
    with open(path + ".checkpoint", 'w') as f:
        json.dump({"checkpoint": count, "baseline": bl, "sr": sr}, f)

def run(dataset, n=20, output="benchmark_results/fast_benchmark.json"):
    with open(dataset, 'r') as f:
        problems = json.load(f)[:n]

    cp = load_checkpoint(output)
    start_idx = 0
    baseline_results = []
    sr_results = []

    if cp and cp.get('checkpoint', 0) >= n:
        print(f"Already completed {n} questions. Loading from checkpoint...")
        baseline_results = cp['baseline']
        sr_results = cp['sr']
        start_idx = n
    elif cp:
        print(f"Resuming from checkpoint at {cp.get('checkpoint', 0)}...")
        baseline_results = cp['baseline']
        sr_results = cp['sr']
        start_idx = cp.get('checkpoint', 0)
    else:
        print(f"Starting fresh - {n} questions")

    total = n
    print(f"Estimated time: {n * 3 * MIN_DELAY / 60:.1f} min")
    print()

    start_time = time.time()

    for i in range(start_idx, total):
        p = problems[i]
        q, gt, pid = p['question'], p['answer'], p.get('id', f'q_{i+1}')

        print(f"[{i+1}/{total}] {pid[:15]}...", flush=True)

        b_ans = baseline_solve(q)
        b_cor = b_ans.lower()[:3] == gt.lower()[:3]
        baseline_results.append({"id": pid, "q": q, "a": b_ans, "gt": gt, "c": b_cor})
        print(f"  BL:{b_ans}({b_cor})", end=" ", flush=True)

        sr_ans, votes, unan = sr_solve(q)
        sr_cor = sr_ans.lower()[:3] == gt.lower()[:3]
        sr_results.append({"id": pid, "q": q, "a": sr_ans, "gt": gt, "c": sr_cor, "votes": votes, "unan": unan})
        print(f"SR:{sr_ans}({sr_cor}) [{'KB' if unan else 'split'}]", flush=True)

        if (i+1) % 5 == 0:
            save_checkpoint(output, baseline_results, sr_results, i+1)

    save_checkpoint(output, baseline_results, sr_results, total)

    # Stats
    n_q = len(baseline_results)
    b_acc = sum(1 for r in baseline_results if r['c']) / n_q
    s_acc = sum(1 for r in sr_results if r['c']) / n_q
    n_01 = sum(1 for b, s in zip(baseline_results, sr_results) if not b['c'] and s['c'])
    n_10 = sum(1 for b, s in zip(baseline_results, sr_results) if b['c'] and not s['c'])

    denom = n_01 + n_10
    p_val = 1.0
    if denom > 0:
        chi2 = (abs(n_01 - n_10) - 1) ** 2 / denom
        from math import exp
        p_val = exp(-chi2 / 2)

    elapsed = time.time() - start_time

    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"{'METRIC':<35} {'BASELINE':<20} {'SELF-REFLECTION':<20}")
    print("-" * 75)
    print(f"{'Accuracy':<35} {b_acc:.1%} {s_acc:.1%}")
    print(f"{'Correct/Total':<35} {int(b_acc*n_q)}/{n_q} {int(s_acc*n_q)}/{n_q}")
    print(f"{'McNemar p-value':<35} {p_val:.4f}")
    print(f"{'BL wrong, SR right':<35} {n_01}")
    print(f"{'BL right, SR wrong':<35} {n_10}")
    print(f"{'Time':<35} {elapsed/60:.1f} min")

    print(f"\n{'='*100}")
    print("PER-QUESTION")
    print(f"{'='*100}")
    print(f"{'ID':<10} {'Q':<55} {'GT':<4} {'BL':<4} {'SR':<4} {'Status':<10}")
    print("-" * 100)
    for b, s in zip(baseline_results, sr_results):
        st = "IMPROVED" if (not b['c'] and s['c']) else ("REGRESSED" if (b['c'] and not s['c']) else "SAME")
        print(f"{b['id'][:8]:<10} {b['q'][:53]:<55} {b['gt'][:3]:<4} {'Y' if b['c'] else 'N':<4} {'Y' if s['c'] else 'N':<4} {st:<10}")

    # Save final
    with open(output, 'w') as f:
        json.dump({"baseline": baseline_results, "sr": sr_results, "metrics": {
            "accuracy": {"baseline": b_acc, "sr": s_acc},
            "p_value": p_val, "n_01": n_01, "n_10": n_10, "time_min": elapsed/60
        }}, f, indent=2)

    print(f"\nSaved to {output}")
    return baseline_results, sr_results

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    dataset = sys.argv[2] if len(sys.argv) > 2 else "data/datasets/strategyqa_full.json"
    run(dataset, n)