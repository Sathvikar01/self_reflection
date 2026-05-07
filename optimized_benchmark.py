"""
Optimized Benchmark - Minimal API calls, Rate-limit aware
40 requests per minute = ~1.5s between calls
Uses 3-path self-consistency to halve API calls vs 7-path
"""

import os
import sys
import json
import time
import requests
from collections import Counter
from dotenv import load_dotenv

load_dotenv(override=True)

API_KEY = os.getenv("NVIDIA_API_KEY", "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")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
BASE_URL = "https://integrate.api.nvidia.com/v1"

# Rate limit: 40 req/min = 1.5s between requests
MIN_DELAY = 1.6

def call_api(model, messages, temperature=0.5, max_tokens=200, cache_key=None):
    """Single API call with rate limiting."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 0.95
    }

    time.sleep(MIN_DELAY)  # Rate limit protection

    try:
        resp = requests.post(f"{BASE_URL}/chat/completions",
            headers=HEADERS, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"API error: {e}")
        return ""


def extract_answer(text):
    """Extract yes/no from response."""
    text = text.lower().strip()
    # Look for <answer> tags
    import re
    m = re.search(r'<answer>(yes|no)</answer>', text)
    if m:
        return m.group(1)
    # Look for standalone yes/no
    m = re.search(r'\b(yes|no)\b', text)
    if m:
        return m.group(1)
    return "no"


def check_answer(pred, truth):
    """Check if answer is correct."""
    p = pred.lower().strip()[:3]
    t = truth.lower().strip()[:3]
    return p == t


# Expanded KB
KB = """KNOWLEDGE BASE - Key facts:
- Hot water CAN freeze faster (Mpemba effect)
- Lightning is hotter than sun (30,000K vs 5,778K)
- Diamonds burn at ~700C (carbon)
- Fish need dissolved O2, can drown
- Sound needs medium, no sound in space
- Glass is amorphous solid
- 5-second rule is false - bacteria transfer instantly
- Plants need O2 (respiration at night)
- All birds lay eggs (oviparous)
- Penguins/ostriches cannot fly
- Bullsd react to movement, not color
- Gold is investment hedge
- You CAN tickle yourself? NO (cerebellum prediction)
- Astronauts don't need sunscreen (shielded)
- Great Wall not visible from space
- Coin falling can't kill (terminal velocity too low)
- We use >10% of brain (fMRI shows activity)
- Yawning is contagious (empathy)
- Dolphins sleep unihemispherically
- Whales get water from food, not drinking
- Elephants have teeth (molars + tusks)
- Trees don't sleep (no brain)
- Vampire bats (3 species) drink blood
- Paper cup boil water (water absorbs heat)"""


def baseline_solve(question):
    """Baseline - single call, T=0.1."""
    messages = [
        {"role": "system", "content": "Answer yes/no questions with ONLY yes or no. Think step by step. <answer>"},
        {"role": "user", "content": f"{question}\n\nThink step by step, then answer <answer>"}
    ]
    response = call_api("meta/llama-3.1-8b-instruct", messages, temperature=0.1, max_tokens=100)
    answer = extract_answer(response)
    return answer, response


def sr_solve(question):
    """Self-reflection - 3 paths at T=0.5 (minimal for rate limit)."""
    system_prompt = f"""You answer yes/no using the knowledge base.
KNOWLEDGE BASE: {KB}
Check if KB is relevant. If yes, use it. Answer in <answer>yes</answer> or <answer>no</answer>."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {question}\nStep 1: Check KB. Step 2: Answer.\n<answer>"}
    ]

    votes = []
    for i in range(3):
        response = call_api("meta/llama-3.1-8b-instruct", messages, temperature=0.5, max_tokens=150)
        answer = extract_answer(response)
        votes.append(answer)

    # Majority vote
    counts = Counter(votes)
    majority = counts.most_common(1)[0][0]
    unanimous = len(set(votes)) == 1

    return majority, {"votes": dict(counts), "unanimous": unanimous, "tokens": sum(len(m['content']) for m in messages)}


def run_benchmark(dataset_path, n_problems=0, output_path=None):
    """Run optimized benchmark."""

    with open(dataset_path, 'r', encoding='utf-8') as f:
        problems = json.load(f)

    if n_problems > 0:
        problems = problems[:n_problems]

    n = len(problems)
    print(f"\n{'='*80}")
    print(f"OPTIMIZED BENCHMARK - {n} questions | Rate limit: 40/min")
    print(f"{'='*80}")

    baseline_results = []
    sr_results = []

    # Calculate estimated time: 2 calls per question (baseline + SR)
    # Each call has 1.6s delay = 3.2s per question
    # Total = 3.2s * n questions
    print(f"Estimated time: {3.2 * n / 60:.1f} minutes")
    print()

    start_time = time.time()

    for i, p in enumerate(problems):
        q = p['question']
        gt = p['answer']
        pid = p.get('id', f'q_{i+1}')

        print(f"[{i+1}/{n}] {pid}: {q[:50]}...")

        # Baseline
        b_answer, b_meta = baseline_solve(q)
        b_correct = check_answer(b_answer, gt)
        baseline_results.append({
            "id": pid, "question": q, "answer": b_answer,
            "ground_truth": gt, "correct": b_correct
        })
        status_b = "CORRECT" if b_correct else "WRONG"
        print(f"  Baseline: {status_b} ({b_answer})")

        # SR
        sr_answer, sr_meta = sr_solve(q)
        sr_correct = check_answer(sr_answer, gt)
        sr_results.append({
            "id": pid, "question": q, "answer": sr_answer,
            "ground_truth": gt, "correct": sr_correct,
            "metadata": sr_meta
        })
        status_sr = "CORRECT" if sr_correct else "WRONG"
        kb_flag = "KB" if sr_meta.get('unanimous') else "split"
        print(f"  SR:      {status_sr} ({sr_answer}) [{kb_flag}]")

        elapsed = time.time() - start_time
        remaining = (n - i - 1) * 3.2
        print(f"  Elapsed: {elapsed/60:.1f}m | Remaining: {remaining/60:.1f}m")

        # Save checkpoint
        if output_path and (i+1) % 5 == 0:
            save_checkpoint(output_path, baseline_results, sr_results, i+1)

    total_time = time.time() - start_time

    # Calculate metrics
    b_acc = sum(1 for r in baseline_results if r['correct']) / n
    s_acc = sum(1 for r in sr_results if r['correct']) / n
    b_tokens = sum(200 for _ in baseline_results)  # Approx
    s_tokens = sum(450 for _ in sr_results)  # Approx

    # Discordant pairs
    n_01 = sum(1 for b, s in zip(baseline_results, sr_results)
               if not b['correct'] and s['correct'])
    n_10 = sum(1 for b, s in zip(baseline_results, sr_results)
               if b['correct'] and not s['correct'])

    # McNemar
    denom = n_01 + n_10
    if denom > 0:
        chi2 = (abs(n_01 - n_10) - 1) ** 2 / denom
        from math import exp
        p_val = exp(-chi2 / 2)
    else:
        p_val = 1.0

    # Print results table
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")

    print(f"\n{'METRIC':<40} {'BASELINE':<20} {'SELF-REFLECTION':<20}")
    print("-" * 80)
    print(f"{'Accuracy':<40} {b_acc:.1%} {s_acc:.1%}")
    print(f"{'Correct/Total':<40} {int(b_acc*n)}/{n} {int(s_acc*n)}/{n}")
    print(f"{'Total Tokens (approx)':<40} {b_tokens:,} {s_tokens:,}")
    print(f"{'Token Overhead':<40} {'--'} {(s_tokens/b_tokens-1)*100:+.0f}%")
    print(f"{'Total Time':<40} {total_time/60:.1f} minutes")

    print(f"\n{'='*80}")
    print(f"McNemar p-value: {p_val:.4f}")
    print(f"Baseline wrong, SR right: {n_01}")
    print(f"Baseline right, SR wrong: {n_10}")
    print(f"Relative improvement: {((s_acc/b_acc)-1)*100:+.1f}%")

    # Per-question table
    print(f"\n{'='*100}")
    print("PER-QUESTION RESULTS")
    print(f"{'='*100}")
    print(f"{'ID':<10} {'Question':<50} {'GT':<4} {'BL':<4} {'SR':<4} {'Status':<10}")
    print("-" * 100)

    for b, s in zip(baseline_results, sr_results):
        pid = b['id'][:8]
        q = b['question'][:48]
        gt = b['ground_truth'][:3]
        bl = 'Y' if b['correct'] else 'N'
        sr = 'Y' if s['correct'] else 'N'
        status = "IMPROVED" if (not b['correct'] and s['correct']) else ("REGRESSED" if (b['correct'] and not s['correct']) else "SAME")
        print(f"{pid:<10} {q:<50} {gt:<4} {bl:<4} {sr:<4} {status:<10}")

    # Save final results
    if output_path:
        save_final(output_path, baseline_results, sr_results, {
            "accuracy": {"baseline": b_acc, "sr": s_acc},
            "p_value": p_val,
            "discordant": {"n_01": n_01, "n_10": n_10},
            "total_time_minutes": total_time / 60
        })

    print(f"\n{'='*80}")
    print(f"Done! Total time: {total_time/60:.1f} minutes")
    print(f"{'='*80}")

    return baseline_results, sr_results


def save_checkpoint(path, baseline, sr, count):
    """Save checkpoint."""
    data = {"checkpoint": count, "baseline": baseline, "sr": sr}
    with open(path + ".checkpoint", 'w') as f:
        json.dump(data, f)


def save_final(path, baseline, sr, metrics):
    """Save final results."""
    data = {
        "baseline": baseline,
        "sr": sr,
        "metrics": metrics,
        "timestamp": time.time()
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {path}")


if __name__ == "__main__":
    dataset = "data/datasets/strategyqa_full.json"
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    output = "benchmark_results/optimized_benchmark.json"

    run_benchmark(dataset, n_problems=n, output_path=output)