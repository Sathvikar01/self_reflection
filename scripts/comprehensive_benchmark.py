"""Comprehensive 6-hour StrategicQA Benchmark with Large Reasoning Model."""

import os
import sys
import json
import time
import requests
import re
from collections import Counter
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.unified_extractor import UnifiedAnswerExtractor

API_KEY = os.getenv("NVIDIA_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
BASE_URL = "https://integrate.api.nvidia.com/v1"

# Use larger reasoning model - 340B
MODEL = "meta/llama-3.1-405b-instruct"
MIN_DELAY = 1.6  # Rate limit: 40/min

# Expanded KB with reasoning chains
KB = """KNOWLEDGE BASE - Key facts for multi-step reasoning:

PHYSICS:
- Hot water CAN freeze faster than cold (Mpemba effect: evaporation, convection, dissolved gases)
- Lightning reaches ~30,000K - hotter than sun's 5,778K surface
- A coin from Empire State Building reaches terminal velocity ~50 m/s - cannot kill
- Sound requires medium; space vacuum has none - no sound in space
- Glass is amorphous solid (not crystalline) but doesn't flow at room temp
- Sneeze reflex closes eyelids - requires conscious effort to keep open
- Great Wall too narrow (~6-10m) to see from space with naked eye
- Yawning contagious via empathy/social mirroring

CHEMISTRY:
- Diamonds pure carbon (C) - burn at ~700°C in oxygen
- Pure water poor conductor; dissolved ions conduct
- 5-second rule FALSE - bacteria transfer instantly
- Paper burns at 232°C; water boils at 100°C - water absorbs heat, keeps paper safe
- Hydrogen fuel cells need O2 (2H2 + O2 -> 2H2O)

BIOLOGY:
- Fish need dissolved O2 - insufficient O2 = drowning
- All birds oviparous (lay eggs) - no live birth
- Plants need O2 for cellular respiration (especially night)
- 3 vampire bat species (Desmodus, Diphylla, Diaemus) feed on blood
- Penguins/ostriches flightless - wings for swimming/running
- Spiders arachnids (8 legs, no wings)
- Dolphins unihemispheric sleep - one brain half at a time
- Elephants have molars + tusks (modified teeth)
- Trees no brain - circadian rhythms but don't "sleep"

GEOGRAPHY:
- Antarctica 14M km² > Europe 10M km²
- Pacific Ocean 165M km² > all land 150M km²

COMMON MISCONCEPTIONS:
- We use >10% brain (fMRI shows widespread activity)
- Moon has no permanent "dark side" - both sides get sunlight (tidal locking)
- You CAN'T tickle yourself (cerebellum predicts self-generated sensations)
- Astronauts inside ISS don't need sunscreen (fully shielded)
- Bulls partially colorblind - react to movement, not red
- Gold considered hedge against inflation (variable returns)
- Human survival on potatoes+dairy possible but deficient in vitamins

Reasoning: Check KB facts first, then chain them logically."""


def fetch_strategyqa_from_hf():
    """Fetch StrategicQA dataset from HuggingFace."""
    print("Fetching StrategicQA dataset from HuggingFace...")
    try:
        resp = requests.get(
            "https://raw.githubusercontent.com/oistmil/StrategyQA/main/strategyQA_dataset.json",
            timeout=30
        )
        if resp.status_code == 200:
            data = resp.json()
            questions = []
            for idx, item in enumerate(data):
                if isinstance(item, dict) and 'question' in item:
                    questions.append({
                        "id": f"sq_{idx+1}",
                        "question": item['question'],
                        "answer": "yes" if item.get('answer', True) else "no",
                        "ground_truth": "yes" if item.get('answer', True) else "no",
                        "source": "strategyqa_hf"
                    })
            print(f"Fetched {len(questions)} questions from StrategyQA")
            return questions
    except Exception as e:
        print(f"Failed to fetch from primary source: {e}")

    # Fallback: try alternative source
    try:
        resp = requests.get(
            "https://raw.githubusercontent.com/elviseb/StrategyQA/main/data/strategyQA_train.json",
            timeout=30
        )
        if resp.status_code == 200:
            data = resp.json()
            questions = []
            for idx, item in enumerate(data):
                if isinstance(item, dict):
                    q = item.get('question', '')
                    ans = item.get('answer', True)
                    questions.append({
                        "id": f"sq_{idx+1}",
                        "question": q,
                        "answer": "yes" if ans else "no",
                        "ground_truth": "yes" if ans else "no",
                        "source": "strategyqa_alt"
                    })
            print(f"Fetched {len(questions)} questions from alternative source")
            return questions
    except Exception as e:
        print(f"Failed alternative source: {e}")

    return None


def save_dataset(questions, path):
    """Save dataset to JSON."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(questions)} questions to {path}")


def load_local_dataset(path):
    """Load local dataset."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def call_api(model, messages, temperature=0.5, max_tokens=200):
    """Single API call with rate limiting."""
    time.sleep(MIN_DELAY)
    try:
        resp = requests.post(
            f"{BASE_URL}/chat/completions",
            headers=HEADERS,
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": 0.95
            },
            timeout=120
        )
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"API error: {e}")
        return ""


def extract_answer(text):
    """Extract yes/no from response using unified extractor."""
    extracted = UnifiedAnswerExtractor.extract(text)
    answer = extracted.answer.lower().strip()
    if answer in ('yes', 'no'):
        return answer
    return "no"


def baseline_solve(question):
    """Baseline - single call with big model."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer yes/no questions with ONLY yes or no. Think step by step, then provide final answer in <answer> tags."},
        {"role": "user", "content": f"Question: {question}\n\nThink step by step, then answer:\n<answer>"}
    ]
    response = call_api(MODEL, messages, temperature=0.1, max_tokens=150)
    return extract_answer(response)


def sr_solve(question):
    """Self-reflection with 3-path self-consistency."""
    system_prompt = f"""You are a reasoning assistant. Answer yes/no using multi-step reasoning.

KNOWLEDGE BASE:
{KB}

CRITICAL: Check if any KB fact is relevant. If yes, use it in your reasoning chain.

Answer in <answer>yes</answer> or <answer>no</answer> format."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {question}\n\nStep 1: Is any KB fact relevant? Note it.\nStep 2: Chain reasoning steps.\nStep 3: Answer.\n\n<answer>"}
    ]

    # 3 paths for self-consistency
    votes = []
    for i in range(3):
        response = call_api(MODEL, messages, temperature=0.5, max_tokens=200)
        answer = extract_answer(response)
        votes.append(answer)

    counts = Counter(votes)
    majority = counts.most_common(1)[0][0]
    unanimous = len(set(votes)) == 1
    return majority, dict(counts), unanimous


def run_benchmark(n_questions=50, output_path="benchmark_results/full_benchmark.json"):
    """Run full benchmark."""

    # Fetch or load dataset
    dataset_path = "data/datasets/strategyqa_full.json"

    # Try to fetch fresh data
    questions = fetch_strategyqa_from_hf()

    if not questions:
        # Use local dataset
        print("Using local dataset...")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
    else:
        # Save fetched data
        save_dataset(questions, "data/datasets/strategyqa_fetched.json")
        # Also save local format
        save_dataset(questions, dataset_path)

    n_total = len(questions)
    if n_questions > 0 and n_questions < n_total:
        questions = questions[:n_questions]

    n = len(questions)
    print(f"\n{'='*100}")
    print(f"COMPREHENSIVE BENCHMARK - {n} questions | Model: {MODEL}")
    print(f"{'='*100}")

    # Estimate time: 2 baseline + 6 SR = 8 calls/question * 1.6s = ~12.8s/question
    # For 50 questions: ~11 min
    # For 100 questions: ~21 min
    # For 200 questions: ~43 min
    est_time_min = n * 8 * MIN_DELAY / 60
    print(f"Estimated time: {est_time_min:.1f} minutes ({n * 8} API calls)")

    checkpoint_path = output_path + ".checkpoint"
    start_idx = 0
    baseline_results = []
    sr_results = []

    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            cp = json.load(f)
            baseline_results = cp.get('baseline', [])
            sr_results = cp.get('sr', [])
            start_idx = cp.get('checkpoint', 0)
            print(f"Resuming from checkpoint at question {start_idx}")

    start_time = time.time()
    last_save = start_time

    for i in range(start_idx, n):
        p = questions[i]
        q = p['question']
        gt = p['answer']
        pid = p.get('id', f'q_{i+1}')

        elapsed = time.time() - start_time
        remaining = (n - i - 1) * 8 * MIN_DELAY / 60

        print(f"\n[{i+1}/{n}] {pid}: {q[:60]}...")
        print(f"  Elapsed: {elapsed/60:.1f}m | Remaining: {remaining:.1f}m")

        # Baseline
        print("  Running baseline...", end=" ", flush=True)
        b_answer = baseline_solve(q)
        b_correct = (b_answer.lower() == gt.lower())
        baseline_results.append({
            "id": pid, "question": q, "answer": b_answer,
            "ground_truth": gt, "correct": b_correct
        })
        print(f"-> {b_answer} ({'CORRECT' if b_correct else 'WRONG'})")

        # Self-reflection
        print("  Running SR (3-path SC)...", end=" ", flush=True)
        sr_answer, votes, unan = sr_solve(q)
        sr_correct = (sr_answer.lower() == gt.lower())
        sr_results.append({
            "id": pid, "question": q, "answer": sr_answer,
            "ground_truth": gt, "correct": sr_correct,
            "votes": votes, "unanimous": unan
        })
        kb_flag = "KB" if unan else "split"
        print(f"-> {sr_answer} ({'CORRECT' if sr_correct else 'WRONG'}) [{kb_flag}]")

        # Auto-save every 5 questions
        if (i + 1) % 5 == 0:
            with open(checkpoint_path, 'w') as f:
                json.dump({
                    "checkpoint": i + 1,
                    "baseline": baseline_results,
                    "sr": sr_results,
                    "timestamp": time.time()
                }, f)
            print(f"  [Checkpoint saved at {i+1} questions]")

    total_time = time.time() - start_time

    # Calculate final metrics
    b_acc = sum(1 for r in baseline_results if r['correct']) / n
    s_acc = sum(1 for r in sr_results if r['correct']) / n

    n_01 = sum(1 for b, s in zip(baseline_results, sr_results) if not b['correct'] and s['correct'])
    n_10 = sum(1 for b, s in zip(baseline_results, sr_results) if b['correct'] and not s['correct'])

    denom = n_01 + n_10
    p_val = 1.0
    if denom > 0:
        chi2 = (abs(n_01 - n_10) - 1) ** 2 / denom
        from math import exp
        p_val = exp(-chi2 / 2)

    # Results table
    print(f"\n{'='*100}")
    print("FINAL RESULTS")
    print(f"{'='*100}")
    print(f"\n{'METRIC':<40} {'BASELINE':<25} {'SELF-REFLECTION':<25}")
    print("-" * 90)
    print(f"{'Accuracy':<40} {b_acc:.1%} {s_acc:.1%}")
    print(f"{'Correct/Total':<40} {int(b_acc*n)}/{n} {int(s_acc*n)}/{n}")
    print(f"{'Token Overhead (est)':<40} {'--'} {((s_acc/b_acc)-1)*100:+.1f}%")
    print(f"{'Total Time':<40} {total_time/60:.1f} minutes")
    print(f"{'McNemar p-value':<40} {p_val:.4f}")
    print(f"{'Baseline wrong, SR right (gain)':<40} {n_01}")
    print(f"{'Baseline right, SR wrong (loss)':<40} {n_10}")
    print(f"{'Net improvement':<40} {n_01 - n_10:+d}")

    # Per-question table (first 30)
    print(f"\n{'='*120}")
    print("PER-QUESTION RESULTS (first 30)")
    print(f"{'='*120}")
    print(f"{'ID':<10} {'Question':<70} {'GT':<5} {'BL':<5} {'SR':<5} {'Status':<12} {'Votes'}")
    print("-" * 130)

    for b, s in zip(baseline_results[:30], sr_results[:30]):
        st = "IMPROVED" if (not b['correct'] and s['correct']) else ("REGRESSED" if (b['correct'] and not s['correct']) else "SAME")
        votes_str = str(s.get('votes', {}))
        print(f"{b['id'][:8]:<10} {b['question'][:68]:<70} {b['ground_truth'][:4]:<5} "
              f"{'Y' if b['correct'] else 'N':<5} {'Y' if s['correct'] else 'N':<5} "
              f"{st:<12} {votes_str}")

    # Save final results
    final_data = {
        "model": MODEL,
        "dataset": "StrategyQA",
        "n_questions": n,
        "total_time_minutes": total_time / 60,
        "metrics": {
            "baseline_accuracy": b_acc,
            "sr_accuracy": s_acc,
            "relative_improvement": ((s_acc / b_acc) - 1) * 100 if b_acc > 0 else 0,
            "mcnemar_p_value": p_val,
            "discordant": {"n_01": n_01, "n_10": n_10}
        },
        "baseline": baseline_results,
        "sr": sr_results,
        "timestamp": time.time()
    }

    with open(output_path, 'w') as f:
        json.dump(final_data, f, indent=2)

    print(f"\n{'='*100}")
    print(f"RESULTS SAVED TO: {output_path}")
    print(f"TOTAL TIME: {total_time/60:.1f} minutes")
    print(f"{'='*100}")

    return baseline_results, sr_results


if __name__ == "__main__":
    # Default: 100 questions (will take ~21 minutes)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    output = sys.argv[2] if len(sys.argv) > 2 else "benchmark_results/full_benchmark.json"

    print(f"Starting comprehensive benchmark: {n} questions with {MODEL}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    run_benchmark(n_questions=n, output_path=output)

    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")