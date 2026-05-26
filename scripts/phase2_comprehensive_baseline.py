"""
Phase 2: Comprehensive 5-baseline benchmark on 405B model.

Methods:
1. Zero-Shot: Simple yes/no question, no KB
2. CoT: "Think step by step" prompt, no KB
3. SC-only: 7-path self-consistency WITHOUT KB (proves SC needs KB)
4. Simple RAG: KB in system prompt (no Step1/Step2), 1 path
5. KB+SC+Step1/2: Our full pipeline (7-path SC + KB + Step1/2 prompt)

This benchmark answers the key paper question: Is KB+SC better than established methods?
"""
import json
import time
import requests
import hashlib
import re
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.unified_extractor import UnifiedAnswerExtractor

MODEL = "meta/llama-3.1-405b-instruct"
API_KEY = os.getenv("NVIDIA_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# Knowledge base (same as anti_overfitting_pipeline.py)
KNOWLEDGE_BASE = """IMPORTANT KNOWLEDGE BASE - Use for multi-step reasoning:

PHYSICS:
- Hot water CAN freeze faster than cold water (Mpemba effect: evaporation, convection, dissolved gases under specific conditions)
- Lightning reaches ~30,000 K, hotter than the sun's surface (~5,778 K)
- A coin falling from Empire State Building reaches terminal velocity ~50 m/s - insufficient to kill
- Sound requires a medium (air, water, solid) to travel; space is a vacuum with no medium
- Glass is an amorphous solid (not crystalline), sometimes called supercooled liquid, but does not flow appreciably
- The sneeze reflex usually closes eyelids - it requires conscious effort to keep them open
- The Great Wall of China is too narrow (~6-10m wide) to be visible from space with naked eye
- Yawning is contagious - linked to empathy and social mirroring in humans

CHEMISTRY:
- Diamonds are pure carbon (C) and will burn at ~700C in oxygen atmosphere
- Pure water is a poor conductor; dissolved ions (salts, minerals) conduct electricity
- The 5-second rule is a myth - bacteria transfer instantaneously upon contact
- Paper burns at ~232C; water boils at 100C - water in paper cup absorbs heat, keeping paper below ignition
- Hydrogen fuel cells require oxygen to generate electricity

BIOLOGY:
- Fish need dissolved OXYGEN in water, not water itself - insufficient O2 causes them to drown
- All birds are oviparous (egg-laying) - no bird gives live birth
- Plants perform cellular respiration using O2, especially at night
- Bats: 3 vampire bat species feed on blood of other animals
- Penguins are flightless birds - wings evolved into flippers for swimming
- Ostriches are flightless - wings too small, body too heavy for flight
- Spiders are arachnids (not insects) - have 8 legs, no wings
- Dolphins practice unihemispheric sleep - one brain hemisphere at a time
- Elephants have 6 sets of molars in lifetime
- Trees have no brain/nervous system
- Gold is considered a good investment hedge against inflation - but returns are variable
- Bulls are partially colorblind - they react to movement, not red color
- You CANNOT swallow your tongue - it's attached to floor of mouth

COMMON MISCONCEPTIONS:
- We do NOT use only 10% of our brain - fMRI shows widespread activity
- The moon does NOT have a permanent "dark side" - both sides receive sunlight due to tidal locking"""

SYSTEM_ZEROSHOT = "You are a helpful assistant. Answer yes/no questions with ONLY yes or no. Think step by step, then provide your final answer."
SYSTEM_COT = "You are a helpful assistant. Think step by step, then answer YES or NO to the question."
SYSTEM_RAG = f"""You are a helpful assistant. Use the following knowledge base when answering questions.

{KNOWLEDGE_BASE}

Answer yes/no questions with ONLY yes or no. Provide your final answer."""


def call_api(messages, max_tokens=150, temp=0.3, retries=5):
    """Call NIM API with retry and backoff."""
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
            if resp.status_code in (502, 503, 504):
                wait = (attempt + 1) * 15
                print(f"  [{resp.status_code} retry {attempt+1}, wait {wait}s]", flush=True)
                time.sleep(wait)
                continue
            if resp.status_code == 429:
                wait = (attempt + 1) * 30
                print(f"  [429 rate limit, wait {wait}s]", flush=True)
                time.sleep(wait)
                continue
            raise Exception(f"API error {resp.status_code}: {resp.text[:100]}")
        except requests.exceptions.Timeout:
            print(f"  [Timeout retry {attempt+1}]", flush=True)
            time.sleep(10)
            continue
    raise Exception("All retries failed")


def extract_yesno(text):
    """Extract yes/no from response using unified extractor."""
    if not text:
        return 'unknown'
    extracted = UnifiedAnswerExtractor.extract(text)
    answer = extracted.answer.lower().strip()
    if answer in ('yes', 'no'):
        return answer
    return 'unknown'


def method_zeroshot(question, delay=1.8):
    """Method 1: Zero-shot baseline."""
    time.sleep(delay)
    messages = [
        {"role": "system", "content": SYSTEM_ZEROSHOT},
        {"role": "user", "content": f"Question: {question}\nAnswer with ONLY yes or no."}
    ]
    resp = call_api(messages, max_tokens=80)
    return extract_yesno(resp), resp


def method_cot(question, delay=1.8):
    """Method 2: Chain-of-Thought (Wei et al., 2022)."""
    time.sleep(delay)
    messages = [
        {"role": "system", "content": SYSTEM_COT},
        {"role": "user", "content": f"Question: {question}\n\nThink step by step, then answer YES or NO."}
    ]
    resp = call_api(messages, max_tokens=300)
    return extract_yesno(resp), resp


def method_sc_only(question, delay=2.2):
    """Method 3: Vanilla Self-Consistency WITHOUT KB (Wang et al., 2022).

    7-path majority voting, no knowledge injection.
    Tests: Does reasoning diversity alone overcome knowledge gaps?
    """
    votes = []
    t0 = time.time()

    def gen_path(i):
        time.sleep(delay)
        messages = [
            {"role": "system", "content": SYSTEM_COT},
            {"role": "user", "content": f"Question: {question}\n\nThink step by step, then answer YES or NO."}
        ]
        try:
            resp = call_api(messages, max_tokens=300, temp=0.4)
        except Exception as e:
            print(f"  [SC path {i} error: {e}]", flush=True)
            return 'unknown'
        return extract_yesno(resp)

    # Generate 7 paths
    for i in range(7):
        votes.append(gen_path(i))
        if i < 6:
            time.sleep(0.5)

    elapsed = time.time() - t0
    # Majority vote
    yes_votes = sum(1 for v in votes if v == 'yes')
    no_votes = sum(1 for v in votes if v == 'no')
    if yes_votes > no_votes:
        answer = 'yes'
    elif no_votes > yes_votes:
        answer = 'no'
    else:
        answer = votes[0]  # Tie: pick first

    return answer, {'votes': votes, 'yes': yes_votes, 'no': no_votes, 'elapsed': elapsed}


def method_rag(question, delay=1.8):
    """Method 4: Simple RAG (Lewis et al., 2020) — KB in system prompt, no Step1/Step2.

    Tests: Does simple KB injection beat zero-shot?
    """
    time.sleep(delay)
    messages = [
        {"role": "system", "content": SYSTEM_RAG},
        {"role": "user", "content": f"Question: {question}\n\nRefer to the knowledge base above and answer YES or NO."}
    ]
    resp = call_api(messages, max_tokens=300)
    return extract_yesno(resp), resp


def method_kb_sc(question, delay=2.2):
    """Method 5: Full pipeline — 7-path SC + KB + Step1/Step2 (our method)."""
    votes = []
    t0 = time.time()

    def gen_path(i):
        time.sleep(delay)
        messages = [
            {"role": "system", "content": f"""You are a helpful assistant. Use the following knowledge base when answering.

{KNOWLEDGE_BASE}

Answer yes/no questions with ONLY yes or no."""},
            {"role": "user", "content": f"""Question: {question}

Step 1: Does any fact in the knowledge base directly relate to this question? If yes, state it.
Step 2: Based on the knowledge base, answer YES or NO.

Reason step by step, then give your final answer."""}
        ]
        try:
            resp = call_api(messages, max_tokens=400, temp=0.4)
        except Exception as e:
            print(f"  [KB+SC path {i} error: {e}]", flush=True)
            return 'unknown'
        return extract_yesno(resp)

    # Generate 7 paths
    for i in range(7):
        votes.append(gen_path(i))
        if i < 6:
            time.sleep(0.5)

    elapsed = time.time() - t0
    # Majority vote
    yes_votes = sum(1 for v in votes if v == 'yes')
    no_votes = sum(1 for v in votes if v == 'no')
    if yes_votes > no_votes:
        answer = 'yes'
    elif no_votes > yes_votes:
        answer = 'no'
    else:
        answer = votes[0]

    return answer, {'votes': votes, 'yes': yes_votes, 'no': no_votes, 'elapsed': elapsed}


def run_benchmark(n_questions=None, start_idx=0, checkpoint_every=10):
    """Run full 5-method benchmark on benchmark_final.json."""
    with open('data/datasets/benchmark_final.json') as f:
        questions = json.load(f)

    if n_questions:
        questions = questions[start_idx:start_idx + n_questions]
    else:
        questions = questions[start_idx:]

    print(f"Benchmarking {len(questions)} questions starting at index {start_idx}")
    print(f"Methods: (1) Zero-Shot, (2) CoT, (3) SC-only, (4) RAG, (5) KB+SC+Step1/2")
    print("=" * 70)

    results = []
    checkpoint_path = f"benchmark_results/comprehensive_checkpoint_{start_idx}.json"

    # Load checkpoint if exists
    try:
        with open(checkpoint_path) as f:
            results = json.load(f)
        print(f"Resuming from checkpoint: {len(results)} already done")
    except:
        pass

    start_from = len(results)

    for i, q in enumerate(questions[start_from:], start=start_from):
        qid = q.get('id', f'q_{i}')
        question = q['question']
        correct = q['answer'].lower().strip()
        correct = 'yes' if correct.startswith('yes') else 'no' if correct.startswith('no') else correct

        print(f"\n[{i+1}/{len(questions)}] {qid}: {question[:55]}... (GT={correct})", flush=True)
        t0 = time.time()

        # Method 1: Zero-shot
        t0m = time.time()
        z_answer, z_resp = method_zeroshot(question, delay=2.2)
        t1m = time.time()
        print(f"  (1) ZeroShot: {z_answer} (GT={correct}) {'OK' if z_answer==correct else 'ERR'} [{t1m-t0m:.0f}s]", flush=True)

        # Method 2: CoT
        t0m = time.time()
        cot_answer, cot_resp = method_cot(question, delay=2.2)
        t1m = time.time()
        print(f"  (2) CoT:     {cot_answer} (GT={correct}) {'OK' if cot_answer==correct else 'ERR'} [{t1m-t0m:.0f}s]", flush=True)

        # Method 3: SC-only (no KB)
        t0m = time.time()
        sc_answer, sc_info = method_sc_only(question, delay=2.2)
        t1m = time.time()
        print(f"  (3) SC-only: {sc_answer} (GT={correct}) {'OK' if sc_answer==correct else 'ERR'} [{t1m-t0m:.0f}s] votes={sc_info['yes']}-{sc_info['no']}", flush=True)

        # Method 4: Simple RAG
        t0m = time.time()
        rag_answer, rag_resp = method_rag(question, delay=2.2)
        t1m = time.time()
        print(f"  (4) RAG:     {rag_answer} (GT={correct}) {'OK' if rag_answer==correct else 'ERR'} [{t1m-t0m:.0f}s]", flush=True)

        # Method 5: KB+SC+Step1/2
        t0m = time.time()
        kb_answer, kb_info = method_kb_sc(question, delay=2.2)
        t1m = time.time()
        print(f"  (5) KB+SC:   {kb_answer} (GT={correct}) {'OK' if kb_answer==correct else 'ERR'} [{t1m-t0m:.0f}s] votes={kb_info['yes']}-{kb_info['no']}", flush=True)

        total_elapsed = time.time() - t0
        print(f"  Total: {total_elapsed:.0f}s", flush=True)

        results.append({
            'id': qid,
            'question': question,
            'correct': correct,
            'zeroshot': z_answer,
            'cot': cot_answer,
            'sc_only': sc_answer,
            'rag': rag_answer,
            'kb_sc': kb_answer,
            'sc_only_votes': sc_info['votes'],
            'kb_sc_votes': kb_info['votes'],
            'elapsed': total_elapsed
        })

        # Checkpoint
        if (i + 1) % checkpoint_every == 0:
            with open(checkpoint_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  [Checkpoint saved: {len(results)}/{len(questions)}]", flush=True)

    # Final save
    final_path = f"benchmark_results/comprehensive_405b_{start_idx}.json"
    with open(final_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    methods = ['zeroshot', 'cot', 'sc_only', 'rag', 'kb_sc']
    method_names = ['Zero-Shot', 'CoT (Wei 2022)', 'SC-only (Wang 2022)', 'RAG (Lewis 2020)', 'KB+SC+Step1/2']
    for name, m in zip(method_names, methods):
        correct = sum(1 for r in results if r[m] == r['correct'])
        pct = 100 * correct / len(results)
        print(f"  {name}: {correct}/{len(results)} = {pct:.1f}%")

    return results


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    run_benchmark(n_questions=n, start_idx=start)