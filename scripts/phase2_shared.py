"""
Shared benchmark module for all 5 methods.
"""
import json, time, requests, re, sys, hashlib, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.unified_extractor import UnifiedAnswerExtractor
from dotenv import load_dotenv
load_dotenv()

MODEL = "meta/llama-3.1-405b-instruct"
API_KEY = os.getenv("NVIDIA_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
DELAY = 3.0

KNOWLEDGE_BASE = """IMPORTANT KNOWLEDGE BASE - Use for multi-step reasoning:

PHYSICS:
- Hot water CAN freeze faster than cold water (Mpemba effect: evaporation, convection, dissolved gases under specific conditions)
- Lightning reaches ~30,000 K, hotter than the sun's surface (~5,778 K)
- A coin falling from Empire State Building reaches terminal velocity ~50 m/s - insufficient to kill
- Sound requires a medium (air, water, solid) to travel; space is a vacuum with no medium
- Glass is an amorphous solid (not crystalline), sometimes called supercooled liquid, but does not flow appreciably
- The sneeze reflex usually closes eyelids - it requires conscious effort to keep them open
- The Great Wall of China is too narrow (~6-10m wide) to be visible from space with naked eye

CHEMISTRY:
- Diamonds are pure carbon (C) and will burn at ~700C in oxygen atmosphere
- Pure water is a poor conductor; dissolved ions (salts, minerals) conduct electricity
- The 5-second rule is a myth - bacteria transfer instantaneously upon contact
- Paper burns at ~232C; water boils at 100C - water in paper cup absorbs heat, keeping paper below ignition

BIOLOGY:
- Fish need dissolved OXYGEN in water, not water itself - insufficient O2 causes them to drown
- All birds are oviparous (egg-laying) - no bird gives live birth
- Bats: 3 vampire bat species (Desmodus, Diphylla, Diaemus) feed on blood of other animals
- Penguins are flightless birds - wings evolved into flippers for swimming
- Spiders are arachnids (not insects) - have 8 legs, no wings
- You CANNOT swallow your tongue - it's attached to floor of mouth by frenulum
- Bulls are partially colorblind - they react to movement, not red color

COMMON MISCONCEPTIONS:
- We do NOT use only 10% of our brain - fMRI shows widespread activity
- The moon does NOT have a permanent "dark side" - both sides receive sunlight due to tidal locking
- Gold is considered a good investment hedge against inflation - but returns are variable"""

SYSTEM_ZEROSHOT = "You are a helpful assistant. Answer yes/no questions with ONLY yes or no. Think step by step, then provide your final answer."
SYSTEM_COT = "You are a helpful assistant. Think step by step, then answer YES or NO to the question."
SYSTEM_RAG = f"""You are a helpful assistant. Use the following knowledge base when answering questions.

{KNOWLEDGE_BASE}

Answer yes/no questions with ONLY yes or no. Provide your final answer."""


def call_api(messages, max_tokens=150, temp=0.3, retries=8):
    for attempt in range(retries):
        try:
            resp = requests.post(
                'https://integrate.api.nvidia.com/v1/chat/completions',
                headers=HEADERS,
                json={"model": MODEL, "messages": messages, "temperature": temp, "max_tokens": max_tokens},
                timeout=240
            )
            if resp.status_code == 200:
                return resp.json()['choices'][0]['message']['content']
            if resp.status_code in (400, 502, 503, 504):
                wait = min(120, 15 * (attempt + 1))
                print(f"\n    [{resp.status_code} retry {attempt+1}/{retries}, wait {wait}s]", flush=True)
                time.sleep(wait)
                continue
            if resp.status_code == 429:
                print(f"\n    [429 rate limit, wait 120s]", flush=True)
                time.sleep(120)
                continue
            raise Exception(f"API error {resp.status_code}: {resp.text[:150]}")
        except requests.exceptions.Timeout:
            print(f"\n    [Timeout retry {attempt+1}/{retries}]", flush=True)
            time.sleep(20)
            continue
    raise Exception("All retries failed")


def extract_yesno(text):
    extracted = UnifiedAnswerExtractor.extract(text)
    answer = extracted.answer.lower().strip()
    if answer in ('yes', 'no'):
        return answer
    return 'unknown'


def load_questions(start, end):
    with open('data/datasets/benchmark_final.json') as f:
        data = json.load(f)
    data = [q for q in data if str(q.get('answer','')).lower().startswith('yes') or str(q.get('answer','')).lower().startswith('no')]
    return data[start:end]


def normalize_answer(a):
    a = a.lower().strip()
    if a.startswith('yes'): return 'yes'
    if a.startswith('no'): return 'no'
    return 'unknown'


def get_checkpoint_path(method_name, start, end):
    return f"benchmark_results/comp_{method_name}_s{start}_e{end}.json"


def run_benchmark(method, questions, method_name, start_idx=0, checkpoint_every=10):
    results = []
    checkpoint_path = get_checkpoint_path(method_name, start_idx, len(questions) + start_idx)

    try:
        with open(checkpoint_path) as f:
            results = json.load(f)
        print(f"Resuming from checkpoint: {len(results)} done", flush=True)
    except:
        pass

    start_from = len(results)

    for i, q in enumerate(questions[start_from:], start=start_from):
        qid = q.get('id', f'q_{i}')
        correct = normalize_answer(q['answer'])
        print(f"\n[{i+1}/{len(questions)}] {qid}: {q['question'][:55]}... (GT={correct})", flush=True)
        t0 = time.time()
        try:
            answer, info = method.run(q['question'])
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            answer = 'error'
            info = {}
        elapsed = time.time() - t0
        ok = "OK" if answer == correct else "ERR"
        extra = ""
        if isinstance(info, dict) and 'votes' in info:
            v = info['votes']
            extra = f" votes={sum(1 for x in v if x=='yes')}-{sum(1 for x in v if x=='no')}"
        print(f"  {method_name}: {answer} (GT={correct}) [{ok}] [{elapsed:.0f}s]{extra}", flush=True)
        results.append({
            'id': qid,
            'question': q['question'],
            'answer': answer,
            'correct': correct,
            'elapsed': elapsed
        })
        if isinstance(info, dict) and 'votes' in info:
            results[-1]['votes'] = info['votes']
        if isinstance(info, dict) and 'response' in info:
            results[-1]['response'] = info['response']

        if (i + 1) % checkpoint_every == 0:
            with open(checkpoint_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  [Checkpoint {len(results)} saved]", flush=True)

    with open(checkpoint_path, 'w') as f:
        json.dump(results, f, indent=2)

    correct_count = sum(1 for r in results if r['answer'] == r['correct'])
    print(f"\n=== {method_name.upper()} ===")
    print(f"Accuracy: {correct_count}/{len(results)} = {100*correct_count/len(results):.1f}%")
    return results