"""
Fixed comprehensive benchmark with improved extraction.
Resumes from checkpoints automatically.
Usage: python phase2_fixed.py [all|zs|cot|rag|sc|kbsc]
"""
import json, time, requests, re, sys, os, subprocess

MODEL = "mistralai/mistral-large-3-675b-instruct-2512"
API_KEY = "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"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
DELAY = 2.5

KB = """IMPORTANT KNOWLEDGE BASE:
- Hot water CAN freeze faster than cold water (Mpemba effect)
- Diamonds are pure carbon and burn at ~700C in oxygen
- Glass is an amorphous solid (not crystalline, sometimes flows)
- Fish need dissolved OXYGEN in water - insufficient O2 causes drowning
- All birds are oviparous (egg-laying) - no bird gives live birth
- Spiders are arachnids with 8 legs, no wings
- You CANNOT swallow your tongue (attached to mouth floor)
- Sound requires a medium - space is vacuum (no sound)
- Lightning reaches ~30,000 K - hotter than sun's surface (~5,778 K)
- Paper burns at ~232C; water boils at 100C (water in paper cup absorbs heat)
- A coin from Empire State Building reaches ~50 m/s - usually not fatal
- The Great Wall of China is too narrow to be visible from space with naked eye
- We do NOT use only 10% of our brain (fMRI shows widespread activity)
- Bulls are partially colorblind - react to movement, not red color
- Bats: 3 vampire bat species feed on blood
- Penguins are flightless birds (wings as flippers)
- Dolphins practice unihemispheric sleep
- Gold is considered a good investment hedge against inflation
- The moon has no permanent dark side (tidal locking)"""

SYSTEM_ZS = "You are a helpful assistant. Answer yes/no questions with ONLY yes or no."
SYSTEM_COT = "You are a helpful assistant. Think step by step, then answer YES or NO."
SYSTEM_RAG = f"You are a helpful assistant. Use this knowledge base.\n{KB}\nAnswer yes/no with ONLY yes or no."
SYSTEM_KB = f"You are a helpful assistant. Use this knowledge base.\n{KB}\nAnswer yes/no with ONLY yes or no."

CALL_COUNT = [0]

def call_api(messages, max_tokens=150, temp=0.3, retries=8):
    for attempt in range(retries):
        try:
            CALL_COUNT[0] += 1
            resp = requests.post(
                'https://integrate.api.nvidia.com/v1/chat/completions',
                headers=HEADERS,
                json={"model": MODEL, "messages": messages, "temperature": temp, "max_tokens": max_tokens},
                timeout=300
            )
            if resp.status_code == 200:
                return resp.json()['choices'][0]['message']['content']
            if resp.status_code in (400, 502, 503, 429):
                wait = min(120, 20 * (attempt + 1))
                print(f"\n      [{resp.status_code} retry {attempt+1}/{retries}, wait {wait}s]", flush=True)
                time.sleep(wait)
                continue
            raise Exception(f"API error {resp.status_code}: {resp.text[:80]}")
        except requests.exceptions.Timeout:
            print(f"\n      [Timeout retry {attempt+1}/{retries}]", flush=True)
            time.sleep(30)
            continue
        except requests.exceptions.ConnectionError:
            print(f"\n      [Connection reset retry {attempt+1}/{retries}]", flush=True)
            time.sleep(30)
            continue
    return None

def extract_yesno(text):
    """Improved extraction: search entire response for yes/no."""
    if not text:
        return 'unknown'
    t = text.lower().strip()
    
    # 1. Check first 100 chars for direct answer
    first100 = t[:100]
    if first100.startswith('yes'):
        return 'yes'
    if first100.startswith('no'):
        return 'no'
    
    # 2. Look for "answer: yes" or "answer: no" patterns anywhere
    m = re.search(r'answer[:\s]+(yes|no)\b', t)
    if m:
        return m.group(1)
    
    # 3. Look for "final answer" pattern
    m = re.search(r'final answer[:\s]+(yes|no)\b', t)
    if m:
        return m.group(1)
    
    # 4. Look for "the answer is yes/no" pattern
    m = re.search(r'the answer is[:\s]+(yes|no)\b', t)
    if m:
        return m.group(1)
    
    # 5. Check last 300 chars for standalone yes/no
    last300 = t[-300:]
    m_yes = re.search(r'\byes\b', last300)
    m_no = re.search(r'\bno\b', last300)
    if m_yes and not m_no:
        return 'yes'
    if m_no and not m_yes:
        return 'no'
    if m_yes and m_no:
        # Take the last occurrence
        return last300[max(m_yes.start(), m_no.start()):][:2]
    
    # 6. Check entire response for standalone yes/no (last resort)
    m_yes = re.search(r'\byes\b', t)
    m_no = re.search(r'\bno\b', t)
    if m_yes and not m_no:
        return 'yes'
    if m_no and not m_yes:
        return 'no'
    
    return 'unknown'

def norm_ans(a):
    a = str(a).lower().strip()
    return 'yes' if a.startswith('yes') else 'no' if a.startswith('no') else 'unknown'

def load_questions():
    with open('data/datasets/benchmark_final_v2.json') as f:
        data = json.load(f)
    return [q for q in data if norm_ans(q.get('answer','')) in ('yes','no')][:220]

def cp(name):
    p = f'benchmark_results/v6_{name}.json'
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return []

def save(name, results):
    with open(f'benchmark_results/v6_{name}.json', 'w') as f:
        json.dump(results, f, indent=2)

def run_method(name, system_prompt, user_template, delay=DELAY, sc_paths=0, sc_temp=0.4, max_tokens=150):
    results = cp(name)
    questions = load_questions()[len(results):]
    print(f"\n{name}: resuming from {len(results)}, {len(questions)} remaining", flush=True)

    for qi, q in enumerate(questions):
        idx = len(results) + qi
        gt = norm_ans(q['answer'])
        qid = q.get('id', f'q_{idx}')
        print(f"\n[{idx+1}] {qid}: {q['question'][:50]}... (GT={gt})", flush=True)
        t0 = time.time()

        if sc_paths == 0:
            time.sleep(delay)
            resp = call_api([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_template.format(question=q['question'])}
            ], max_tokens=max_tokens)
            ans = extract_yesno(resp) if resp else 'unknown'
            votes = []
        else:
            votes = []
            for j in range(sc_paths):
                time.sleep(delay)
                try:
                    resp = call_api([
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_template.format(question=q['question'])}
                    ], max_tokens=max_tokens, temp=sc_temp)
                    votes.append(extract_yesno(resp) if resp else 'unknown')
                except Exception as e:
                    print(f"      [path {j} error: {e}]", flush=True)
                    votes.append('unknown')
                if j < sc_paths - 1:
                    time.sleep(1)
            yes_v = sum(1 for v in votes if v == 'yes')
            no_v = sum(1 for v in votes if v == 'no')
            ans = 'yes' if yes_v > no_v else 'no' if no_v > yes_v else votes[0]

        elapsed = time.time() - t0
        ok = "OK" if ans == gt else "ERR"
        vote_str = f" votes={yes_v}-{no_v}" if votes else ""
        print(f"  {name}: {ans} (GT={gt}) [{ok}] [{elapsed:.0f}s]{vote_str}", flush=True)

        item = {'id': qid, 'question': q['question'], 'answer': ans, 'correct': gt}
        if votes:
            item['votes'] = votes
        results.append(item)

        if (idx + 1) % 3 == 0:
            save(name, results)
            correct = sum(1 for r in results if r['answer'] == r['correct'])
            print(f"  [Checkpoint {idx+1} | {correct}/{len(results)} = {100*correct/len(results):.1f}%]", flush=True)
            # Push to GitHub periodically
            try:
                subprocess.run(['git', 'add', 'benchmark_results/'], capture_output=True, timeout=10)
                subprocess.run(['git', 'commit', '-m', f'checkpoint: {name} {len(results)}/220'], capture_output=True, timeout=10)
            except:
                pass

    save(name, results)
    correct = sum(1 for r in results if r['answer'] == r['correct'])
    print(f"\n=== {name}: {correct}/{len(results)} = {100*correct/len(results):.1f}% ===")
    return results

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    questions = load_questions()
    print(f"Dataset: {len(questions)} yes/no questions", flush=True)

    if arg in ("zs", "all"):
        run_method("zeroshot", SYSTEM_ZS, "Question: {question}\nAnswer with ONLY yes or no.", max_tokens=100)

    if arg in ("cot", "all"):
        run_method("cot", SYSTEM_COT, "Question: {question}\n\nThink step by step, then answer YES or NO.", max_tokens=300)

    if arg in ("rag", "all"):
        run_method("rag", SYSTEM_RAG, "Question: {question}\n\nRefer to the knowledge base and answer YES or NO.", max_tokens=300)

    if arg in ("sc", "all"):
        run_method("sc_only", SYSTEM_COT, "Question: {question}\n\nThink step by step, then answer YES or NO.",
                   sc_paths=3, sc_temp=0.4, max_tokens=300)

    if arg in ("kbsc", "all"):
        step12 = """Question: {question}

Step 1: Does any fact in the knowledge base directly relate? If yes, state it.
Step 2: Based on the knowledge base, answer YES or NO.

Reason step by step."""
        run_method("kbsc", SYSTEM_KB, step12, sc_paths=3, sc_temp=0.4, max_tokens=400)

    print(f"\nTotal API calls: {CALL_COUNT[0]}")