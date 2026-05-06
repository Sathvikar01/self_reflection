"""Phase 1c: Generate ~170 more questions to reach 220 total benchmark size."""
import json, time, requests, re, hashlib

MODEL = "meta/llama-3.1-405b-instruct"
API_KEY = "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"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

with open("data/datasets/benchmark_final.json") as f:
    existing = json.load(f)
existing_qs = set(q["question"].lower().strip() for q in existing)
print(f"Existing: {len(existing)} questions")

CATEGORIES = [
    "causal_chain", "temporal_sequence", "conditional_logic", "comparative_inference",
    "negation_reasoning", "abductive_reasoning", "counterfactual", "analogy_reasoning",
    "probability_reasoning", "mathematical_reasoning", "spatial_reasoning", "commonsense_reasoning"
]

def call_model(messages, max_tokens=300, temp=0.8, retries=5):
    for attempt in range(retries):
        try:
            resp = requests.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                headers=HEADERS,
                json={"model": MODEL, "messages": messages, "temperature": temp, "max_tokens": max_tokens},
                timeout=180
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            if resp.status_code in (400, 502, 503, 429):
                wait = min(60, 10 * (attempt + 1))
                print(f"\n  [{resp.status_code} retry {attempt+1}, wait {wait}s]", flush=True)
                time.sleep(wait)
                continue
            raise Exception(f"API error {resp.status_code}")
        except requests.exceptions.Timeout:
            print(f"\n  [Timeout retry {attempt+1}]", flush=True)
            time.sleep(10)
            continue
    return None

def extract_json(text):
    if not text:
        return None
    m = re.search(r"\{[^{}]*\"question\"[^{}]*\"answer\"[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except:
            pass
    # Try full JSON
    try:
        return json.loads(text)
    except:
        return None

SYSTEM = """You are an expert at generating complex multi-step yes/no reasoning questions.
Output ONLY valid JSON with these exact fields: question, answer (yes/no), reasoning_steps (array of 3+ strings), category (one word)"""

TARGET_QUESTIONS = 170
new_qs = []
generated_count = 0

for i in range(TARGET_QUESTIONS):
    cat = CATEGORIES[i % len(CATEGORIES)]
    print(f"[{i+1}/{TARGET_QUESTIONS}] {cat}...", end=" ", flush=True)

    time.sleep(2)

    few_shot = """Example: {"question": "If all mammals breathe air and whales are mammals, can whales survive on land?", "answer": "no", "reasoning_steps": ["Whales are mammals", "Mammals need air to breathe", "Whales cannot breathe on land", "So whales cannot survive on land"], "category": "causal_chain"}"""

    resp = call_model([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Generate a {cat} question. {few_shot}\n\nOutput JSON:"}
    ])

    if resp:
        data = extract_json(resp)
        if data and "question" in data and "answer" in data:
            q_text = data["question"].lower().strip()
            if q_text not in existing_qs:
                item = {
                    "id": f"new_{i}",
                    "question": data["question"],
                    "answer": data["answer"].lower().strip()[:3],
                    "reasoning_steps": data.get("reasoning_steps", []),
                    "category": cat,
                    "source": "api_generated"
                }
                new_qs.append(item)
                existing_qs.add(q_text)
                generated_count += 1
                print(f"OK ({generated_count} total)")
            else:
                print("DUPLICATE")
        else:
            print(f"PARSE ERROR: {resp[:100]}")
    else:
        print("FAILED")

    if (i + 1) % 25 == 0:
        print(f"  [Checkpoint {generated_count}/{TARGET_QUESTIONS} new questions]", flush=True)
        with open("data/datasets/generated_extra_v2.json", "w") as f:
            json.dump(new_qs, f, indent=2)

print(f"\nGenerated {generated_count} new questions")

# Merge with existing
existing_ids = set(q["id"] for q in existing)
all_qs = list(existing)
for q in new_qs:
    if q["id"] not in existing_ids:
        all_qs.append(q)
        existing_ids.add(q["id"])

print(f"Total benchmark: {len(all_qs)}")

with open("data/datasets/benchmark_final_v2.json", "w") as f:
    json.dump(all_qs, f, indent=2)
print("Saved benchmark_final_v2.json")

y = sum(1 for q in all_qs if str(q.get("answer","")).lower().startswith("yes"))
n = sum(1 for q in all_qs if str(q.get("answer","")).lower().startswith("no"))
print(f"Yes={y}, No={n}")