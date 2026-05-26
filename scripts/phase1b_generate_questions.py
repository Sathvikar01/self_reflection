"""Phase 1b: Generate additional questions to reach ~300 total via NIM API."""
import json
import time
import requests
import re
import os
from dotenv import load_dotenv
load_dotenv()

MODEL = "meta/llama-3.1-405b-instruct"
API_KEY = os.getenv("NVIDIA_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# Load existing to avoid duplicates
with open("data/datasets/benchmark_yesno.json") as f:
    existing = json.load(f)

existing_qs = set(q["question"].lower().strip() for q in existing)
print(f"Existing: {len(existing)} yes/no questions")

CATEGORIES = [
    "causal_chain", "temporal_sequence", "conditional_logic",
    "comparative_inference", "negation_reasoning", "abductive_reasoning",
    "counterfactual", "analogy_reasoning", "probability_reasoning",
    "mathematical_reasoning", "spatial_reasoning"
]

def call_model(messages, max_tokens=300):
    resp = requests.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        headers=HEADERS,
        json={"model": MODEL, "messages": messages, "temperature": 0.8, "max_tokens": max_tokens},
        timeout=120
    )
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"]
    raise Exception(f"API error {resp.status_code}: {resp.text}")

def extract_json(text):
    m = re.search(r"\{[^}]+\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    return None

NEW_QUESTIONS = []
TARGET = 120  # Generate 120 new questions

system_prompt = """You are an expert at generating complex multi-step reasoning questions.
Create questions in the style of StrategyQA — multi-step yes/no questions requiring implicit reasoning.
Each question should require 2-4 reasoning steps and involve counterintuitive knowledge.

Output JSON with exactly this format:
{"question": "...", "answer": "yes" or "no", "reasoning_steps": ["step1", "step2", "step3"], "category": "..."}"""

for i in range(TARGET):
    cat = CATEGORIES[i % len(CATEGORIES)]
    print(f"[{i+1}/{TARGET}] {cat}...", end=" ", flush=True)

    # Pick 5 examples of that category for few-shot
    few_shot = """Example outputs:
{"question": "If increasing temperature accelerates reactions, and faster reactions produce more heat, and excessive heat melts sensors, would sensors in a hot environment eventually fail?", "answer": "yes", "reasoning_steps": ["High temp accelerates reactions", "Accelerated reactions produce more heat", "More heat causes sensor melting", "Therefore sensors will eventually fail"], "category": "causal_chain"}
{"question": "If all mammals breathe air and whales are mammals, can whales survive on land?", "answer": "no", "reasoning_steps": ["Whales are mammals", "Mammals need air to breathe", "Whales breathe air but cannot move on land", "Therefore whales cannot survive on land"], "category": "causal_chain"}"""

    time.sleep(2)

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate a {cat} question. {few_shot}\n\nOutput JSON:"}
        ]
        resp = call_model(messages)
        data = extract_json(resp)

        if data and "question" in data and "answer" in data:
            q_text = data["question"].lower().strip()

            # Deduplicate
            if q_text not in existing_qs:
                item = {
                    "id": f"generated_{i}",
                    "question": data["question"],
                    "answer": data["answer"].lower().strip()[:3],
                    "reasoning_steps": data.get("reasoning_steps", []),
                    "category": cat,
                    "source": "api_generated"
                }
                NEW_QUESTIONS.append(item)
                existing_qs.add(q_text)
                print(f"OK ({len(NEW_QUESTIONS)} new total)")
            else:
                print("DUPLICATE")
        else:
            print(f"PARSE ERROR")
    except Exception as e:
        print(f"ERROR: {e}")

    # Checkpoint every 20
    if (i + 1) % 20 == 0:
        with open("data/datasets/generated_extra.json", "w") as f:
            json.dump(NEW_QUESTIONS, f, indent=2)
        print(f"  Checkpoint saved: {len(NEW_QUESTIONS)} new questions")

print(f"\nGenerated {len(NEW_QUESTIONS)} new questions")

# Merge with existing
all_yesno = existing + NEW_QUESTIONS
with open("data/datasets/benchmark_yesno_v2.json", "w") as f:
    json.dump(all_yesno, f, indent=2)
print(f"Total yes/no: {len(all_yesno)}")

# Stats
cats = {}
for q in all_yesno:
    cat = q.get("category", "unknown")
    cats[cat] = cats.get(cat, 0) + 1
print(f"Categories: {cats}")

# Answers
yes = sum(1 for q in all_yesno if q["answer"].startswith("yes"))
no = sum(1 for q in all_yesno if q["answer"].startswith("no"))
print(f"Yes: {yes}, No: {no}")