"""
Generate multi-step reasoning questions (StrategyQA-style) using NIM API.

These questions require reasoning chains, not just direct KB lookup.
"""
import os
import sys
import json
import time
import requests

# Fix encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

API_KEY = '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'
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def call_model(model: str, messages: list, temperature: float = 0.7, max_tokens: int = 500) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 0.95
    }
    try:
        resp = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers=HEADERS,
            json=payload,
            timeout=60
        )
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"API error: {e}")
        return ""


def generate_reasoning_question(category: str, difficulty: str) -> dict:
    """Generate a multi-step reasoning question."""

    system_prompt = """You are an expert at creating multi-step reasoning questions that require chains of thought.
Each question should require 2-3 logical steps to answer.
Do NOT create questions that can be answered by simple fact recall.
Format: Output only valid JSON with fields: question, answer, reasoning_steps (array of 2-3 steps), category"""

    user_prompt = f"""Create a StrategyQA-style yes/no question about {category} at {difficulty} difficulty.

Requirements:
- Must require MULTI-STEP REASONING (not just "Can X?" facts)
- Should have 2-3 reasoning steps to reach answer
- About half yes, half no answers
- No direct KB lookup questions

Example multi-step questions:
- "If John eats only potatoes for a month, will he survive?" (reasoning: potatoes have carbs+vitamin C+calories, historically Irish survived, but lacks complete nutrition)
- "Can a person born in France who never learns to swim still become an Olympic diver?" (reasoning: need training, no swimming=no training, impossible)

Generate 1 question now. JSON only:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = call_model("nvidia/nemotron-3-nano-30b-a3b", messages, temperature=0.8, max_tokens=800)
    if not response:
        return None

    try:
        # Extract JSON
        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        data = json.loads(text)

        if "question" in data and "answer" in data:
            return {
                "id": f"reasoning_{int(time.time())}_{len(questions)}",
                "question": data["question"],
                "answer": data["answer"].lower().strip(),
                "reasoning_steps": data.get("reasoning_steps", []),
                "category": category,
                "type": "multi_step_reasoning",
                "difficulty": difficulty,
                "source": "nim_api_generated"
            }
    except Exception as e:
        print(f"Parse error: {e}")

    return None


# Define categories and generate
CATEGORIES = [
    ("causal_reasoning", "Given A causes B, and B causes C, does A cause C?"),
    ("temporal_reasoning", "If X happened before Y, and Y before Z, what is the order?"),
    ("conditional_reasoning", "If P implies Q, and P is false, what can we conclude about Q?"),
    ("comparative_reasoning", "If A is bigger than B, and B is bigger than C, which is smallest?"),
    ("negative_reasoning", "What cannot be true if X is true and X contradicts Y?"),
    ("chained_effects", "If A increases B, and B decreases C, what happens to C when A increases?"),
    ("constraint_satisfaction", "If X must be different from Y, and Y must be different from Z, can X=Z?"),
    ("implication_chains", "If all S are P and all P are M, are all S M?"),
]

questions = []
DIFFICULTIES = ["easy", "medium", "hard"]

print("=" * 60)
print("Generating multi-step reasoning questions")
print("=" * 60)

for cat, desc in CATEGORIES:
    for diff in DIFFICULTIES:
        print(f"\nGenerating {diff} {cat} question...")
        q = generate_reasoning_question(cat, diff)
        if q:
            questions.append(q)
            print(f"  Q: {q['question'][:60]}...")
            print(f"  A: {q['answer']}")
        time.sleep(0.5)

        if len(questions) >= 40:
            break
    if len(questions) >= 40:
        break

# Save
output_path = "data/datasets/multi_step_reasoning.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(questions, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"Generated {len(questions)} multi-step reasoning questions")
print(f"Saved to: {output_path}")

# Show distribution
answers = {"yes": 0, "no": 0}
for q in questions:
    a = q["answer"].lower()
    if a in answers:
        answers[a] += 1

print(f"Answer distribution: {answers}")
print(f"Categories: {set(q['category'] for q in questions)}")