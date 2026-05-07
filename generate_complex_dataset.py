"""Generate Complex Multi-Step Reasoning Questions."""

import os
import json
import time
import requests
import re

API_KEY = "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"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
MODEL = "meta/llama-3.1-405b-instruct"

all_questions = []  # Global tracking


def generate_handcrafted_questions():
    """Generate hand-crafted complex questions (guaranteed quality)."""
    return [
        # Causal Chains
        {
            "id": "cc_001",
            "question": "If increasing temperature accelerates chemical reactions, and faster reactions produce more heat, and excessive heat melts sensors, would sensors in a hot environment with accelerated reactions eventually fail?",
            "answer": "yes",
            "reasoning_steps": ["High temp accelerates reactions", "Accelerated reactions produce more heat", "More heat causes sensor melting", "Therefore sensors will eventually fail"],
            "category": "causal_chain", "difficulty": "hard", "type": "multi_step_reasoning"
        },
        {
            "id": "cc_002",
            "question": "If learning requires attention, and attention decreases with fatigue, and experts have developed automatic skills that need less attention, would experts learn a new domain faster than novices despite fatigue?",
            "answer": "yes",
            "reasoning_steps": ["Learning requires attention", "Attention decreases with fatigue", "Experts have automatic skills needing less attention", "Therefore experts can continue learning despite fatigue"],
            "category": "causal_chain", "difficulty": "hard", "type": "multi_step_reasoning"
        },

        # Temporal Sequence
        {
            "id": "ts_001",
            "question": "If event A precedes B, and B enables C, but D prevents C, and D occurs after A but before B's completion, can C ever occur?",
            "answer": "no",
            "reasoning_steps": ["D occurs after A but before B completes", "D prevents C from occurring", "C requires B to complete first", "Since D occurs before B completes, C cannot occur"],
            "category": "temporal_sequence", "difficulty": "hard", "type": "multi_step_reasoning"
        },
        {
            "id": "ts_002",
            "question": "If the alarm rings before snooze is pressed, snooze resets alarm for 9 minutes, and meeting starts in 10 minutes, would someone using snooze miss the meeting?",
            "answer": "no",
            "reasoning_steps": ["Alarm rings, snooze pressed", "Snooze resets for 9 minutes", "Meeting starts in 10 minutes", "9 min < 10 min, so alarm would wake before meeting"],
            "category": "temporal_sequence", "difficulty": "medium", "type": "multi_step_reasoning"
        },

        # Conditional Logic
        {
            "id": "cl_001",
            "question": "If all efficient processes reduce costs, and reducing costs is necessary for sustainability, and unsustainable practices harm the environment, would an efficient process that harms the environment be a contradiction?",
            "answer": "yes",
            "reasoning_steps": ["All efficient processes reduce costs", "Sustainability requires reduced costs", "Unsustainable practices harm environment", "Efficient + harmful = contradiction"],
            "category": "conditional_logic", "difficulty": "hard", "type": "multi_step_reasoning"
        },
        {
            "id": "cl_002",
            "question": "If no mammals can fly, and bats are mammals, and flying squirrels are mammals, would it be correct to say that some mammals can fly?",
            "answer": "no",
            "reasoning_steps": ["Statement says no mammals can fly", "Bats and flying squirrels are mammals", "Despite names suggesting flight, they cannot truly fly", "Therefore statement is incorrect"],
            "category": "conditional_logic", "difficulty": "medium", "type": "multi_step_reasoning"
        },

        # Comparative Inference
        {
            "id": "ci_001",
            "question": "If company A's revenue is 3x company B's, company B's revenue is 2x company C's, and company C's profit margin is 2x company A's, would company A necessarily have the highest profit?",
            "answer": "no",
            "reasoning_steps": ["A=6C in revenue", "A margin = C margin / 2", "Higher revenue doesn't mean higher profit with lower margin", "A has lowest margin despite highest revenue"],
            "category": "comparative_inference", "difficulty": "hard", "type": "multi_step_reasoning"
        },

        # Negation Reasoning
        {
            "id": "nr_001",
            "question": "If Alice cannot be in two places at once, and Bob must be where Alice is, and Carol cannot be where Bob is, and Alice is in Room X, could Carol be in Room X?",
            "answer": "no",
            "reasoning_steps": ["Alice is in Room X", "Bob must be where Alice is", "So Bob is in Room X", "Carol cannot be where Bob is", "Therefore Carol cannot be in Room X"],
            "category": "negation_reasoning", "difficulty": "medium", "type": "multi_step_reasoning"
        },

        # Abductive Reasoning
        {
            "id": "ar_001",
            "question": "If the ground is wet, it rained recently, and the sprinkler was on timer, but it didn't rain according to weather records, what is the most likely explanation?",
            "answer": "yes",
            "reasoning_steps": ["Ground is wet (observation)", "Either rain or sprinkler", "Weather records say no rain", "Therefore sprinkler is most likely"],
            "category": "abductive_reasoning", "difficulty": "medium", "type": "multi_step_reasoning"
        },

        # Common Knowledge
        {
            "id": "ck_001",
            "question": "If Alice knows that Bob doesn't know the answer, and Alice knows the answer, and Bob asks Alice for help, would Alice be justified in giving the answer directly?",
            "answer": "no",
            "reasoning_steps": ["Alice knows Bob doesn't know", "Giving answer directly reveals Alice knew", "This violates information asymmetry", "Alice should not give answer directly"],
            "category": "common_knowledge", "difficulty": "hard", "type": "multi_step_reasoning"
        },

        # Math Word Problem
        {
            "id": "mw_001",
            "question": "If a store offers buy 2 get 1 free, each item costs $5, and you need 7 items, would paying $25 be sufficient?",
            "answer": "yes",
            "reasoning_steps": ["Buy 2 get 1 free = $10 for 3 items", "2 sets = $20 for 6 items", "Plus 1 more item = $5", "Total $25 for 7 items"],
            "category": "math_word_problem", "difficulty": "hard", "type": "multi_step_reasoning"
        },

        # Strategic Planning
        {
            "id": "sp_001",
            "question": "If completing task A unlocks task B, completing B unlocks C, but completing A consumes resources needed for D, and you must complete both C and D, should you start with A?",
            "answer": "no",
            "reasoning_steps": ["A unlocks B which unlocks C", "Starting A consumes resources for D", "Must complete C and D", "Alternative: Get resources for D first, then do A→B→C", "So should NOT start with A"],
            "category": "strategic_planning", "difficulty": "hard", "type": "multi_step_reasoning"
        },

        # Counterfactual
        {
            "id": "cf_001",
            "question": "If the experiment had used twice the concentration, would the reaction time necessarily be halved?",
            "answer": "no",
            "reasoning_steps": ["Doubling concentration doesn't guarantee linear rate change", "Reaction rates depend on multiple factors", "Chemical kinetics follow non-linear models", "Cannot assume linear relationship without more info"],
            "category": "counterfactual", "difficulty": "hard", "type": "multi_step_reasoning"
        },

        # Analogy Reasoning
        {
            "id": "an_001",
            "question": "If a surgeon must cut to heal, and cutting always causes pain, and healing requires cutting, is pain an unavoidable part of healing?",
            "answer": "yes",
            "reasoning_steps": ["Surgeon must cut to heal", "Cutting causes pain", "Healing requires cutting", "Therefore healing always involves pain"],
            "category": "analogy_reasoning", "difficulty": "medium", "type": "multi_step_reasoning"
        },

        # Probability Reasoning
        {
            "id": "pr_001",
            "question": "If it has rained 5 days in a row, and weather patterns show rain continues with 80% probability, is it guaranteed to rain tomorrow?",
            "answer": "no",
            "reasoning_steps": ["80% probability means 20% chance of no rain", "Past 5 days don't affect tomorrow's probability", "Probability is not certainty", "Even 80% doesn't guarantee rain"],
            "category": "probability_reasoning", "difficulty": "medium", "type": "multi_step_reasoning"
        }
    ]


def generate_api_questions():
    """Generate more questions via API."""
    global all_questions

    categories = ["causal_chain", "temporal_sequence", "conditional_logic", "comparative_inference"]
    difficulties = ["medium", "hard"]

    for cat in categories:
        for diff in difficulties:
            print(f"  {diff} {cat}...", end=" ", flush=True)
            time.sleep(1.6)

            messages = [
                {"role": "system", "content": "Generate one complex multi-step reasoning question. Output JSON with: question, answer (yes/no), reasoning_steps (array of 3+ steps)."},
                {"role": "user", "content": f"Create a {diff} {cat} question that requires 3+ reasoning steps. JSON only:"}
            ]

            try:
                resp = requests.post(
                    'https://integrate.api.nvidia.com/v1/chat/completions',
                    headers=HEADERS,
                    json={"model": MODEL, "messages": messages, "temperature": 0.8, "max_tokens": 400},
                    timeout=90
                )
                if resp.status_code == 200:
                    content = resp.json()['choices'][0]['message']['content']
                    m = re.search(r'\{[^}]+\}', content, re.DOTALL)
                    if m:
                        data = json.loads(m.group(0))
                        if 'question' in data:
                            q = {
                                "id": f"api_{len(all_questions)}",
                                "question": data['question'],
                                "answer": data.get('answer', 'no')[:3].lower(),
                                "ground_truth": data.get('answer', 'no')[:3].lower(),
                                "reasoning_steps": data.get('reasoning_steps', []),
                                "category": cat,
                                "difficulty": diff,
                                "type": "multi_step_reasoning",
                                "source": "api_generated"
                            }
                            all_questions.append(q)
                            print(f"OK ({len(all_questions)} total)")
                            continue
            except Exception as e:
                print(f"FAILED: {e}")
            print("FAILED")


def main():
    global all_questions

    print("=" * 80)
    print("GENERATING COMPLEX MULTI-STEP REASONING DATASET")
    print("=" * 80)

    # Hand-crafted questions
    all_questions = generate_handcrafted_questions()
    print(f"Generated {len(all_questions)} hand-crafted questions")

    # API-generated questions
    print("\nGenerating via API...")
    generate_api_questions()

    # Save
    output_path = "data/datasets/complex_reasoning.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_questions, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(all_questions)} questions to {output_path}")

    # Stats
    cats, diffs, answers = {}, {}, {"yes": 0, "no": 0}
    for q in all_questions:
        cats[q.get("category", "unk")] = cats.get(q.get("category", "unk"), 0) + 1
        diffs[q.get("difficulty", "unk")] = diffs.get(q.get("difficulty", "unk"), 0) + 1
        ans = q.get("answer", "")[:3].lower()
        if ans in answers:
            answers[ans] += 1

    print(f"\nCategories: {cats}")
    print(f"Difficulties: {diffs}")
    print(f"Answers: {answers}")

    return all_questions


if __name__ == "__main__":
    questions = main()