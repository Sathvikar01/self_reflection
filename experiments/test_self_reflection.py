"""Test self-reflection pipeline on simple questions."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestration.self_reflection_pipeline import SelfReflectionPipeline, SelfReflectionConfig

# Test on simple but tricky questions
TEST_QUESTIONS = [
    {
        "id": "tomato",
        "question": "Is a tomato a fruit?",
        "answer": "yes"
    },
    {
        "id": "diamond",
        "question": "Can you light a diamond on fire?",
        "answer": "yes"  # Diamonds CAN burn at high temperatures!
    },
    {
        "id": "paper",
        "question": "Can you fold a piece of paper more than 7 times?",
        "answer": "yes"  # With long enough paper or thin enough paper, it's possible
    },
    {
        "id": "earth",
        "question": "Is the Earth flat?",
        "answer": "no"
    },
    {
        "id": "spider",
        "question": "Do spiders have bones?",
        "answer": "no"
    },
]

def main():
    print("=" * 70)
    print("SELF-REFLECTION PIPELINE TEST")
    print("=" * 70)
    print("\nThis pipeline makes the LLM ACTUALLY reflect on its own reasoning,")
    print("find flaws, and correct itself before giving a final answer.")
    print("=" * 70)
    
    api_key = os.getenv("NVIDIA_API_KEY")
    config = SelfReflectionConfig(
        max_iterations=6,
        min_reasoning_steps=3,
        reflection_depth=2,  # Reflect twice!
        temperature_reason=0.7,
        temperature_reflect=0.3,
        force_reflection=True,
    )
    
    pipeline = SelfReflectionPipeline(
        api_key=api_key,
        config=config,
        results_dir="data/results",
    )
    
    results = pipeline.solve_batch(TEST_QUESTIONS)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    correct_count = sum(1 for r in results if r.correct)
    print(f"\nAccuracy: {correct_count}/{len(results)} = {correct_count/len(results):.1%}")
    
    print("\nResults by question:")
    for r in results:
        status = "✓ CORRECT" if r.correct else "✗ WRONG"
        print(f"  {r.problem_id}: {status}")
        print(f"    Answer: {r.final_answer}")
        print(f"    Reflections: {len(r.reflections)}, Corrections: {len(r.corrections)}")
    
    pipeline.close()
    
    print("\n" + "=" * 70)
    print("SELF-REFLECTION DEMONSTRATES:")
    print("  1. LLM generates initial reasoning")
    print("  2. LLM reflects on its OWN reasoning (self-critique)")
    print("  3. LLM finds issues and applies corrections")
    print("  4. LLM verifies before final answer")
    print("=" * 70)

if __name__ == "__main__":
    main()
