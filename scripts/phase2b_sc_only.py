"""SC-only runner (Method 3) - run separately or in parallel with phase2a and phase2c"""
import sys
sys.path.insert(0, '.')
from phase2a_zeroshot_cot_rag import run_sc_only, load_questions

if __name__ == "__main__":
    questions = load_questions(0, 220)
    print(f"SC-only: {len(questions)} questions", flush=True)
    run_sc_only(questions)