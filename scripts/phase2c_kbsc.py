"""KB+SC runner (Method 5) - run separately or in parallel"""
import sys
sys.path.insert(0, '.')
from phase2a_zeroshot_cot_rag import run_kbsc, load_questions

if __name__ == "__main__":
    questions = load_questions(0, 220)
    print(f"KB+SC: {len(questions)} questions", flush=True)
    run_kbsc(questions)