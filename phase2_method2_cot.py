"""Method 2: Chain-of-Thought (Wei et al., 2022)"""
import sys
sys.path.insert(0, '.')
from phase2_shared import *

class MethodCoT:
    name = "cot"
    def run(self, question, delay=DELAY):
        time.sleep(delay)
        resp = call_api([
            {"role": "system", "content": SYSTEM_COT},
            {"role": "user", "content": f"Question: {question}\n\nThink step by step, then answer YES or NO."}
        ], max_tokens=300)
        return extract_yesno(resp), {"response": resp}

if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 220
    questions = load_questions(start, end)
    print(f"CoT: questions {start}-{end} ({len(questions)} total)", flush=True)
    m = MethodCoT()
    run_benchmark(m, questions, "cot", start_idx=start)