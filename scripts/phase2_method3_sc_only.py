"""Method 3: Vanilla Self-Consistency WITHOUT KB (Wang et al., 2022)"""
import sys
sys.path.insert(0, '.')
from phase2_shared import *

class MethodSCOnly:
    name = "sc_only"
    def run(self, question, delay=DELAY):
        votes = []
        for i in range(5):
            time.sleep(delay)
            try:
                resp = call_api([
                    {"role": "system", "content": SYSTEM_COT},
                    {"role": "user", "content": f"Question: {question}\n\nThink step by step, then answer YES or NO."}
                ], max_tokens=300, temp=0.4)
                votes.append(extract_yesno(resp))
            except Exception as e:
                print(f"\n    [path {i} error: {e}]", flush=True)
                votes.append('unknown')
            if i < 4:
                time.sleep(1)
        yes_v = sum(1 for v in votes if v == 'yes')
        no_v = sum(1 for v in votes if v == 'no')
        answer = 'yes' if yes_v > no_v else 'no' if no_v > yes_v else votes[0]
        return answer, {'votes': votes}

if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 220
    questions = load_questions(start, end)
    print(f"SC-only: questions {start}-{end} ({len(questions)} total)", flush=True)
    m = MethodSCOnly()
    run_benchmark(m, questions, "sc_only", start_idx=start)