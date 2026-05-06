"""Method 5: KB + 5-path Self-Consistency + Step1/Step2 (Our Method)"""
import sys
sys.path.insert(0, '.')
from phase2_shared import *

class MethodKB_SC:
    name = "kb_sc"
    def run(self, question, delay=DELAY):
        votes = []
        for i in range(5):
            time.sleep(delay)
            try:
                resp = call_api([
                    {"role": "system", "content": f"""You are a helpful assistant. Use the following knowledge base when answering.

{KNOWLEDGE_BASE}

Answer yes/no questions with ONLY yes or no."""},
                    {"role": "user", "content": f"""Question: {question}

Step 1: Does any fact in the knowledge base directly relate to this question? If yes, state it.
Step 2: Based on the knowledge base, answer YES or NO.

Reason step by step, then give your final answer."""}
                ], max_tokens=400, temp=0.4)
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
    print(f"KB+SC: questions {start}-{end} ({len(questions)} total)", flush=True)
    m = MethodKB_SC()
    run_benchmark(m, questions, "kb_sc", start_idx=start)