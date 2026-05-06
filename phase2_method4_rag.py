"""Method 4: Simple RAG (Lewis et al., 2020) - KB in system prompt, no Step1/Step2"""
import sys
sys.path.insert(0, '.')
from phase2_shared import *

class MethodRAG:
    name = "rag"
    def run(self, question, delay=DELAY):
        time.sleep(delay)
        resp = call_api([
            {"role": "system", "content": SYSTEM_RAG},
            {"role": "user", "content": f"Question: {question}\n\nRefer to the knowledge base above and answer YES or NO."}
        ], max_tokens=300)
        return extract_yesno(resp), {"response": resp}

if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 220
    questions = load_questions(start, end)
    print(f"RAG: questions {start}-{end} ({len(questions)} total)", flush=True)
    m = MethodRAG()
    run_benchmark(m, questions, "rag", start_idx=start)