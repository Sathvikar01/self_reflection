"""Investigate CoT responses to understand the unknown rate."""
import json, re, requests, time

MODEL = "meta/llama-3.1-405b-instruct"
API_KEY = "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"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

with open("data/datasets/benchmark_final_v2.json") as f:
    questions = json.load(f)
q = questions[0]

# Test CoT response
resp = requests.post(
    "https://integrate.api.nvidia.com/v1/chat/completions",
    headers=HEADERS,
    json={
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Think step by step, then answer YES or NO."},
            {"role": "user", "content": f"Question: {q['question']}\n\nThink step by step, then answer YES or NO."}
        ],
        "temperature": 0.3,
        "max_tokens": 250
    },
    timeout=60
).json()
text = resp["choices"][0]["message"]["content"]
print("Full response:")
print(text)
print()
print("Last 100 chars:", text[-100:])

# Try improved extraction
def extract_improved(text):
    t = text.lower().strip()
    # Check first 50 chars
    if t[:50].startswith('yes'): return 'yes'
    if t[:50].startswith('no'): return 'no'
    # Check last 200 chars for answer
    last200 = t[-200:]
    # Look for "answer: yes" or "answer: no" patterns
    m = re.search(r'answer[:\s]+(yes|no)', last200)
    if m: return m.group(1)
    # Look for standalone yes/no at word boundaries
    m_yes = re.search(r'\byes\b', last200)
    m_no = re.search(r'\bno\b', last200)
    if m_yes and not m_no: return 'yes'
    if m_no and not m_yes: return 'no'
    # If both found, take the last one
    if m_yes and m_no:
        return last200[max(m_yes.start(), m_no.start()):][:2]
    # Check entire response for final answer pattern
    m = re.search(r'(?:final answer|answer|conclusion)[:\s]+(yes|no)', t)
    if m: return m.group(1)
    return 'unknown'

print("Extracted:", extract_improved(text))