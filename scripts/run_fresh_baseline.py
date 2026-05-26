import sys
import os
import json
import time

sys.path.insert(0, '.')
from iter6_pipeline import *

api_key = os.getenv('NVIDIA_API_KEY')
client = NVIDIANIMClient(api_key=api_key)
client._cache.clear()

pipeline = BaselinePipeline(client)

with open('data/datasets/strategyqa_full.json') as f:
    problems = json.load(f)

correct = 0
total = 0
for p in problems:
    r = pipeline.solve(p['question'], p['id'], p['answer'])
    if r.correct:
        correct += 1
    total += 1
    status = 'OK' if r.correct else 'WRONG'
    print(f'[{total}] {p["id"]}: {status} predicted={r.answer} expected={p["answer"]} ({r.latency_seconds:.1f}s)')

print(f'\nBaseline (no cache): {correct}/{total} = {correct/total:.1%}')
client.close()
