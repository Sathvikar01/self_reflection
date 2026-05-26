# Knowledge-Augmented Reasoning: Why Naive RAG Fails and Structured Prompting Succeeds

## Overview

How should knowledge be injected into LLM prompts for knowledge-sensitive question answering? This project empirically compares five prompting strategies on multi-hop yes/no questions using Llama-3.1-405B-Instruct. The central finding: naive retrieval-augmented generation (RAG) and Chain-of-Thought both *hurt* performance, while a structured two-step prompting approach (relevance check then fact application) is the only method that improves over zero-shot.

## Key Findings

1. **Naive RAG hurts performance.** Placing KB facts directly into the context without structure reduces accuracy by 7.3 percentage points versus zero-shot.
2. **Chain-of-Thought causes large degradation.** CoT drops accuracy by 24.4pp (p=0.0098, McNemar's test), turning straightforward questions into non-committal answers.
3. **Self-Consistency alone cannot fix knowledge gaps.** Majority voting across samples shares the same knowledge deficit; accuracy drops 4.9pp.
4. **Step 1/Step 2 structured prompting is the only method that improves over zero-shot.** Explicit relevance checking before fact application yields +2.4pp (42/50 vs 41/50).

## Project Structure

```
src/                  Core library
  generator/          LLM clients (NVIDIA NIM, async, mock) and prompts
  evaluator/          PRM scoring, answer extraction
  orchestration/      Pipeline implementations (baseline, self-reflection, adaptive, RL)
  rl_controller/      MCTS, policy learning, value network, DPO trainer
  knowledge/          Knowledge base retriever
  utils/              Metrics, unified answer extractor, complexity analysis
scripts/              Benchmark runners and analysis scripts
tests/                Test suite (256 tests)
paper/                LaTeX paper draft (IEEE format)
data/                 Datasets and results
  datasets/           Benchmark datasets (StrategyQA, generated, complex reasoning)
benchmark_results/    Benchmark output files (JSON)
```

## Installation

```bash
git clone <repository-url>
cd self_reflection
pip install -e ".[dev]"
```

Set up API access (optional, mock mode available):

```bash
cp .env.example .env
# Edit .env and add your NVIDIA_API_KEY
```

## Running Benchmarks

```bash
# Mock mode (no API key needed, fast)
python scripts/run_benchmark.py --mock --num-questions 10

# Full benchmark with real LLM
python scripts/run_benchmark.py --dataset data/datasets/benchmark_final_v2.json

# With NVIDIA NIM API
export NVIDIA_API_KEY=your_key
python scripts/run_benchmark.py --dataset data/datasets/benchmark_final_v2.json
```

## Running Tests

```bash
pytest tests/ -v
```

## Methodology

- All five methods evaluated on the same set of questions
- `UnifiedAnswerExtractor` ensures fair answer parsing across all methods (yes/no extraction from free-form text)
- McNemar's test for paired statistical significance
- 95% Wilson confidence intervals for accuracy estimates
- Results are from actual LLM inference via NVIDIA NIM API (no simulated data)

## Results

Results from Llama-3.1-405B-Instruct on 50 multi-hop yes/no questions:

| Method | Accuracy | Correct | vs Zero-Shot | p-value |
|--------|----------|---------|--------------|---------|
| Zero-Shot | 82.0% | 41/50 | -- | -- |
| **KB+SC+Step 1/2** | **84.0%** | **42/50** | **+2.4pp** | -- |
| SC-only | 78.0% | 39/50 | -4.0pp | -- |
| RAG | 76.0% | 38/50 | -6.0pp | -- |
| CoT | 62.0% | 31/50 | -20.0pp | 0.0098 |

Source: `benchmark_results/comprehensive_summary.json`

Additional result on StrategyQA (56 questions, 405B): baseline 57.1% vs structured self-reflection 78.6% (p=0.049). Source: `benchmark_results/benchmark_405b.json`.

## Paper

- `paper/self_reflection_paper.tex` — IEEE conference paper (IEEEtran format)
- `paper/references.bib` — Bibliography

## Citation

```bibtex
@article{self_reflection2025,
  title={Knowledge-Augmented Reasoning: Why Naive RAG Fails and Structured Prompting Succeeds},
  year={2025}
}
```

## License

See repository for license details.
