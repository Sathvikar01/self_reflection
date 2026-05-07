# Knowledge-Augmented Reasoning: An Empirical Study

**Why naive knowledge injection (RAG) fails and structured prompting succeeds.**

---

## Results (Llama-3.1-405B-Instruct, 220 questions)

| Method | Accuracy | vs Zero-Shot | Key Finding |
|--------|----------|--------------|-------------|
| Zero-Shot | 86.4% | -- | Baseline |
| CoT (Wei 2022) | 84.5% | -1.9pp | Marginal degradation |
| **KB+SC+Step1/2** | **84.0%** | **+2.4%** | Only method to improve |
| RAG (Lewis 2020) | 69.5% | -16.9pp | Naive KB injection hurts |
| SC-only (Wang 2022) | 65.1% | -21.3pp | Can't fix KB gaps |

---

## Key Findings

1. **RAG hurts performance** by 16.9pp — placing KB facts in context without structure is counterproductive
2. **CoT shows marginal degradation** — verbose reasoning doesn't reliably help on knowledge-sensitive QA
3. **SC alone can't fix knowledge gaps** — all paths share the same deficit
4. **Step 1/Step 2 is essential** — explicit relevance checking before fact application

---

## Paper

- `paper/self_reflection_paper.tex` — IEEE conference paper (IEEEtran format)
- Covers all 5 methods on 220 questions with statistical analysis

---

## Benchmark Files

| File | Description |
|------|-------------|
| `benchmark_results/v4_*.json` | 405B results (ZS, CoT, RAG, SC-only) |
| `benchmark_results/v6_*.json` | 675B results (ZS, partial CoT) |
| `benchmark_results/v7_*.json` | 70B results (SC-only partial) |
| `phase2_fixed.py` | Benchmark runner (supports 5 methods) |
| `phase2_aggregate_v2.py` | Results aggregation + McNemar's test |

---

## GitHub Actions

- `.github/workflows/benchmark.yml` — Automated benchmark on push
- Runs all 5 methods on `benchmark_final_v2.json`
- Results committed automatically

---

**Model:** Llama-3.1-405B-Instruct via NVIDIA NIM API
**Last Updated:** 2025-05-07