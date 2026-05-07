# Self-Reflection Pipeline - Status (Updated 2025-05-01)

## Status: PHASE 3-4 COMPLETE

## Goal
Systematic empirical evaluation of 5 reasoning pipelines on 405B model.

## Results (50 questions, Llama-3.1-405B-Instruct)

| Method | Accuracy | vs Zero-Shot | Significant? |
|--------|----------|--------------|--------------|
| Zero-Shot | 82.0% | -- | -- |
| **KB+SC+Step1/2 (Ours)** | **84.0%** | **+2.4%** | No (p=1.0) |
| SC-only (Wang 2022) | 78.0% | -4.9% | No |
| RAG (Lewis 2020) | 76.0% | -7.3% | No |
| CoT (Wei 2022) | 62.0% | -24.4% | **Yes, p=0.0098** |

## Key Findings

1. **CoT causes non-committal answers**: 405B model fails to commit to YES/NO when asked to think step by step (p=0.0098)
2. **Naive KB injection hurts**: RAG (76.0%) < Zero-shot (82.0%) by 7.3%
3. **SC alone cannot fix KB gaps**: SC-only (78.0%) < Zero-shot (82.0%)
4. **Step 1/Step 2 is essential**: Only KB+SC (84.0%) improves over zero-shot
5. **Knowledge accessibility is the bottleneck**, not reasoning ability

## Documentation Updates (2025-05-01)

- [x] README.md — Updated with 5-method comparison results
- [x] paper/self_reflection_paper.tex — Rewritten as failure analysis / empirical evaluation paper
- [x] tasks/lessons.md — Updated with comprehensive benchmark findings
- [x] tasks/todo.md — This file
- [x] Session log — Updated with Phase 2-4 results

## Benchmark Files

- `benchmark_results/comp_zeroshot_s0_e50.json` — 50 results
- `benchmark_results/comp_cot_s0_e50.json` — 50 results
- `benchmark_results/comp_sc_only_s0_e50.json` — 50 results
- `benchmark_results/comp_rag_s0_e50.json` — 50 results
- `benchmark_results/comp_kb_sc_s0_e50.json` — 50 results
- `benchmark_results/comprehensive_summary.json` — Summary JSON
- `phase2_aggregate_results.py` — Aggregation + McNemar analysis

## Phase Scripts

- `phase1_merge_datasets.py` — Merged 220 unique yes/no questions
- `phase1b_generate_questions.py` — API question generation
- `phase2_method{1-5}_*.py` — Individual method runners
- `phase2_aggregate_results.py` — Results aggregation