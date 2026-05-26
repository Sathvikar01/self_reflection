# Benchmark Report

**Date:** 2026-05-27 01:30:44
**Model:** meta/llama-3.1-8b-instruct
**Dataset:** data/datasets/benchmark_dataset.json
**Questions:** 5
**SC Samples:** 5

## Results Summary

| Method | Accuracy | 95% CI | Correct | Latency (ms) |
|--------|----------|--------|---------|--------------|
| Zero-Shot | 40.0% | [11.8%, 76.9%] | 2/5 | 215 ± 33 |
| Chain-of-Thought | 0.0% | [0.0%, 43.4%] | 0/5 | 217 ± 57 |
| Self-Consistency | 0.0% | [0.0%, 43.4%] | 0/5 | 1024 ± 96 |
| RAG | 0.0% | [0.0%, 43.4%] | 0/5 | 190 ± 57 |
| KB+SC+Step1/2 | 20.0% | [3.6%, 62.4%] | 1/5 | 1912 ± 63 |

## Statistical Tests (McNemar's)

| Comparison | χ² | p-value | Significant (α=0.05) |
|------------|-----|---------|---------------------|
| zero_shot_vs_chain_of_thought | 0.500 | 0.4795 | No |
| zero_shot_vs_self_consistency | 0.500 | 0.4795 | No |
| zero_shot_vs_rag | 0.500 | 0.4795 | No |
| zero_shot_vs_kb_sc_step | 0.000 | 1.0000 | No |
| chain_of_thought_vs_self_consistency | 0.000 | 1.0000 | No |
| chain_of_thought_vs_rag | 0.000 | 1.0000 | No |
| chain_of_thought_vs_kb_sc_step | 0.000 | 1.0000 | No |
| self_consistency_vs_rag | 0.000 | 1.0000 | No |
| self_consistency_vs_kb_sc_step | 0.000 | 1.0000 | No |
| rag_vs_kb_sc_step | 0.000 | 1.0000 | No |

## Methodology

- **Zero-Shot:** Direct answer without reasoning chain
- **Chain-of-Thought:** Step-by-step reasoning before answering
- **Self-Consistency:** 5 samples with majority voting (temp=0.8)
- **RAG:** Knowledge retrieval + CoT prompt
- **KB+SC+Step1/2:** Structured 2-step prompting with knowledge + SC

All methods use the same UnifiedAnswerExtractor for fair evaluation.
Confidence intervals use Wilson score interval.

## Per-Question Results

### merged_0: Do hamsters provide food for any animals?
**Ground Truth:** yes

| Method | Extracted | Correct |
|--------|-----------|---------|
| Zero-Shot | no | ✗ |
| Chain-of-Thought | This step follows from the previous observation about the relationship | ✗ |
| Self-Consistency | n scenario. | ✗ |
| RAG | This step follows from the previous observation about the relationship | ✗ |
| KB+SC+Step1/2 | ning follows logically from the premises. | ✗ |

### merged_1: Could a person survive falling from the Empire State Building?
**Ground Truth:** no

| Method | Extracted | Correct |
|--------|-----------|---------|
| Zero-Shot | ning follows logically from the premises. | ✗ |
| Chain-of-Thought | This step follows from the previous observation about the relationship | ✗ |
| Self-Consistency | First, I need to identify the key components of this problem | ✗ |
| RAG | First, I need to identify the key components of this problem | ✗ |
| KB+SC+Step1/2 | The answer depends on additional context not provided | ✓ |

### merged_2: Is the sun brighter than a light bulb?
**Ground Truth:** yes

| Method | Extracted | Correct |
|--------|-----------|---------|
| Zero-Shot | ning follows logically from the premises. | ✗ |
| Chain-of-Thought | First, I need to identify the key components of this problem | ✗ |
| Self-Consistency | against the constraints. | ✗ |
| RAG | Looking at the evidence, we can conclude that the relationship holds | ✗ |
| KB+SC+Step1/2 | The answer depends on additional context not provided | ✗ |

### merged_3: Would a person born in 2000 be eligible to vote in the 2016 US election?
**Ground Truth:** no

| Method | Extracted | Correct |
|--------|-----------|---------|
| Zero-Shot | The answer depends on additional context not provided | ✓ |
| Chain-of-Thought | First, I need to identify the key components of this problem | ✗ |
| Self-Consistency | This step follows from the previous observation about the relationship | ✗ |
| RAG | n scenario. | ✗ |
| KB+SC+Step1/2 | n scenario. | ✗ |

### merged_4: Does hydrogen power work without oxygen?
**Ground Truth:** no

| Method | Extracted | Correct |
|--------|-----------|---------|
| Zero-Shot | The answer depends on additional context not provided | ✓ |
| Chain-of-Thought | This step follows from the previous observation about the relationship | ✗ |
| Self-Consistency | This step follows from the previous observation about the relationship | ✗ |
| RAG | Looking at the evidence, we can conclude that the relationship holds | ✗ |
| KB+SC+Step1/2 | against the constraints. | ✗ |
