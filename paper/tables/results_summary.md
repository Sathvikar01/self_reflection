# Benchmark Results Summary

Generated: 2026-05-27 01:41:18

Results directory: `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results`

Files parsed: 59 | Skipped: 9 | Errors: 3


## Parse Errors

- `benchmark_results_20260527_004400.json: Expecting value: line 69 column 28 (char 1893)`
- `final_consolidated_benchmark_1775364480.json: 'int' object has no attribute 'items'`
- `final_consolidated_benchmark_1775366152.json: 'int' object has no attribute 'items'`


## Main Results

| Method | Accuracy | 95% CI | N | p-value (vs Zero-Shot) |
|--------|----------|--------|---|----------------------|
| Zero-Shot | 82.7% | [80.4%, 84.8%] | 1150 | --- |
| KB+SC | 80.6% | [69.1%, 88.6%] | 62 | 0.8026 |
| CoT | 80.0% | [76.5%, 83.0%] | 574 | 0.3487 |
| Self Reflection | 73.2% | [70.1%, 76.0%] | 879 | 1.0000 |
| RAG | 70.7% | [65.1%, 75.8%] | 270 | 0.0003 |
| Baseline | 62.3% | [59.3%, 65.3%] | 988 | 1.0000 |
| SC | 44.9% | [40.2%, 49.7%] | 419 | 0.0000 |
| Adaptive | 13.3% | [8.4%, 20.6%] | 120 | 1.0000 |
| Rl Based | 7.5% | [4.0%, 13.6%] | 120 | 1.0000 |

## Multi-Run Sources

Methods with results from multiple files (potential cross-validation):

- **Baseline**: `benchmark_405b.json`, `full_benchmark.json`, `iter6_benchmark_1777159471.json`, `iter6_benchmark_1777159949.json`, `iter6_benchmark_1777176301.json`, `iter6_benchmark_1777177188.json`, `iter6_benchmark_1777235571.json`, `iter6_benchmark_1777243389.json`, `iter6_benchmark_1777251640.json`, `iter6_benchmark_1777251803.json`, `iter6_benchmark_1777254861.json`, `iter6_benchmark_1777254993.json`, `iter6_benchmark_1777257901.json`, `iter6_benchmark_1777258868.json`, `iter6_benchmark_1777266709.json`, `iter6_benchmark_1777319007.json`, `iter6_benchmark_1777319829.json`, `iter6_benchmark_1777320832.json`, `iter6_benchmark_1777321180.json`, `iter6_benchmark_1777322876.json`, `iter6_benchmark_1777323285.json`, `iter6_benchmark_1777325924.json`, `iter6_benchmark_1777326299.json`, `iter6_benchmark_1777432488.json`, `real_benchmark_1776969333.json`, `real_benchmark_1776999299.json`, `real_benchmark_1777000380.json`, `real_benchmark_1777000666.json`, `real_benchmark_1777053764.json`, `real_benchmark_1777067226.json`, `real_benchmark_1777098379.json`, `real_benchmark_1777102041.json`, `real_llm_all_40_1775379195.json`, `real_llm_all_40_1775509964.json`, `real_llm_all_40_1775537986.json`
- **CoT**: `comp_cot_s0_e50.json`, `v4_cot.json`, `v5_cot.json`, `v6_cot.json`, `v8_cot.json`, `v9_cot.json`
- **KB+SC**: `comp_kb_sc_s0_e50.json`, `v7_kbsc.json`
- **RAG**: `comp_rag_s0_e50.json`, `v4_rag.json`
- **SC**: `comp_sc_only_s0_e50.json`, `v4_sc_only.json`, `v7_sc_only.json`
- **Zero-Shot**: `comp_zeroshot_s0_e50.json`, `v3_zeroshot.json`, `v5_zeroshot.json`, `v6_zeroshot.json`, `v8_zeroshot.json`, `v9_zeroshot.json`
- **Self Reflection**: `iter6_benchmark_1777159471.json`, `iter6_benchmark_1777159949.json`, `iter6_benchmark_1777176301.json`, `iter6_benchmark_1777177188.json`, `iter6_benchmark_1777251640.json`, `iter6_benchmark_1777251803.json`, `iter6_benchmark_1777254861.json`, `iter6_benchmark_1777254993.json`, `iter6_benchmark_1777257901.json`, `iter6_benchmark_1777258868.json`, `iter6_benchmark_1777319007.json`, `iter6_benchmark_1777319829.json`, `iter6_benchmark_1777320832.json`, `iter6_benchmark_1777321180.json`, `iter6_benchmark_1777322876.json`, `iter6_benchmark_1777323285.json`, `iter6_benchmark_1777325924.json`, `iter6_benchmark_1777326299.json`, `iter6_benchmark_1777432488.json`, `real_benchmark_1777001790.json`, `real_benchmark_1777055288.json`, `real_benchmark_1777066596.json`, `real_benchmark_1777069177.json`, `real_benchmark_1777087605.json`, `real_benchmark_1777098379.json`, `real_benchmark_1777102041.json`, `real_llm_all_40_1775379195.json`, `real_llm_all_40_1775509964.json`, `real_llm_all_40_1775537986.json`
- **Adaptive**: `real_llm_all_40_1775379195.json`, `real_llm_all_40_1775509964.json`, `real_llm_all_40_1775537986.json`
- **Rl Based**: `real_llm_all_40_1775379195.json`, `real_llm_all_40_1775509964.json`, `real_llm_all_40_1775537986.json`


## Per-Source Breakdown

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\benchmark_405b.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Baseline | 32 | 56 | 57.1% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\comp_cot_s0_e50.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| CoT | 31 | 50 | 62.0% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\comp_kb_sc_s0_e50.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| KB+SC | 42 | 50 | 84.0% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\comp_rag_s0_e50.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| RAG | 38 | 50 | 76.0% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\comp_sc_only_s0_e50.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| SC | 39 | 50 | 78.0% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\comp_zeroshot_s0_e50.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Zero-Shot | 41 | 50 | 82.0% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\full_benchmark.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Baseline | 10 | 20 | 50.0% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\iter6_benchmark_1777159471.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Baseline | 19 | 28 | 67.9% |
| Self Reflection | 27 | 28 | 96.4% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\iter6_benchmark_1777159949.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Baseline | 25 | 28 | 89.3% |
| Self Reflection | 28 | 28 | 100.0% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\iter6_benchmark_1777176301.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Baseline | 21 | 28 | 75.0% |
| Self Reflection | 28 | 28 | 100.0% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\iter6_benchmark_1777177188.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Baseline | 23 | 28 | 82.1% |
| Self Reflection | 25 | 28 | 89.3% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\iter6_benchmark_1777235571.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Baseline | 5 | 5 | 100.0% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\iter6_benchmark_1777243389.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Baseline | 12 | 14 | 85.7% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\iter6_benchmark_1777251640.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Baseline | 24 | 28 | 85.7% |
| Self Reflection | 23 | 28 | 82.1% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\iter6_benchmark_1777251803.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Baseline | 20 | 28 | 71.4% |
| Self Reflection | 22 | 28 | 78.6% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\iter6_benchmark_1777254861.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Baseline | 22 | 28 | 78.6% |
| Self Reflection | 18 | 28 | 64.3% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\iter6_benchmark_1777254993.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Baseline | 21 | 28 | 75.0% |
| Self Reflection | 20 | 28 | 71.4% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\iter6_benchmark_1777257901.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Baseline | 7 | 7 | 100.0% |
| Self Reflection | 7 | 7 | 100.0% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\iter6_benchmark_1777258868.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Baseline | 7 | 14 | 50.0% |
| Self Reflection | 11 | 14 | 78.6% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\iter6_benchmark_1777266709.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Baseline | 1 | 1 | 100.0% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\iter6_benchmark_1777319007.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Baseline | 21 | 28 | 75.0% |
| Self Reflection | 25 | 28 | 89.3% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\iter6_benchmark_1777319829.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Baseline | 25 | 28 | 89.3% |
| Self Reflection | 24 | 28 | 85.7% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\iter6_benchmark_1777320832.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Baseline | 21 | 28 | 75.0% |
| Self Reflection | 24 | 28 | 85.7% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\iter6_benchmark_1777321180.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Baseline | 24 | 28 | 85.7% |
| Self Reflection | 22 | 28 | 78.6% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\iter6_benchmark_1777322876.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Baseline | 20 | 28 | 71.4% |
| Self Reflection | 24 | 28 | 85.7% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\iter6_benchmark_1777323285.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Baseline | 22 | 28 | 78.6% |
| Self Reflection | 24 | 28 | 85.7% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\iter6_benchmark_1777325924.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Baseline | 23 | 28 | 82.1% |
| Self Reflection | 28 | 28 | 100.0% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\iter6_benchmark_1777326299.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Baseline | 22 | 28 | 78.6% |
| Self Reflection | 28 | 28 | 100.0% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\iter6_benchmark_1777432488.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Baseline | 8 | 10 | 80.0% |
| Self Reflection | 10 | 10 | 100.0% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\real_benchmark_1776969333.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Baseline | 0 | 10 | 0.0% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\real_benchmark_1776999299.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Baseline | 0 | 56 | 0.0% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\real_benchmark_1777000380.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Baseline | 3 | 3 | 100.0% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\real_benchmark_1777000666.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Baseline | 39 | 56 | 69.6% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\real_benchmark_1777001790.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Self Reflection | 36 | 56 | 64.3% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\real_benchmark_1777053764.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Baseline | 44 | 56 | 78.6% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\real_benchmark_1777055288.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Self Reflection | 37 | 56 | 66.1% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\real_benchmark_1777066596.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Self Reflection | 18 | 28 | 64.3% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\real_benchmark_1777067226.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Baseline | 43 | 56 | 76.8% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\real_benchmark_1777069177.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Self Reflection | 39 | 56 | 69.6% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\real_benchmark_1777087605.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Self Reflection | 19 | 28 | 67.9% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\real_benchmark_1777098379.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Baseline | 19 | 28 | 67.9% |
| Self Reflection | 21 | 28 | 75.0% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\real_benchmark_1777102041.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Baseline | 21 | 28 | 75.0% |
| Self Reflection | 27 | 28 | 96.4% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\real_llm_all_40_1775379195.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Adaptive | 4 | 40 | 10.0% |
| Baseline | 0 | 40 | 0.0% |
| Rl Based | 0 | 40 | 0.0% |
| Self Reflection | 9 | 40 | 22.5% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\real_llm_all_40_1775509964.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Adaptive | 2 | 40 | 5.0% |
| Baseline | 7 | 40 | 17.5% |
| Rl Based | 6 | 40 | 15.0% |
| Self Reflection | 6 | 40 | 15.0% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\real_llm_all_40_1775537986.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Adaptive | 10 | 40 | 25.0% |
| Baseline | 5 | 40 | 12.5% |
| Rl Based | 3 | 40 | 7.5% |
| Self Reflection | 13 | 40 | 32.5% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\v3_zeroshot.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Zero-Shot | 190 | 220 | 86.4% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\v4_cot.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| CoT | 186 | 220 | 84.5% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\v4_rag.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| RAG | 153 | 220 | 69.5% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\v4_sc_only.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| SC | 52 | 220 | 23.6% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\v5_cot.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| CoT | 184 | 220 | 83.6% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\v5_zeroshot.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Zero-Shot | 183 | 220 | 83.2% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\v6_cot.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| CoT | 50 | 71 | 70.4% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\v6_zeroshot.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Zero-Shot | 177 | 220 | 80.5% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\v7_kbsc.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| KB+SC | 8 | 12 | 66.7% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\v7_sc_only.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| SC | 97 | 149 | 65.1% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\v8_cot.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| CoT | 4 | 5 | 80.0% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\v8_zeroshot.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Zero-Shot | 177 | 220 | 80.5% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\v9_cot.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| CoT | 4 | 8 | 50.0% |

### `C:\Users\arsat\OneDrive\Documents\Desktop\self_reflection\benchmark_results\v9_zeroshot.json`

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Zero-Shot | 183 | 220 | 83.2% |


## McNemar's Test (Pairwise)

| Method A | Method B | N (common) | b | c | chi2 | p-value | Sig (0.05) |
|----------|----------|------------|---|---|------|---------|------------|
| Adaptive | Baseline | 40 | 2 | 7 | 1.778 | 0.1824 | No |
| Adaptive | CoT | 0 | 0 | 0 | 0.000 | 1.0000 | No |
| Adaptive | KB+SC | 0 | 0 | 0 | 0.000 | 1.0000 | No |
| Adaptive | RAG | 0 | 0 | 0 | 0.000 | 1.0000 | No |
| Adaptive | Rl Based | 40 | 2 | 9 | 3.273 | 0.0704 | No |
| Adaptive | SC | 0 | 0 | 0 | 0.000 | 1.0000 | No |
| Adaptive | Self Reflection | 40 | 5 | 2 | 0.571 | 0.4497 | No |
| Adaptive | Zero-Shot | 0 | 0 | 0 | 0.000 | 1.0000 | No |
| Baseline | CoT | 0 | 0 | 0 | 0.000 | 1.0000 | No |
| Baseline | KB+SC | 0 | 0 | 0 | 0.000 | 1.0000 | No |
| Baseline | RAG | 0 | 0 | 0 | 0.000 | 1.0000 | No |
| Baseline | Rl Based | 40 | 2 | 4 | 0.167 | 0.6831 | No |
| Baseline | SC | 0 | 0 | 0 | 0.000 | 1.0000 | No |
| Baseline | Self Reflection | 96 | 18 | 10 | 1.750 | 0.1859 | No |
| Baseline | Zero-Shot | 0 | 0 | 0 | 0.000 | 1.0000 | No |
| CoT | KB+SC | 50 | 8 | 5 | 0.308 | 0.5791 | No |
| CoT | RAG | 220 | 24 | 47 | 6.817 | 0.0090 | Yes |
| CoT | Rl Based | 0 | 0 | 0 | 0.000 | 1.0000 | No |
| CoT | SC | 220 | 10 | 89 | 61.455 | 0.0000 | Yes |
| CoT | Self Reflection | 0 | 0 | 0 | 0.000 | 1.0000 | No |
| CoT | Zero-Shot | 220 | 24 | 17 | 0.878 | 0.3487 | No |
| KB+SC | RAG | 50 | 3 | 5 | 0.125 | 0.7237 | No |
| KB+SC | Rl Based | 0 | 0 | 0 | 0.000 | 1.0000 | No |
| KB+SC | SC | 50 | 5 | 15 | 4.050 | 0.0442 | Yes |
| KB+SC | Self Reflection | 0 | 0 | 0 | 0.000 | 1.0000 | No |
| KB+SC | Zero-Shot | 50 | 7 | 9 | 0.062 | 0.8026 | No |
| RAG | Rl Based | 0 | 0 | 0 | 0.000 | 1.0000 | No |
| RAG | SC | 220 | 25 | 81 | 28.538 | 0.0000 | Yes |
| RAG | Self Reflection | 0 | 0 | 0 | 0.000 | 1.0000 | No |
| RAG | Zero-Shot | 220 | 47 | 17 | 13.141 | 0.0003 | Yes |
| Rl Based | SC | 0 | 0 | 0 | 0.000 | 1.0000 | No |
| Rl Based | Self Reflection | 40 | 11 | 1 | 6.750 | 0.0094 | Yes |
| Rl Based | Zero-Shot | 0 | 0 | 0 | 0.000 | 1.0000 | No |
| SC | Self Reflection | 0 | 0 | 0 | 0.000 | 1.0000 | No |
| SC | Zero-Shot | 220 | 96 | 10 | 68.160 | 0.0000 | Yes |
| Self Reflection | Zero-Shot | 0 | 0 | 0 | 0.000 | 1.0000 | No |

## Latency Statistics

| Method | Mean (ms) | Std (ms) |
|--------|-----------|----------|
| Zero-Shot | 830.2 | 12272.0 |
| KB+SC | 114973.5 | 95959.3 |
| CoT | 5556.4 | 27471.3 |
| Self Reflection | 15489.1 | 25171.6 |
| RAG | 1549.0 | 5442.3 |
| Baseline | 4535.5 | 12250.4 |
| SC | 25354.3 | 80844.3 |
| Adaptive | 0.0 | 0.0 |
| Rl Based | 0.0 | 0.0 |
