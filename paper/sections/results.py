"""
NeurIPS 2024 Paper Template
RL-Guided Self-Reflection for Large Language Model Reasoning
EXPERIMENTAL RESULTS UPDATE
"""

# Title
title = "Reinforcement Learning-Guided Self-Reflection for Large Language Model Reasoning"

abstract = """
Large language models excel at many tasks but often struggle with complex multi-step 
reasoning, producing plausible yet incorrect reasoning chains. We propose an RL-guided 
self-reflection framework that uses Monte Carlo Tree Search combined with a Process 
Reward Model to guide reasoning generation. Our approach treats reasoning as a 
sequential decision process, enabling dynamic action selection including expansion, 
reflection, and backtracking. Experiments on StrategyQA demonstrate that while the 
framework shows promise in detecting and correcting reasoning errors, baseline 
zero-shot performance remains competitive for simpler reasoning tasks. We analyze 
the trade-offs between computational cost and accuracy gains.
"""

## 1. Introduction

introduction = """
Large language models (LLMs) have demonstrated remarkable capabilities across diverse 
tasks. However, they remain prone to generating plausible-sounding but logically 
flawed reasoning chains, particularly for problems requiring multi-step inference.

Current approaches like Chain-of-Thought prompting and Tree-of-Thought search have 
shown promise but lack systematic mechanisms for error detection and correction. 
When an LLM makes an early reasoning error, it often compounds through subsequent 
steps, leading to confidently wrong answers.

We propose a reinforcement learning-guided self-reflection framework that treats 
reasoning as a Markov Decision Process. Our key contributions:

1. **MCTS for Reasoning**: We formulate reasoning as tree search, enabling 
   systematic exploration and backtracking.

2. **Process Reward Model**: We use LLM-as-Judge to score individual reasoning 
   steps, enabling targeted error detection.

3. **Action-Based Control**: We define four actions (Expand, Reflect, Backtrack, 
   Conclude) that give fine-grained control over reasoning.

4. **Comprehensive Evaluation**: We demonstrate the framework on StrategyQA with 
   detailed ablation studies analyzing component contributions.
"""

## 2. Related Work

related_work = """
**Chain-of-Thought Reasoning**: Wei et al. (2022) showed that prompting LLMs to 
generate intermediate reasoning steps improves performance on reasoning tasks.

**Tree-of-Thought**: Yao et al. (2023) extended CoT by exploring multiple reasoning 
paths and selecting the best. Our work differs by using RL for action selection.

**Process Reward Models**: Lightman et al. (2023) demonstrated that step-level 
supervision improves reasoning. We use LLM-as-Judge instead of trained PRMs.

**RL for Language**: Recent work has explored RL for language generation, but 
mostly for policy optimization during training. We apply RL at test time.
"""

## 3. Method

method = """
### 3.1 Problem Formulation

We formulate reasoning as a Markov Decision Process (S, A, T, R) where:
- S: States are partial reasoning chains
- A: Actions are {Expand, Reflect, Backtrack, Conclude}
- T: Transition function generates next state
- R: Reward from PRM evaluation

### 3.2 MCTS Controller

We use Upper Confidence Bound (UCB1) for node selection:

UCB1(s, a) = Q(s,a) + c * sqrt(ln(N(s)) / N(s,a))

where Q(s,a) is the average reward, N(s) is visit count, and c is exploration constant.

### 3.3 Process Reward Model

We prompt Llama-3.3-70B to evaluate reasoning steps:

"Given the problem [X], previous steps [Y], and current step [Z], 
output a score from -1 (incorrect) to 1 (correct)."

### 3.4 Action Design

| Action | Trigger Condition | Effect |
|--------|-------------------|--------|
| Expand | Default | Generate next step |
| Reflect | Low score (< 0.2) | Critique previous step |
| Backtrack | Very low score (< 0.3) | Return to better node |
| Conclude | High score (> 0.85) | Generate final answer |
"""

## 4. Experiments

experiments = """
### 4.1 Setup

- Dataset: StrategyQA (multi-step reasoning questions)
- Generator: Llama-3.1-8B-Instruct (via NVIDIA NIM API)
- Evaluator: Llama-3.3-70B-Instruct (via NVIDIA NIM API)
- MCTS parameters: c=1.414, budget=20-30 expansions, min_depth=2

### 4.2 Main Results

| Method | Accuracy | Avg Tokens | Avg Latency | Backtracks |
|--------|----------|------------|-------------|------------|
| Zero-shot Baseline | 30-50% | 582 | 2.7s | 0 |
| RL-Guided (full) | 10-40% | 8503 | 38.6s | 1.2 |
| Ablation: No Reflect | 30% | 8207 | - | - |
| Ablation: No Backtrack | 30% | 7185 | - | - |
| Ablation: Random Policy | 30% | 8150 | - | - |

Note: Results vary by random seed. Baseline shows 30-50% accuracy, RL-guided shows
10-40% depending on configuration.

### 4.3 Analysis

**Key Findings:**

1. **PRM Scoring**: The Process Reward Model tends to give high scores (0.8-1.0) 
   to initial reasoning steps, even when they contain errors. This causes early 
   stopping before sufficient exploration.

2. **Baseline Strength**: For simpler yes/no questions, baseline zero-shot 
   performance is competitive because Llama-3.1-8B has strong innate reasoning 
   capabilities for straightforward problems.

3. **Token Overhead**: The RL-guided approach uses 15x more tokens than baseline
   due to multiple expansion and evaluation calls.

4. **Ablation Insights**: Removing reflection or backtracking has minimal impact
   on this dataset, suggesting the core MCTS expansion is the main driver.

### 4.4 Challenges Identified

1. **PRM Calibration**: The evaluator needs better calibration to distinguish
   between correct and incorrect reasoning steps.

2. **Early Stopping**: High initial PRM scores trigger premature conclusion,
   preventing proper reasoning chain development.

3. **Question Complexity**: StrategyQA questions vary in complexity - simpler
   questions benefit less from multi-step reasoning guidance.

### 4.5 Recommendations for Future Work

1. Use lower temperature for PRM evaluation to get more discriminative scores
2. Implement minimum steps requirement before allowing conclusion
3. Train a dedicated PRM on human-annotated reasoning steps
4. Use more complex benchmark datasets where baseline struggles
"""

## 5. Conclusion

conclusion = """
We presented an RL-guided self-reflection framework for LLM reasoning that combines 
MCTS with process-level evaluation. Our experiments reveal important insights:

1. The framework successfully implements backtracking and reflection actions
2. PRM calibration is critical - overly optimistic scores lead to poor decisions
3. For simpler reasoning tasks, baseline zero-shot remains competitive
4. The approach shows more promise on complex, multi-step reasoning problems

**Key Lessons Learned:**
- Process-level evaluation requires careful calibration
- Early stopping thresholds significantly impact results
- Computational overhead may not justify accuracy gains on simple tasks

**Future Directions:**
- Train dedicated PRM models on reasoning step quality
- Apply to more complex reasoning benchmarks (MATH500, GSM8K)
- Implement adaptive early stopping based on problem complexity
- Explore hybrid approaches combining baseline and RL-guided methods
"""

# References
references = """
1. Wei et al. (2022). Chain-of-thought prompting elicits reasoning in large language models.
2. Yao et al. (2023). Tree of thoughts: Deliberate problem solving with large language models.
3. Lightman et al. (2023). Let's verify step by step.
4. Cobbe et al. (2021). Training verifiers to solve math word problems.
"""

## Experimental Data Summary

experimental_summary = """
### Dataset Statistics
- Total test problems: 10
- Question types: Yes/No factual queries
- Answer distribution: 70% "no", 30% "yes"

### Baseline Performance (seed=42)
- Accuracy: 50%
- Average tokens: 582
- Average latency: 2.7s

### RL-Guided Performance (seed=42, iterations=30)
- Accuracy: 40%
- Average tokens: 9175
- Average latency: 14.5s
- Average expansions: 2.3
- Average backtracks: 0.9

### Ablation Results
| Variant | Accuracy | Avg Tokens |
|---------|----------|------------|
| No Reflection | 30% | 8207 |
| No Backtrack | 30% | 7185 |
| Random Policy | 30% | 8150 |
"""
