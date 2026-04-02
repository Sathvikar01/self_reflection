# Self-Reflection Pipeline - Next Steps Plan

## Goal
Achieve statistically significant improvement with:
- 100+ problems
- Knowledge retrieval for factual questions
- Selective reflection based on problem type

## Phase 1: Expand Dataset to 100+ Problems
- [ ] Generate additional StrategyQA-style problems programmatically
- [ ] Create diverse question types (factual, reasoning, strategic)
- [ ] Ensure balanced yes/no distribution
- [ ] Validate all ground truth answers

## Phase 2: Implement Knowledge Retrieval
- [ ] Add fact-checking step before final answer
- [ ] Create knowledge base for scientific facts
- [ ] Implement retrieval-augmented generation for factual questions
- [ ] Test on known failure cases (diamond, fish drowning)

## Phase 3: Implement Selective Reflection
- [ ] Classify problem type (factual vs reasoning)
- [ ] Skip reflection for high-confidence baseline answers
- [ ] Apply deeper reflection for reasoning-heavy problems
- [ ] A/B test selective vs full reflection

## Phase 4: Large-Scale Benchmark
- [ ] Run baseline on 100+ problems
- [ ] Run self-reflection on 100+ problems  
- [ ] Statistical analysis with McNemar's test
- [ ] Document results and significance

## Success Criteria
- [ ] p < 0.05 on McNemar's test
- [ ] Clear accuracy improvement over baseline
- [ ] No degradation on previously correct answers
