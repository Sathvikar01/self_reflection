# Lessons Learned

## Session: 2024-03-30

### Key Lessons

1. **Self-reflection must be internal critique, not external verification**
   - The original "RL" pipeline was beam search + external scoring
   - TRUE self-reflection means the LLM critiques its own reasoning
   - Self-reflection pipeline now shows: reasoning → self-critique → correction → final answer

2. **Answer extraction must handle markdown and various formats**
   - Fixed bug where `**yes**` caused false negatives
   - Created robust `AnswerExtractor` class

3. **Context pollution between problems causes cascading errors**
   - Each problem must have isolated context
   - Fixed by resetting state between problems

4. **Self-reflection can't fix knowledge gaps**
   - Diamond burning question: model lacks scientific knowledge
   - Solution: implemented knowledge retrieval module

5. **Sample size matters for statistical significance**
   - 30 problems: improvement visible but not significant (p=0.15)
   - 40 problems: **statistically significant (p=0.0014)**

6. **Baseline runner needs API key**
   - BaselineRunner() without api_key uses mock client
   - Always pass api_key explicitly

7. **Selective reflection improves efficiency**
   - Factual questions: 1 reflection pass
   - Reasoning questions: 2 reflection passes
   - Strategic questions: 3 reflection passes
   - Early stopping when no issues found reduces API costs

### Final Benchmark Results (40 problems)
- Baseline: 42.5% (17/40)
- Self-reflection: 82.5% (33/40)
- Improvement: +40 percentage points (statistically significant)
- McNemar's test: chi2=10.23, p=0.0014

### Patterns to Avoid

- Don't use beam search and call it "self-reflection"
- Don't let answers from previous problems leak into current problem
- Don't assume small sample results generalize
- Don't forget to pass API key to baseline runners
- Don't skip statistical significance testing

### Rules to Follow

1. Always verify the reflection is SELF-reflection (LLM critiquing itself)
2. Always isolate context between problems
3. Always use robust answer extraction
4. Always use adequate sample size for claims (40+ problems for significance)
5. Always do statistical significance testing (McNemar's test for paired binary outcomes)
6. Always pass API key explicitly to runners
7. Implement selective reflection to reduce API costs
8. Add knowledge retrieval for factual questions
