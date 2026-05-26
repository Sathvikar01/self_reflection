# Tasks

## Plan
- [x] Fix BasePipeline inheritance for adaptive_reflection_pipeline.py and adaptive_tree_pipeline.py
- [x] Consolidate GenerationConfig/GenerationResponse into src/generator/types.py
- [x] Consolidate QueryComplexityAnalyzer into src/utils/complexity.py
- [x] Consolidate MockEmbedder (keep in state_embedder.py, import in value_network.py)
- [x] Unify TreeNode (adaptive_tree_pipeline imports from rl_controller/tree.py)
- [x] Add LRU cache bounds to nim_client.py and async_nim_client.py
- [x] Verify imports work

## Progress
- [x] Step 1: Create shared types.py
- [x] Step 2: Create shared complexity.py
- [x] Step 3: Consolidate MockEmbedder
- [x] Step 4: Unify TreeNode
- [x] Step 5: Add BasePipeline inheritance
- [x] Step 6: Add bounded caches
- [x] Step 7: Update all imports
- [x] Step 8: Verify

## Review
- All changes implemented and verified
- All imports working correctly
- Inheritance chain confirmed
