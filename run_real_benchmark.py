"""
REAL BENCHMARK - Using Actual LLM API Calls
============================================

This script runs the REAL pipeline implementations with actual NVIDIA NIM API calls.
No simulations, no probabilities - only real LLM reasoning.

Uses the actual implementations from src/orchestration/:
- SelfReflectionPipeline
- AdaptiveReflectionPipeline  
- RLPipeline
- BaselineRunner
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import REAL implementations
from src.orchestration.baseline import BaselineRunner, BaselineConfig
from src.orchestration.self_reflection_pipeline import SelfReflectionPipeline, SelfReflectionConfig
from src.orchestration.adaptive_reflection_pipeline import AdaptiveReflectionPipeline, AdaptiveReflectionConfig
from src.orchestration.pipeline import RLPipeline, PipelineConfig
from src.rl_controller.mcts import MCTSConfig


def load_dataset(path: str = "data/datasets/complex_extended.json") -> List[Dict]:
    """Load the complex extended dataset."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_real_baseline(problems: List[Dict], api_key: str, n_problems: int = 5):
    """Run REAL baseline with actual LLM calls."""
    print(f"\n{'='*100}")
    print("RUNNING: Baseline (Zero-Shot) - REAL LLM CALLS")
    print(f"{'='*100}")
    
    config = BaselineConfig(temperature=0.7, max_tokens=2048)
    runner = BaselineRunner(api_key=api_key, config=config)
    
    results = []
    correct = 0
    
    for i, problem in enumerate(problems[:n_problems]):
        problem_id = problem.get("id", f"problem_{i}")
        question = problem.get("question", "")
        ground_truth = problem.get("answer", "")
        
        print(f"\n[{i+1}/{n_problems}] Problem: {problem_id}")
        print(f"Question: {question[:150]}...")
        
        try:
            start = time.time()
            result = runner.solve(problem=question, problem_id=problem_id, ground_truth=ground_truth)
            latency = time.time() - start
            
            is_correct = result.correct if result.correct is not None else False
            if is_correct:
                correct += 1
            
            answer = getattr(result, 'answer', getattr(result, 'final_answer', 'N/A'))
            
            print(f"  Answer: {answer}")
            print(f"  Expected: {ground_truth}")
            print(f"  Correct: {is_correct}")
            print(f"  Latency: {latency:.2f}s")
            print(f"  Tokens: {result.total_tokens}")
            
            results.append({
                "problem_id": problem_id,
                "question": question[:100],
                "answer": answer,
                "expected": ground_truth,
                "correct": is_correct,
                "latency": latency,
                "tokens": result.total_tokens,
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "problem_id": problem_id,
                "error": str(e),
                "correct": False
            })
    
    runner.close()
    
    accuracy = correct / len(results) if results else 0
    print(f"\n{'='*100}")
    print(f"BASELINE RESULTS: {correct}/{len(results)} correct ({accuracy:.1%})")
    print(f"{'='*100}")
    
    return results, accuracy


def run_real_self_reflection(problems: List[Dict], api_key: str, n_problems: int = 5):
    """Run REAL self-reflection with actual LLM calls."""
    print(f"\n{'='*100}")
    print("RUNNING: Fixed Self-Reflection - REAL LLM CALLS")
    print(f"{'='*100}")
    
    config = SelfReflectionConfig(
        max_iterations=8,
        min_reasoning_steps=2,
        max_reasoning_steps=4,
        reflection_depth=2,
        enable_selective_reflection=True,
    )
    
    pipeline = SelfReflectionPipeline(api_key=api_key, config=config)
    
    results = []
    correct = 0
    
    for i, problem in enumerate(problems[:n_problems]):
        problem_id = problem.get("id", f"problem_{i}")
        question = problem.get("question", "")
        ground_truth = problem.get("answer", "")
        
        print(f"\n[{i+1}/{n_problems}] Problem: {problem_id}")
        print(f"Question: {question[:150]}...")
        
        try:
            start = time.time()
            result = pipeline.solve(problem=question, problem_id=problem_id, ground_truth=ground_truth)
            latency = time.time() - start
            
            is_correct = result.correct if result.correct is not None else False
            if is_correct:
                correct += 1
            
            print(f"  Answer: {result.final_answer}")
            print(f"  Expected: {ground_truth}")
            print(f"  Correct: {is_correct}")
            print(f"  Reflections: {len(result.reflections) if hasattr(result, 'reflections') and result.reflections else 0}")
            print(f"  Latency: {latency:.2f}s")
            print(f"  Tokens: {result.total_tokens}")
            
            results.append({
                "problem_id": problem_id,
                "question": question,
                "answer": result.final_answer,
                "expected": ground_truth,
                "correct": is_correct,
                "latency": latency,
                "tokens": result.total_tokens,
                "reflections": len(result.reflections) if hasattr(result, 'reflections') else 0,
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "problem_id": problem_id,
                "error": str(e),
                "correct": False
            })
    
    pipeline.close()
    
    accuracy = correct / len(results) if results else 0
    print(f"\n{'='*100}")
    print(f"SELF-REFLECTION RESULTS: {correct}/{len(results)} correct ({accuracy:.1%})")
    print(f"{'='*100}")
    
    return results, accuracy


def run_real_adaptive(problems: List[Dict], api_key: str, n_problems: int = 5):
    """Run REAL adaptive self-reflection with actual LLM calls."""
    print(f"\n{'='*100}")
    print("RUNNING: Adaptive Self-Reflection - REAL LLM CALLS")
    print(f"{'='*100}")
    
    config = AdaptiveReflectionConfig(
        min_reflections=1,
        max_reflections=5,
        confidence_threshold_increase=0.7,
        confidence_threshold_stop=0.9,
        degradation_threshold=0.1,
        enable_cross_validation=True,
    )
    
    pipeline = AdaptiveReflectionPipeline(api_key=api_key, config=config)
    
    results = []
    correct = 0
    
    for i, problem in enumerate(problems[:n_problems]):
        problem_id = problem.get("id", f"problem_{i}")
        question = problem.get("question", "")
        ground_truth = problem.get("answer", "")
        
        print(f"\n[{i+1}/{n_problems}] Problem: {problem_id}")
        print(f"Question: {question[:150]}...")
        
        try:
            start = time.time()
            result = pipeline.solve(problem=question, problem_id=problem_id, ground_truth=ground_truth)
            latency = time.time() - start
            
            is_correct = result.correct if result.correct is not None else False
            if is_correct:
                correct += 1
            
            print(f"  Answer: {result.final_answer}")
            print(f"  Expected: {ground_truth}")
            print(f"  Correct: {is_correct}")
            print(f"  Complexity: {result.complexity_score:.2f}")
            print(f"  Rollbacks: {result.rolled_back}")
            print(f"  Latency: {latency:.2f}s")
            print(f"  Tokens: {result.total_tokens}")
            
            results.append({
                "problem_id": problem_id,
                "question": question,
                "answer": result.final_answer,
                "expected": ground_truth,
                "correct": is_correct,
                "latency": latency,
                "tokens": result.total_tokens,
                "complexity": result.complexity_score,
                "rollbacks": result.rolled_back,
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "problem_id": problem_id,
                "error": str(e),
                "correct": False
            })
    
    pipeline.close()
    
    accuracy = correct / len(results) if results else 0
    print(f"\n{'='*100}")
    print(f"ADAPTIVE SELF-REFLECTION RESULTS: {correct}/{len(results)} correct ({accuracy:.1%})")
    print(f"{'='*100}")
    
    return results, accuracy


def run_real_rl_guided(problems: List[Dict], api_key: str, n_problems: int = 5):
    """Run REAL RL-guided MCTS with actual LLM calls."""
    print(f"\n{'='*100}")
    print("RUNNING: RL-Based Self-Reflection - REAL LLM CALLS")
    print(f"{'='*100}")
    
    mcts_config = MCTSConfig(
        exploration_constant=1.414,
        expansion_budget=20,
        use_value_network=False,
    )
    
    config = PipelineConfig(
        max_iterations=20,
        early_stop_score=0.85,
        mcts=mcts_config,
    )
    
    pipeline = RLPipeline(api_key=api_key, config=config)
    
    results = []
    correct = 0
    
    for i, problem in enumerate(problems[:n_problems]):
        problem_id = problem.get("id", f"problem_{i}")
        question = problem.get("question", "")
        ground_truth = problem.get("answer", "")
        
        print(f"\n[{i+1}/{n_problems}] Problem: {problem_id}")
        print(f"Question: {question[:150]}...")
        
        try:
            start = time.time()
            result = pipeline.solve(problem=question, problem_id=problem_id, ground_truth=ground_truth)
            latency = time.time() - start
            
            is_correct = result.correct if result.correct is not None else False
            if is_correct:
                correct += 1
            
            print(f"  Answer: {result.answer}")
            print(f"  Expected: {ground_truth}")
            print(f"  Correct: {is_correct}")
            print(f"  Expansions: {result.num_expansions}")
            print(f"  Backtracks: {result.num_backtracks}")
            print(f"  Score: {result.final_score:.2f}")
            print(f"  Latency: {latency:.2f}s")
            print(f"  Tokens: {result.total_tokens}")
            
            results.append({
                "problem_id": problem_id,
                "question": question,
                "answer": result.answer,
                "expected": ground_truth,
                "correct": is_correct,
                "latency": latency,
                "tokens": result.total_tokens,
                "expansions": result.num_expansions,
                "backtracks": result.num_backtracks,
                "score": result.final_score,
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "problem_id": problem_id,
                "error": str(e),
                "correct": False
            })
    
    pipeline.close()
    
    accuracy = correct / len(results) if results else 0
    print(f"\n{'='*100}")
    print(f"RL-GUIDED RESULTS: {correct}/{len(results)} correct ({accuracy:.1%})")
    print(f"{'='*100}")
    
    return results, accuracy


def main():
    """Run all pipelines with REAL LLM calls."""
    
    print("\n" + "="*100)
    print("REAL BENCHMARK - ACTUAL NVIDIA NIM API CALLS")
    print("NO SIMULATIONS, NO PROBABILITIES - ONLY REAL LLM REASONING")
    print("="*100)
    
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("\nERROR: NVIDIA_API_KEY not found in .env")
        return
    
    # Load dataset
    problems = load_dataset()
    print(f"\nLoaded {len(problems)} problems from complex_extended.json")
    
    # Run on subset (5 problems for speed)
    n_problems = 5
    print(f"\nRunning on {n_problems} problems (to save API costs)")
    print("Each pipeline will make REAL LLM calls...")
    
    # Run all pipelines
    all_results = {}
    
    baseline_results, baseline_acc = run_real_baseline(problems, api_key, n_problems)
    all_results["Baseline"] = {"results": baseline_results, "accuracy": baseline_acc}
    
    self_reflect_results, self_reflect_acc = run_real_self_reflection(problems, api_key, n_problems)
    all_results["Self-Reflection"] = {"results": self_reflect_results, "accuracy": self_reflect_acc}
    
    adaptive_results, adaptive_acc = run_real_adaptive(problems, api_key, n_problems)
    all_results["Adaptive"] = {"results": adaptive_results, "accuracy": adaptive_acc}
    
    rl_results, rl_acc = run_real_rl_guided(problems, api_key, n_problems)
    all_results["RL-Guided"] = {"results": rl_results, "accuracy": rl_acc}
    
    # Print final summary
    print("\n" + "="*100)
    print("FINAL RESULTS - REAL LLM CALLS")
    print("="*100)
    
    print(f"\n{'Pipeline':<30} {'Accuracy':>10} {'Correct':>8} {'Avg Latency':>12}")
    print("-"*100)
    
    for name, data in all_results.items():
        acc = data["accuracy"]
        correct = sum(1 for r in data["results"] if r.get("correct", False))
        avg_latency = sum(r.get("latency", 0) for r in data["results"]) / len(data["results"])
        print(f"{name:<30} {acc:>9.1%} {correct:>8} {avg_latency:>11.2f}s")
    
    # Save results
    output_path = Path("benchmark_results") / f"real_llm_benchmark_{int(time.time())}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": time.time(),
            "type": "REAL_LLM_CALLS",
            "n_problems": n_problems,
            "results": all_results
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
