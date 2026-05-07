"""
REAL BENCHMARK - All 40 Problems with Actual LLM API Calls
===========================================================

Runs all 4 pipelines on all 40 problems with real NVIDIA NIM API calls.
No simulations, no cost savings, stop on errors for debugging.
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from src.orchestration.baseline import BaselineRunner, BaselineConfig
from src.orchestration.self_reflection_pipeline import SelfReflectionPipeline, SelfReflectionConfig
from src.orchestration.adaptive_reflection_pipeline import AdaptiveReflectionPipeline, AdaptiveReflectionConfig
from src.orchestration.pipeline import RLPipeline, PipelineConfig
from src.rl_controller.mcts import MCTSConfig


def safe_print(text: str, max_len: int = 150):
    """Safely print text, handling unicode characters."""
    try:
        # Replace common unicode characters
        text = text.encode('ascii', 'replace').decode('ascii')
        return text[:max_len]
    except:
        return text[:max_len]


def load_dataset(path: str = "data/datasets/complex_extended.json") -> List[Dict]:
    """Load the complex extended dataset."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_pipeline_with_retry(pipeline_name: str, pipeline_obj, problems: List[Dict], api_key: str):
    """Run a pipeline on all problems with detailed error handling."""
    
    print(f"\n{'='*120}")
    print(f"RUNNING: {pipeline_name} - ALL 40 PROBLEMS")
    print(f"{'='*120}")
    
    results = []
    correct_count = 0
    
    for i, problem in enumerate(problems):
        problem_id = problem.get("id", f"problem_{i}")
        question = problem.get("question", "")
        ground_truth = problem.get("answer", "")
        complexity = problem.get("complexity", "unknown")
        category = problem.get("category", "unknown")
        
        print(f"\n{'-'*120}")
        print(f"[{i+1}/40] Problem ID: {problem_id}")
        print(f"Complexity: {complexity} | Category: {category}")
        print(f"Question: {safe_print(question, 200)}...")
        print(f"Expected Answer: {safe_print(ground_truth, 100)}")
        print(f"{'-'*120}")
        
        start_time = time.time()
        try:
            
            # Call solve method
            result = pipeline_obj.solve(
                problem=question,
                problem_id=problem_id,
                ground_truth=ground_truth
            )
            
            latency = time.time() - start_time
            
            # Extract answer (different pipelines have different result types)
            if hasattr(result, 'final_answer'):
                answer = result.final_answer
            elif hasattr(result, 'answer'):
                answer = result.answer
            else:
                answer = str(result)
            
            # Check correctness
            is_correct = result.correct if result.correct is not None else False
            if is_correct:
                correct_count += 1
            
            # Get tokens
            tokens = result.total_tokens if hasattr(result, 'total_tokens') else 0
            
            # Get reflections (for self-reflection pipelines)
            num_reflections = 0
            if hasattr(result, 'reflections') and result.reflections:
                num_reflections = len(result.reflections)
            
            # Get expansions/backtracks (for RL pipeline)
            num_expansions = result.num_expansions if hasattr(result, 'num_expansions') else 0
            num_backtracks = result.num_backtracks if hasattr(result, 'num_backtracks') else 0
            
            # Get complexity score (for adaptive)
            complexity_score = result.complexity_score if hasattr(result, 'complexity_score') else 0.0
            
            print(f"\n[SUCCESS]")
            print(f"  Answer: {safe_print(str(answer), 200)}")
            print(f"  Correct: {is_correct}")
            print(f"  Tokens: {tokens}")
            print(f"  Latency: {latency:.2f}s")
            if num_reflections > 0:
                print(f"  Reflections: {num_reflections}")
            if num_expansions > 0:
                print(f"  Expansions: {num_expansions}, Backtracks: {num_backtracks}")
            
            results.append({
                "problem_id": problem_id,
                "problem": safe_print(question, 200),
                "expected": ground_truth,
                "answer": safe_print(str(answer), 200),
                "correct": is_correct,
                "tokens": tokens,
                "latency": latency,
                "complexity": complexity,
                "category": category,
                "num_reflections": num_reflections,
                "num_expansions": num_expansions,
                "num_backtracks": num_backtracks,
                "complexity_score": complexity_score,
            })
            
        except Exception as e:
            # LOG ERROR AND CONTINUE
            print(f"\n[ERROR] Error occurred!")
            print(f"Error: {str(e)}")
            print(f"\nFull traceback:")
            traceback.print_exc()
            
            # Record the error and continue
            results.append({
                "problem_id": problem_id,
                "problem": safe_print(question, 200),
                "expected": ground_truth,
                "answer": "ERROR",
                "correct": False,
                "tokens": 0,
                "latency": time.time() - start_time,
                "complexity": complexity,
                "category": category,
                "num_reflections": 0,
                "num_expansions": 0,
                "num_backtracks": 0,
                "complexity_score": 0.0,
                "error": str(e),
            })
    
    accuracy = correct_count / len(results) if results else 0
    
    print(f"\n{'='*120}")
    print(f"{pipeline_name} COMPLETE")
    print(f"{'='*120}")
    print(f"Total: {len(results)}/40 problems")
    print(f"Correct: {correct_count}/{len(results)} ({accuracy:.1%})")
    print(f"Total Tokens: {sum(r['tokens'] for r in results):,}")
    print(f"Avg Latency: {sum(r['latency'] for r in results)/len(results):.2f}s")
    print(f"{'='*120}")
    
    return results, accuracy


def main():
    """Run all 4 pipelines on all 40 problems."""
    
    print("\n" + "="*120)
    print("COMPREHENSIVE REAL LLM BENCHMARK")
    print("ALL 40 PROBLEMS - ALL 4 PIPELINES")
    print("NO SIMULATIONS - ACTUAL NVIDIA NIM API CALLS")
    print("="*120)
    
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("\nERROR: NVIDIA_API_KEY not found in .env")
        return
    
    # Load dataset
    problems = load_dataset()
    print(f"\n[OK] Loaded {len(problems)} problems")
    
    # Show complexity distribution
    complexity_counts = {}
    for p in problems:
        c = p.get("complexity", "unknown")
        complexity_counts[c] = complexity_counts.get(c, 0) + 1
    
    print("\nComplexity Distribution:")
    for c, count in sorted(complexity_counts.items()):
        print(f"  {c}: {count} problems")
    
    all_results = {}
    
    # ========================================================================
    # PIPELINE 1: BASELINE (Zero-Shot)
    # ========================================================================
    print("\n\n" + "="*120)
    print("PIPELINE 1/4: BASELINE (ZERO-SHOT)")
    print("="*120)
    
    baseline_config = BaselineConfig(temperature=0.7, max_tokens=2048)
    baseline_runner = BaselineRunner(api_key=api_key, config=baseline_config)
    
    try:
        baseline_results, baseline_acc = run_pipeline_with_retry(
            "Baseline", baseline_runner, problems, api_key
        )
        all_results["Baseline"] = {
            "results": baseline_results,
            "accuracy": baseline_acc
        }
    finally:
        baseline_runner.close()
    
    # ========================================================================
    # PIPELINE 2: FIXED SELF-REFLECTION
    # ========================================================================
    print("\n\n" + "="*120)
    print("PIPELINE 2/4: FIXED SELF-REFLECTION")
    print("="*120)
    
    self_reflect_config = SelfReflectionConfig(
        max_iterations=8,
        min_reasoning_steps=2,
        max_reasoning_steps=4,
        reflection_depth=2,
        enable_selective_reflection=True,
    )
    self_reflect_pipeline = SelfReflectionPipeline(api_key=api_key, config=self_reflect_config)
    
    try:
        self_reflect_results, self_reflect_acc = run_pipeline_with_retry(
            "Self-Reflection", self_reflect_pipeline, problems, api_key
        )
        all_results["Self-Reflection"] = {
            "results": self_reflect_results,
            "accuracy": self_reflect_acc
        }
    finally:
        self_reflect_pipeline.close()
    
    # ========================================================================
    # PIPELINE 3: ADAPTIVE SELF-REFLECTION
    # ========================================================================
    print("\n\n" + "="*120)
    print("PIPELINE 3/4: ADAPTIVE SELF-REFLECTION")
    print("="*120)
    
    adaptive_config = AdaptiveReflectionConfig(
        min_reflections=1,
        max_reflections=5,
        confidence_threshold_increase=0.7,
        confidence_threshold_stop=0.9,
        degradation_threshold=0.1,
        enable_cross_validation=True,
    )
    adaptive_pipeline = AdaptiveReflectionPipeline(api_key=api_key, config=adaptive_config)
    
    try:
        adaptive_results, adaptive_acc = run_pipeline_with_retry(
            "Adaptive", adaptive_pipeline, problems, api_key
        )
        all_results["Adaptive"] = {
            "results": adaptive_results,
            "accuracy": adaptive_acc
        }
    finally:
        adaptive_pipeline.close()
    
    # ========================================================================
    # PIPELINE 4: RL-BASED SELF-REFLECTION
    # ========================================================================
    print("\n\n" + "="*120)
    print("PIPELINE 4/4: RL-BASED SELF-REFLECTION")
    print("="*120)
    
    mcts_config = MCTSConfig(
        exploration_constant=1.0,  # Reduced from 1.414 for more exploitation
        expansion_budget=10,       # Reduced from 20 to cut token usage
        max_tree_depth=8,          # Limit depth to prevent runaway trees
        use_value_network=False,
    )

    rl_config = PipelineConfig(
        max_iterations=10,        # Reduced from 20 for efficiency
        early_stop_score=0.85,
        mcts=mcts_config,
    )
    rl_pipeline = RLPipeline(api_key=api_key, config=rl_config)
    
    try:
        rl_results, rl_acc = run_pipeline_with_retry(
            "RL-Based", rl_pipeline, problems, api_key
        )
        all_results["RL-Based"] = {
            "results": rl_results,
            "accuracy": rl_acc
        }
    finally:
        rl_pipeline.close()
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n\n" + "="*120)
    print("FINAL RESULTS - REAL LLM API CALLS")
    print("="*120)
    
    print(f"\n{'Pipeline':<30} {'Accuracy':>10} {'Correct':>8} {'Avg Tokens':>12} {'Avg Latency':>12}")
    print("-"*120)
    
    for name, data in all_results.items():
        acc = data["accuracy"]
        correct = sum(1 for r in data["results"] if r.get("correct", False))
        avg_tokens = sum(r.get("tokens", 0) for r in data["results"]) / len(data["results"])
        avg_latency = sum(r.get("latency", 0) for r in data["results"]) / len(data["results"])
        print(f"{name:<30} {acc:>9.1%} {correct:>8} {avg_tokens:>12.0f} {avg_latency:>11.2f}s")
    
    print("\n" + "="*120)
    
    # Save results
    output_path = Path("benchmark_results") / f"real_llm_all_40_{int(time.time())}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": time.time(),
            "type": "REAL_LLM_CALLS_ALL_40",
            "total_problems": 40,
            "pipelines": {name: {
                "accuracy": data["accuracy"],
                "results": data["results"]
            } for name, data in all_results.items()}
        }, f, indent=2, default=str)
    
    print(f"\n[OK] Results saved to {output_path}")
    print("="*120)


if __name__ == "__main__":
    main()
