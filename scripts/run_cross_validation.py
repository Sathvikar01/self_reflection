#!/usr/bin/env python3
"""
K-Fold Cross-Validation Benchmark Runner
=========================================

Runs k-fold cross-validation across multiple reasoning strategies to ensure
results are not cherry-picked. Reports mean ± std per method, per-fold
breakdown, and statistical significance via paired t-test / Wilcoxon
signed-rank across folds.

Methods evaluated:
    1. Zero-Shot
    2. Chain-of-Thought (CoT)
    3. Self-Consistency (SC)
    4. RAG
    5. KB+SC+Step1/2

Usage:
    # Mock mode (no API credits needed)
    python scripts/run_cross_validation.py --mock --folds 3 --num-questions 15

    # Real API
    python scripts/run_cross_validation.py --dataset data/datasets/benchmark_dataset.json --folds 5

Author: Self-Reflection Research Team
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from scipy import stats as sp_stats
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generator.mock_client import MockNVIDIANIMClient
from src.generator.nim_client import NVIDIANIMClient
from src.generator.prompts import PromptBuilder
from src.generator.types import GenerationConfig
from src.knowledge.retriever import KnowledgeRetriever
from src.utils.unified_extractor import UnifiedAnswerExtractor


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MethodFoldResult:
    method: str
    fold: int
    accuracy: float
    correct: int
    total: int
    predictions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CVConfig:
    dataset_path: str
    n_folds: int = 5
    model_name: str = "meta/llama-3.1-8b-instruct"
    temperature: float = 0.7
    max_tokens: int = 512
    num_questions: Optional[int] = None
    sc_samples: int = 5
    mock: bool = False
    output_dir: str = "benchmark_results"
    seed: int = 42
    stratified: bool = True


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

METHOD_NAMES = [
    "zero_shot",
    "chain_of_thought",
    "self_consistency",
    "rag",
    "kb_sc_step",
]

METHOD_LABELS = {
    "zero_shot": "Zero-Shot",
    "chain_of_thought": "CoT",
    "self_consistency": "SC",
    "rag": "RAG",
    "kb_sc_step": "KB+SC+Step1/2",
}


class CrossValidationRunner:
    """Runs k-fold cross-validation over the benchmark dataset."""

    def __init__(self, config: CVConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if config.mock:
            self.client = MockNVIDIANIMClient(api_key="mock_key")
            logger.info("Using MockNVIDIANIMClient (no API credits)")
        else:
            api_key = os.getenv("NVIDIA_API_KEY")
            if not api_key:
                raise ValueError("NVIDIA_API_KEY not set. Use --mock for testing without API.")
            self.client = NVIDIANIMClient(api_key=api_key)

        self.prompt_builder = PromptBuilder()
        self.extractor = UnifiedAnswerExtractor()
        self.knowledge_retriever = KnowledgeRetriever()

        self.gen_config = GenerationConfig(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    def load_dataset(self) -> List[Dict[str, Any]]:
        with open(self.config.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if self.config.num_questions:
            data = data[: self.config.num_questions]
        logger.info(f"Loaded {len(data)} questions from {self.config.dataset_path}")
        return data

    # ------------------------------------------------------------------
    # Five methods (mirrors BenchmarkRunner)
    # ------------------------------------------------------------------
    def _run_zero_shot(self, question: str) -> Tuple[str, str]:
        messages = [
            {"role": "system", "content": "Answer the following question with only 'Yes' or 'No'. No explanation needed."},
            {"role": "user", "content": question},
        ]
        resp = self.client.generate(messages, self.gen_config)
        extracted = self.extractor.extract(resp.text)
        return resp.text, extracted.answer

    def _run_cot(self, question: str) -> Tuple[str, str]:
        messages = self.prompt_builder.build_baseline_prompt(question, question_type="reasoning")
        resp = self.client.generate(messages, self.gen_config)
        extracted = self.extractor.extract(resp.text)
        return resp.text, extracted.answer

    def _run_sc(self, question: str) -> Tuple[str, str]:
        sc_config = GenerationConfig(
            model=self.gen_config.model,
            temperature=0.8,
            max_tokens=self.gen_config.max_tokens,
        )
        answers = []
        raw_parts = []
        for _ in range(self.config.sc_samples):
            messages = self.prompt_builder.build_baseline_prompt(question, question_type="reasoning")
            resp = self.client.generate(messages, sc_config)
            extracted = self.extractor.extract(resp.text)
            answers.append(extracted.answer)
            raw_parts.append(resp.text)
        majority = Counter(answers).most_common(1)[0][0]
        return "\n---\n".join(raw_parts[:3]), majority

    def _run_rag(self, question: str) -> Tuple[str, str]:
        messages = self.prompt_builder.build_baseline_prompt(question, question_type="reasoning")
        user_content = messages[-1]["content"]
        augmented, _ = self.knowledge_retriever.inject_knowledge_into_prompt(question, user_content)
        messages[-1]["content"] = augmented
        resp = self.client.generate(messages, self.gen_config)
        extracted = self.extractor.extract(resp.text)
        return resp.text, extracted.answer

    def _run_kb_sc_step(self, question: str) -> Tuple[str, str]:
        sc_config = GenerationConfig(
            model=self.gen_config.model,
            temperature=0.8,
            max_tokens=self.gen_config.max_tokens,
        )
        facts = self.knowledge_retriever.retrieve_relevant_facts(question)
        fact_text = "\n".join(f"- {f.fact}" for f in facts[:3]) if facts else ""

        answers = []
        raw_parts = []
        for _ in range(self.config.sc_samples):
            knowledge_section = f"Relevant facts:\n{fact_text}\n" if fact_text else ""
            step1_prompt = (
                f"Question: {question}\n\n{knowledge_section}\n"
                "Analyze this question. Consider what facts are relevant and "
                "what logical steps are needed to answer it. Be specific about your reasoning."
            )
            step1_resp = self.client.generate(
                [{"role": "system", "content": "You are a careful reasoner. Analyze questions thoroughly before answering."},
                 {"role": "user", "content": step1_prompt}],
                sc_config,
            )
            step2_prompt = (
                f"Based on this analysis:\n\n{step1_resp.text}\n\n"
                f"What is the final answer to: {question}\n\nAnswer with only 'Yes' or 'No'."
            )
            step2_resp = self.client.generate(
                [{"role": "system", "content": "Provide a clear Yes/No answer based on the analysis."},
                 {"role": "user", "content": step2_prompt}],
                sc_config,
            )
            extracted = self.extractor.extract(step2_resp.text)
            answers.append(extracted.answer)
            raw_parts.append(f"Step1: {step1_resp.text[:200]}...\nStep2: {step2_resp.text}")

        majority = Counter(answers).most_common(1)[0][0]
        return "\n---\n".join(raw_parts[:3]), majority

    # ------------------------------------------------------------------
    # Run a single fold
    # ------------------------------------------------------------------
    def _run_fold(self, fold_idx: int, test_indices: List[int], dataset: List[Dict[str, Any]]) -> Dict[str, MethodFoldResult]:
        """Run all methods on a single test fold."""
        method_runners = {
            "zero_shot": self._run_zero_shot,
            "chain_of_thought": self._run_cot,
            "self_consistency": self._run_sc,
            "rag": self._run_rag,
            "kb_sc_step": self._run_kb_sc_step,
        }

        fold_results: Dict[str, MethodFoldResult] = {}
        for method_name, runner in method_runners.items():
            correct = 0
            total = 0
            predictions = []
            for idx in tqdm(test_indices, desc=f"Fold {fold_idx+1} | {METHOD_LABELS[method_name]}", leave=False):
                q = dataset[idx]
                question = q["question"]
                ground_truth = q["answer"].lower().strip()
                try:
                    raw, extracted_answer = runner(question)
                    is_correct = self.extractor.check_answer(extracted_answer, ground_truth)
                except Exception as e:
                    logger.error(f"Error in {method_name} for {q['id']}: {e}")
                    raw, extracted_answer = "", ""
                    is_correct = False
                if is_correct:
                    correct += 1
                total += 1
                predictions.append({
                    "question_id": q["id"],
                    "question": question,
                    "ground_truth": ground_truth,
                    "extracted": extracted_answer,
                    "correct": is_correct,
                })
            accuracy = correct / total if total > 0 else 0.0
            fold_results[method_name] = MethodFoldResult(
                method=method_name,
                fold=fold_idx,
                accuracy=accuracy,
                correct=correct,
                total=total,
                predictions=predictions,
            )
        return fold_results

    # ------------------------------------------------------------------
    # Main cross-validation loop
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        dataset = self.load_dataset()
        questions = [d["question"] for d in dataset]
        labels = [d["answer"].lower().strip() for d in dataset]

        n_folds = self.config.n_folds
        if self.config.stratified:
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.config.seed)
            splits = list(kf.split(questions, labels))
        else:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.config.seed)
            splits = list(kf.split(questions))

        logger.info(f"Running {n_folds}-fold cross-validation on {len(dataset)} questions")

        # per_method[method] = list of per-fold accuracies
        per_method: Dict[str, List[float]] = {m: [] for m in METHOD_NAMES}
        per_fold_details: List[Dict[str, Any]] = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            logger.info(f"--- Fold {fold_idx+1}/{n_folds}  (test size={len(test_idx)}) ---")
            fold_results = self._run_fold(fold_idx, test_idx.tolist(), dataset)

            fold_detail: Dict[str, Any] = {"fold": fold_idx + 1, "test_size": len(test_idx)}
            for m in METHOD_NAMES:
                acc = fold_results[m].accuracy
                per_method[m].append(acc)
                fold_detail[m] = {
                    "accuracy": round(acc * 100, 1),
                    "correct": fold_results[m].correct,
                    "total": fold_results[m].total,
                }
            per_fold_details.append(fold_detail)
            logger.info(f"Fold {fold_idx+1} done: " + " | ".join(
                f"{METHOD_LABELS[m]}={per_method[m][-1]*100:.1f}%" for m in METHOD_NAMES
            ))

        # ------------------------------------------------------------------
        # Statistical tests
        # ------------------------------------------------------------------
        statistical_tests = self._compute_statistical_tests(per_method, n_folds)

        # ------------------------------------------------------------------
        # Aggregate results
        # ------------------------------------------------------------------
        summary: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "dataset_path": self.config.dataset_path,
                "n_folds": n_folds,
                "num_questions": len(dataset),
                "stratified": self.config.stratified,
                "sc_samples": self.config.sc_samples,
                "seed": self.config.seed,
                "mock": self.config.mock,
            },
            "per_method": {},
            "per_fold": per_fold_details,
            "statistical_tests": statistical_tests,
        }

        for m in METHOD_NAMES:
            accs = per_method[m]
            arr = np.array(accs)
            summary["per_method"][m] = {
                "label": METHOD_LABELS[m],
                "mean_accuracy": round(float(arr.mean()) * 100, 1),
                "std_accuracy": round(float(arr.std()) * 100, 1),
                "fold_accuracies": [round(a * 100, 1) for a in accs],
            }

        # ------------------------------------------------------------------
        # Save & display
        # ------------------------------------------------------------------
        json_path = self.output_dir / "cross_validation_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        logger.info(f"Results saved to {json_path}")

        self._print_table(summary, n_folds)
        self._print_statistical_tests(statistical_tests)

        return summary

    # ------------------------------------------------------------------
    # Statistical significance
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_statistical_tests(per_method: Dict[str, List[float]], n_folds: int) -> Dict[str, Any]:
        """Paired t-test and Wilcoxon signed-rank between all method pairs."""
        tests: Dict[str, Any] = {}
        method_pairs = []
        for i, m1 in enumerate(METHOD_NAMES):
            for m2 in METHOD_NAMES[i + 1:]:
                method_pairs.append((m1, m2))

        for m1, m2 in method_pairs:
            a1 = np.array(per_method[m1])
            a2 = np.array(per_method[m2])
            diff = a1 - a2

            # Paired t-test
            if n_folds >= 2:
                t_stat, t_p = sp_stats.ttest_rel(a1, a2)
                if np.isnan(t_stat):
                    t_stat, t_p = 0.0, 1.0
            else:
                t_stat, t_p = 0.0, 1.0

            # Wilcoxon signed-rank (needs >= 5 pairs for meaningful result)
            if n_folds >= 5 and not np.all(diff == 0):
                try:
                    w_stat, w_p = sp_stats.wilcoxon(a1, a2, alternative="two-sided")
                except ValueError:
                    w_stat, w_p = 0.0, 1.0
            else:
                w_stat, w_p = float("nan"), float("nan")

            key = f"{METHOD_LABELS[m1]} vs {METHOD_LABELS[m2]}"
            tests[key] = {
                "methods": [m1, m2],
                "paired_t_test": {
                    "t_statistic": round(float(t_stat), 4),
                    "p_value": round(float(t_p), 4),
                    "significant_005": bool(t_p < 0.05),
                },
                "wilcoxon_signed_rank": {
                    "statistic": round(float(w_stat), 4) if not np.isnan(w_stat) else None,
                    "p_value": round(float(w_p), 4) if not np.isnan(w_p) else None,
                    "significant_005": bool(w_p < 0.05) if not np.isnan(w_p) else None,
                    "note": "Requires >= 5 folds" if n_folds < 5 else None,
                },
            }
        return tests

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    @staticmethod
    def _print_table(summary: Dict[str, Any], n_folds: int):
        fold_headers = "".join(f"  Fold{i+1}" for i in range(n_folds))
        header = f"{'Method':<20} {'Mean +/- Std':>14}{fold_headers}"
        sep = "-" * (20 + 14 + 7 * n_folds + 2)

        print()
        print(f"{n_folds}-Fold Cross-Validation Results")
        print(sep)
        print(header)
        print(sep)

        for m in METHOD_NAMES:
            info = summary["per_method"][m]
            label = info["label"]
            mean_std = f"{info['mean_accuracy']:.1f} +/- {info['std_accuracy']:.1f}%"
            folds_str = "".join(f"  {v:>4.0f}%" for v in info["fold_accuracies"])
            print(f"{label:<20} {mean_std:>14}{folds_str}")

        print(sep)

    @staticmethod
    def _print_statistical_tests(tests: Dict[str, Any]):
        print("\nStatistical Significance (across folds)")
        print("-" * 70)
        print(f"{'Comparison':<30} {'t-test p':>10} {'Wilcoxon p':>12} {'Sig (0.05)':>10}")
        print("-" * 70)
        for name, t in tests.items():
            tp = f"{t['paired_t_test']['p_value']:.4f}"
            wp = f"{t['wilcoxon_signed_rank']['p_value']:.4f}" if t["wilcoxon_signed_rank"]["p_value"] is not None else "N/A"
            sig = "Yes" if t["paired_t_test"]["significant_005"] else "No"
            print(f"{name:<30} {tp:>10} {wp:>12} {sig:>10}")
        print("-" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> CVConfig:
    parser = argparse.ArgumentParser(
        description="K-Fold Cross-Validation Benchmark for LLM Reasoning Strategies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Mock mode (no API needed)
    python scripts/run_cross_validation.py --mock --folds 3 --num-questions 15

    # Real API
    python scripts/run_cross_validation.py --dataset data/datasets/benchmark_dataset.json --folds 5
        """,
    )
    parser.add_argument("--dataset", type=str, default="data/datasets/benchmark_dataset.json",
                        help="Path to dataset JSON file")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds (default: 5)")
    parser.add_argument("--model", type=str, default="meta/llama-3.1-8b-instruct",
                        help="Model name for NIM API")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per generation")
    parser.add_argument("--num-questions", type=int, default=None,
                        help="Limit number of questions (default: all)")
    parser.add_argument("--sc-samples", type=int, default=5,
                        help="Samples for self-consistency methods (default: 5)")
    parser.add_argument("--mock", action="store_true", help="Use mock client (no API credits needed)")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Output directory (default: benchmark_results)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--no-stratified", action="store_true",
                        help="Use plain KFold instead of StratifiedKFold")

    args = parser.parse_args()
    return CVConfig(
        dataset_path=args.dataset,
        n_folds=args.folds,
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        num_questions=args.num_questions,
        sc_samples=args.sc_samples,
        mock=args.mock,
        output_dir=args.output_dir,
        seed=args.seed,
        stratified=not args.no_stratified,
    )


def main():
    config = parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(Path(config.output_dir) / "cross_validation.log", level="DEBUG", rotation="10 MB")

    logger.info("Starting cross-validation run")
    logger.info(f"Config: {asdict(config)}")

    runner = CrossValidationRunner(config)
    results = runner.run()

    logger.info("Cross-validation complete!")
    return results


if __name__ == "__main__":
    main()
