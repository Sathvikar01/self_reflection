#!/usr/bin/env python3
"""
Ablation Study Runner
=====================

Runs a controlled ablation study to validate each component of the
proposed KB+SC+Step1/2 method. Each ablation condition removes exactly
one component to measure its individual contribution.

Ablation Conditions:
    1. KB+SC+Step1/2 (Full)   - Proposed method with all components
    2. SC+Step1/2 (No KB)     - Remove knowledge base injection
    3. KB+Step1/2 (No SC)     - Remove self-consistency (single gen)
    4. KB+SC (No Step1/2)     - Remove structured prompting
    5. Zero-Shot               - Baseline: direct answer
    6. CoT                     - Baseline: chain-of-thought only

Usage:
    python scripts/run_ablation.py --dataset data/datasets/benchmark_dataset.json
    python scripts/run_ablation.py --mock --num-questions 10
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
from scipy.stats import chi2 as chi2_dist
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
from src.generator.prompts import PromptBuilder, ReasoningContext
from src.generator.types import GenerationConfig
from src.knowledge.retriever import KnowledgeRetriever
from src.utils.unified_extractor import UnifiedAnswerExtractor


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class AblationResult:
    condition: str
    predicted_answer: str
    extracted_answer: str
    is_correct: bool
    raw_response: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    num_samples: int = 1
    sample_answers: List[str] = field(default_factory=list)


@dataclass
class QuestionResult:
    question_id: str
    question: str
    ground_truth: str
    condition_results: Dict[str, AblationResult] = field(default_factory=dict)


# ============================================================================
# Ablation Runner
# ============================================================================

class AblationRunner:
    """Runs controlled ablation studies for the proposed method."""

    CONDITIONS = [
        "kb_sc_step12",   # Full method
        "sc_step12",      # No KB
        "kb_step12",      # No SC
        "kb_sc",          # No Step1/2
        "zero_shot",      # Baseline
        "cot",            # Baseline
    ]

    CONDITION_LABELS = {
        "kb_sc_step12": "KB+SC+Step1/2",
        "sc_step12":    "SC+Step1/2 (No KB)",
        "kb_step12":    "KB+Step1/2 (No SC)",
        "kb_sc":        "KB+SC (No Step1/2)",
        "zero_shot":    "Zero-Shot",
        "cot":          "CoT",
    }

    def __init__(self, dataset_path: str, mock: bool = False,
                 num_questions: Optional[int] = None,
                 sc_samples: int = 5, model: str = "meta/llama-3.1-8b-instruct",
                 output_dir: str = "benchmark_results", seed: int = 42):
        self.dataset_path = dataset_path
        self.mock = mock
        self.num_questions = num_questions
        self.sc_samples = sc_samples
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if mock:
            self.client = MockNVIDIANIMClient(api_key="mock_key")
            logger.info("Using MockNVIDIANIMClient (no API credits)")
        else:
            api_key = os.getenv("NVIDIA_API_KEY")
            if not api_key:
                raise ValueError("NVIDIA_API_KEY not set. Use --mock for testing.")
            self.client = NVIDIANIMClient(api_key=api_key)

        self.prompt_builder = PromptBuilder()
        self.extractor = UnifiedAnswerExtractor()
        self.knowledge_retriever = KnowledgeRetriever()

        self.gen_config = GenerationConfig(
            model=model,
            temperature=0.7,
            max_tokens=512,
        )

        self.results: List[QuestionResult] = []

    def load_dataset(self) -> List[Dict[str, Any]]:
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if self.num_questions:
            data = data[:self.num_questions]
        logger.info(f"Loaded {len(data)} questions from {self.dataset_path}")
        return data

    # ------------------------------------------------------------------
    # Condition 1: Full method (KB+SC+Step1/2)
    # ------------------------------------------------------------------
    def run_full_method(self, question: str) -> AblationResult:
        start_time = time.time()
        sample_answers = []
        total_in, total_out = 0, 0
        raw_parts = []

        sc_config = GenerationConfig(
            model=self.gen_config.model,
            temperature=0.8,
            max_tokens=self.gen_config.max_tokens,
        )

        facts = self.knowledge_retriever.retrieve_relevant_facts(question)
        fact_text = ""
        if facts:
            fact_text = "\n".join(f"• {f.fact}" for f in facts[:3])

        for _ in range(self.sc_samples):
            knowledge_section = f"Relevant facts:\n{fact_text}\n" if fact_text else ""

            step1_prompt = (
                f"Question: {question}\n\n{knowledge_section}\n"
                "Analyze this question. Consider what facts are relevant and "
                "what logical steps are needed to answer it."
            )
            step1_msgs = [
                {"role": "system", "content": "You are a careful reasoner. Analyze questions thoroughly before answering."},
                {"role": "user", "content": step1_prompt},
            ]
            step1_resp = self.client.generate(step1_msgs, sc_config)
            total_in += step1_resp.input_tokens
            total_out += step1_resp.output_tokens

            step2_prompt = (
                f"Based on this analysis:\n\n{step1_resp.text}\n\n"
                f"What is the final answer to: {question}\n\nAnswer with only 'Yes' or 'No'."
            )
            step2_msgs = [
                {"role": "system", "content": "Provide a clear Yes/No answer based on the analysis."},
                {"role": "user", "content": step2_prompt},
            ]
            step2_resp = self.client.generate(step2_msgs, sc_config)
            total_in += step2_resp.input_tokens
            total_out += step2_resp.output_tokens

            extracted = self.extractor.extract(step2_resp.text)
            sample_answers.append(extracted.answer)
            raw_parts.append(f"Step1: {step1_resp.text[:150]}... | Step2: {step2_resp.text}")

        latency = (time.time() - start_time) * 1000
        majority = Counter(sample_answers).most_common(1)[0][0]

        return AblationResult(
            condition="kb_sc_step12",
            predicted_answer=majority,
            extracted_answer=majority,
            is_correct=False,
            raw_response="\n---\n".join(raw_parts[:3]),
            latency_ms=latency,
            input_tokens=total_in,
            output_tokens=total_out,
            num_samples=self.sc_samples,
            sample_answers=sample_answers,
        )

    # ------------------------------------------------------------------
    # Condition 2: No KB (SC+Step1/2 only)
    # ------------------------------------------------------------------
    def run_no_kb(self, question: str) -> AblationResult:
        start_time = time.time()
        sample_answers = []
        total_in, total_out = 0, 0
        raw_parts = []

        sc_config = GenerationConfig(
            model=self.gen_config.model,
            temperature=0.8,
            max_tokens=self.gen_config.max_tokens,
        )

        for _ in range(self.sc_samples):
            step1_prompt = (
                f"Question: {question}\n\n"
                "Analyze this question. Consider what facts are relevant and "
                "what logical steps are needed to answer it."
            )
            step1_msgs = [
                {"role": "system", "content": "You are a careful reasoner. Analyze questions thoroughly before answering."},
                {"role": "user", "content": step1_prompt},
            ]
            step1_resp = self.client.generate(step1_msgs, sc_config)
            total_in += step1_resp.input_tokens
            total_out += step1_resp.output_tokens

            step2_prompt = (
                f"Based on this analysis:\n\n{step1_resp.text}\n\n"
                f"What is the final answer to: {question}\n\nAnswer with only 'Yes' or 'No'."
            )
            step2_msgs = [
                {"role": "system", "content": "Provide a clear Yes/No answer based on the analysis."},
                {"role": "user", "content": step2_prompt},
            ]
            step2_resp = self.client.generate(step2_msgs, sc_config)
            total_in += step2_resp.input_tokens
            total_out += step2_resp.output_tokens

            extracted = self.extractor.extract(step2_resp.text)
            sample_answers.append(extracted.answer)
            raw_parts.append(f"Step1: {step1_resp.text[:150]}... | Step2: {step2_resp.text}")

        latency = (time.time() - start_time) * 1000
        majority = Counter(sample_answers).most_common(1)[0][0]

        return AblationResult(
            condition="sc_step12",
            predicted_answer=majority,
            extracted_answer=majority,
            is_correct=False,
            raw_response="\n---\n".join(raw_parts[:3]),
            latency_ms=latency,
            input_tokens=total_in,
            output_tokens=total_out,
            num_samples=self.sc_samples,
            sample_answers=sample_answers,
        )

    # ------------------------------------------------------------------
    # Condition 3: No SC (KB+Step1/2, single generation)
    # ------------------------------------------------------------------
    def run_no_sc(self, question: str) -> AblationResult:
        start_time = time.time()
        total_in, total_out = 0, 0

        facts = self.knowledge_retriever.retrieve_relevant_facts(question)
        fact_text = ""
        if facts:
            fact_text = "\n".join(f"• {f.fact}" for f in facts[:3])

        knowledge_section = f"Relevant facts:\n{fact_text}\n" if fact_text else ""

        step1_prompt = (
            f"Question: {question}\n\n{knowledge_section}\n"
            "Analyze this question. Consider what facts are relevant and "
            "what logical steps are needed to answer it."
        )
        step1_msgs = [
            {"role": "system", "content": "You are a careful reasoner. Analyze questions thoroughly before answering."},
            {"role": "user", "content": step1_prompt},
        ]
        step1_resp = self.client.generate(step1_msgs, self.gen_config)
        total_in += step1_resp.input_tokens
        total_out += step1_resp.output_tokens

        step2_prompt = (
            f"Based on this analysis:\n\n{step1_resp.text}\n\n"
            f"What is the final answer to: {question}\n\nAnswer with only 'Yes' or 'No'."
        )
        step2_msgs = [
            {"role": "system", "content": "Provide a clear Yes/No answer based on the analysis."},
            {"role": "user", "content": step2_prompt},
        ]
        step2_resp = self.client.generate(step2_msgs, self.gen_config)
        total_in += step2_resp.input_tokens
        total_out += step2_resp.output_tokens

        latency = (time.time() - start_time) * 1000
        extracted = self.extractor.extract(step2_resp.text)

        return AblationResult(
            condition="kb_step12",
            predicted_answer=step2_resp.text,
            extracted_answer=extracted.answer,
            is_correct=False,
            raw_response=f"Step1: {step1_resp.text[:200]}... | Step2: {step2_resp.text}",
            latency_ms=latency,
            input_tokens=total_in,
            output_tokens=total_out,
            num_samples=1,
            sample_answers=[extracted.answer],
        )

    # ------------------------------------------------------------------
    # Condition 4: No Step1/2 (KB+SC, flat KB injection + CoT)
    # ------------------------------------------------------------------
    def run_no_step12(self, question: str) -> AblationResult:
        start_time = time.time()
        sample_answers = []
        total_in, total_out = 0, 0
        raw_parts = []

        sc_config = GenerationConfig(
            model=self.gen_config.model,
            temperature=0.8,
            max_tokens=self.gen_config.max_tokens,
        )

        facts = self.knowledge_retriever.retrieve_relevant_facts(question)
        fact_text = ""
        if facts:
            fact_text = "\n".join(f"• {f.fact}" for f in facts[:3])

        for _ in range(self.sc_samples):
            knowledge_section = f"Relevant facts:\n{fact_text}\n\n" if fact_text else ""

            prompt = (
                f"{knowledge_section}"
                f"Question: {question}\n\n"
                "Think step by step and then answer with 'Yes' or 'No'."
            )
            messages = [
                {"role": "system", "content": "You are a careful reasoner. Think step by step."},
                {"role": "user", "content": prompt},
            ]
            resp = self.client.generate(messages, sc_config)
            total_in += resp.input_tokens
            total_out += resp.output_tokens

            extracted = self.extractor.extract(resp.text)
            sample_answers.append(extracted.answer)
            raw_parts.append(resp.text)

        latency = (time.time() - start_time) * 1000
        majority = Counter(sample_answers).most_common(1)[0][0]

        return AblationResult(
            condition="kb_sc",
            predicted_answer=majority,
            extracted_answer=majority,
            is_correct=False,
            raw_response="\n---\n".join(raw_parts[:3]),
            latency_ms=latency,
            input_tokens=total_in,
            output_tokens=total_out,
            num_samples=self.sc_samples,
            sample_answers=sample_answers,
        )

    # ------------------------------------------------------------------
    # Condition 5: Zero-Shot
    # ------------------------------------------------------------------
    def run_zero_shot(self, question: str) -> AblationResult:
        start_time = time.time()

        messages = [
            {"role": "system", "content": "Answer with only 'Yes' or 'No'. No explanation needed."},
            {"role": "user", "content": question},
        ]
        resp = self.client.generate(messages, self.gen_config)
        latency = (time.time() - start_time) * 1000

        extracted = self.extractor.extract(resp.text)

        return AblationResult(
            condition="zero_shot",
            predicted_answer=resp.text,
            extracted_answer=extracted.answer,
            is_correct=False,
            raw_response=resp.text,
            latency_ms=latency,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
        )

    # ------------------------------------------------------------------
    # Condition 6: CoT only
    # ------------------------------------------------------------------
    def run_cot(self, question: str) -> AblationResult:
        start_time = time.time()

        messages = self.prompt_builder.build_baseline_prompt(question, question_type="reasoning")
        resp = self.client.generate(messages, self.gen_config)
        latency = (time.time() - start_time) * 1000

        extracted = self.extractor.extract(resp.text)

        return AblationResult(
            condition="cot",
            predicted_answer=resp.text,
            extracted_answer=extracted.answer,
            is_correct=False,
            raw_response=resp.text,
            latency_ms=latency,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
        )

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        dataset = self.load_dataset()

        condition_runners = [
            ("kb_sc_step12", self.run_full_method),
            ("sc_step12",    self.run_no_kb),
            ("kb_step12",    self.run_no_sc),
            ("kb_sc",        self.run_no_step12),
            ("zero_shot",    self.run_zero_shot),
            ("cot",          self.run_cot),
        ]

        for q_data in tqdm(dataset, desc="Ablation study"):
            qid = q_data["id"]
            question = q_data["question"]
            ground_truth = q_data["answer"].lower().strip()

            result = QuestionResult(
                question_id=qid,
                question=question,
                ground_truth=ground_truth,
            )

            for cond_name, cond_fn in condition_runners:
                try:
                    cond_result = cond_fn(question)
                    cond_result.is_correct = self.extractor.check_answer(
                        cond_result.extracted_answer, ground_truth
                    )
                    result.condition_results[cond_name] = cond_result
                except Exception as e:
                    logger.error(f"Error in {cond_name} for {qid}: {e}")
                    result.condition_results[cond_name] = AblationResult(
                        condition=cond_name,
                        predicted_answer="ERROR",
                        extracted_answer="",
                        is_correct=False,
                        raw_response=str(e),
                        latency_ms=0,
                        input_tokens=0,
                        output_tokens=0,
                    )

            self.results.append(result)

        stats = self._compute_statistics()
        self._save_results(stats)
        return stats

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def _compute_statistics(self) -> Dict[str, Any]:
        n = len(self.results)
        stats: Dict[str, Any] = {
            "num_questions": n,
            "conditions": {},
            "mcnemar_vs_full": {},
        }

        full_acc = 0.0

        for cond in self.CONDITIONS:
            correct = sum(
                1 for r in self.results
                if cond in r.condition_results and r.condition_results[cond].is_correct
            )
            total = sum(
                1 for r in self.results if cond in r.condition_results
            )
            accuracy = correct / total if total > 0 else 0.0

            latencies = [
                r.condition_results[cond].latency_ms
                for r in self.results if cond in r.condition_results
            ]

            stats["conditions"][cond] = {
                "label": self.CONDITION_LABELS[cond],
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            }

            if cond == "kb_sc_step12":
                full_acc = accuracy

        # McNemar's test: full method vs each ablation
        for cond in self.CONDITIONS[1:]:
            a = b = c = d = 0
            for r in self.results:
                if "kb_sc_step12" in r.condition_results and cond in r.condition_results:
                    full_ok = r.condition_results["kb_sc_step12"].is_correct
                    cond_ok = r.condition_results[cond].is_correct
                    if full_ok and cond_ok:
                        a += 1
                    elif full_ok and not cond_ok:
                        b += 1
                    elif not full_ok and cond_ok:
                        c += 1
                    else:
                        d += 1

            if b + c > 0:
                chi2 = (abs(b - c) - 1) ** 2 / (b + c)
                p_value = float(1 - chi2_dist.cdf(chi2, df=1))
            else:
                chi2 = 0.0
                p_value = 1.0

            delta = stats["conditions"][cond]["accuracy"] - full_acc

            stats["mcnemar_vs_full"][cond] = {
                "chi2": float(chi2),
                "p_value": p_value,
                "significant_005": p_value < 0.05,
                "delta_pp": round(delta * 100, 1),
                "contingency": {"a": a, "b": b, "c": c, "d": d},
            }

        return stats

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def _save_results(self, stats: Dict[str, Any]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output = {
            "timestamp": timestamp,
            "dataset": self.dataset_path,
            "mock": self.mock,
            "sc_samples": self.sc_samples,
            "statistics": stats,
            "per_question": [
                {
                    "question_id": r.question_id,
                    "question": r.question,
                    "ground_truth": r.ground_truth,
                    "conditions": {
                        k: {
                            "extracted": v.extracted_answer,
                            "correct": v.is_correct,
                            "latency_ms": v.latency_ms,
                        }
                        for k, v in r.condition_results.items()
                    },
                }
                for r in self.results
            ],
        }

        json_path = self.output_dir / "ablation_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        logger.info(f"Ablation results saved to {json_path}")

        self._print_table(stats)

    def _print_table(self, stats: Dict[str, Any]):
        n = stats["num_questions"]
        full_acc = stats["conditions"]["kb_sc_step12"]["accuracy"]

        sep = "-" * 55
        print()
        print(f"Ablation Study Results (N={n})")
        print(sep)
        print(f"{'Condition':<22} {'Accuracy':>8} {'Delta Full':>10} {'p-value':>10}")
        print(sep)

        for cond in self.CONDITIONS:
            label = self.CONDITION_LABELS[cond]
            acc = stats["conditions"][cond]["accuracy"]

            if cond == "kb_sc_step12":
                print(f"{label:<22} {acc*100:>7.1f}% {'--':>10} {'--':>10}")
            else:
                mcn = stats["mcnemar_vs_full"][cond]
                delta = mcn["delta_pp"]
                p = mcn["p_value"]
                delta_str = f"{delta:+.1f}pp"
                p_str = f"{p:.3f}" if p >= 0.001 else "<0.001"
                short = {"sc_step12": "w/o KB", "kb_step12": "w/o SC", "kb_sc": "w/o Step1/2", "zero_shot": "Zero-Shot", "cot": "CoT"}
                lbl = short.get(cond, label)
                print(f"  {lbl:<20} {acc*100:>7.1f}% {delta_str:>10} {p_str:>10}")

        print(sep)
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ablation study for KB+SC+Step1/2 method.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default="data/datasets/benchmark_dataset.json")
    parser.add_argument("--mock", action="store_true", help="Use mock client")
    parser.add_argument("--num-questions", type=int, default=None)
    parser.add_argument("--sc-samples", type=int, default=5)
    parser.add_argument("--model", type=str, default="meta/llama-3.1-8b-instruct")
    parser.add_argument("--output-dir", type=str, default="benchmark_results")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    runner = AblationRunner(
        dataset_path=args.dataset,
        mock=args.mock,
        num_questions=args.num_questions,
        sc_samples=args.sc_samples,
        model=args.model,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    stats = runner.run()
    logger.info("Ablation study complete!")
    return stats


if __name__ == "__main__":
    main()
