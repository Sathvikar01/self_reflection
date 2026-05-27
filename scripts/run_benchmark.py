#!/usr/bin/env python3
"""
Benchmark Runner for LLM Reasoning Strategies
=============================================

A clean, reproducible benchmark pipeline that evaluates multiple reasoning
strategies on yes/no questions with proper experimental methodology.

Methods evaluated:
    1. Zero-Shot: Direct answer without reasoning chain
    2. Chain-of-Thought (CoT): Step-by-step reasoning
    3. Self-Consistency (SC): Multiple samples with majority voting
    4. RAG: Knowledge injection without structured prompting
    5. KB+SC+Step1/2: Structured knowledge prompting with self-consistency

Usage:
    # Run with real API
    python scripts/run_benchmark.py --dataset data/datasets/benchmark_dataset.json

    # Run in mock mode (no API credits needed)
    python scripts/run_benchmark.py --mock --num-questions 10

    # Resume from checkpoint
    python scripts/run_benchmark.py --resume --checkpoint benchmark_results/checkpoint.json

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
from scipy.stats import chi2 as chi2_dist
from tqdm import tqdm


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
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

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generator.mock_client import MockNVIDIANIMClient
from src.generator.mimo_client import MiMoClient
from src.generator.nim_client import NVIDIANIMClient
from src.generator.prompts import PromptBuilder, ReasoningContext
from src.generator.types import GenerationConfig, GenerationResponse
from src.knowledge.retriever import KnowledgeRetriever
from src.utils.unified_extractor import UnifiedAnswerExtractor


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class MethodResult:
    """Result for a single method on a single question."""
    method: str
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
    """Result for a single question across all methods."""
    question_id: str
    question: str
    ground_truth: str
    method_results: Dict[str, MethodResult] = field(default_factory=dict)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""
    dataset_path: str
    model_name: str = "mimo-v2.5-pro"
    temperature: float = 0.7
    max_tokens: int = 1024
    num_questions: Optional[int] = None
    sc_samples: int = 5
    mock: bool = False
    output_dir: str = "benchmark_results"
    checkpoint_path: Optional[str] = None
    resume: bool = False
    seed: int = 42
    provider: str = "mimo"  # "mimo" or "nvidia"


# ============================================================================
# Benchmark Methods
# ============================================================================

class BenchmarkRunner:
    """
    Runs benchmark evaluation across multiple reasoning strategies.

    This class implements proper experimental methodology:
    - All methods run on the SAME questions
    - UnifiedAnswerExtractor ensures fair evaluation
    - Checkpoint support for resumable runs
    - Statistical analysis built-in
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize client
        if config.mock:
            self.client = MockNVIDIANIMClient(api_key="mock_key")
            logger.info("Using MockNVIDIANIMClient (no API credits)")
        elif config.provider == "mimo":
            api_key = os.getenv("MIMO_API_KEY")
            if not api_key:
                raise ValueError("MIMO_API_KEY not set. Use --mock for testing without API.")
            self.client = MiMoClient(api_key=api_key, model=config.model_name)
            logger.info(f"Using MiMo client (model={config.model_name})")
        else:
            api_key = os.getenv("NVIDIA_API_KEY")
            if not api_key:
                raise ValueError("NVIDIA_API_KEY not set. Use --mock for testing without API.")
            self.client = NVIDIANIMClient(api_key=api_key)

        # Initialize components
        self.prompt_builder = PromptBuilder()
        self.extractor = UnifiedAnswerExtractor()
        self.knowledge_retriever = KnowledgeRetriever()

        # Generation config
        self.gen_config = GenerationConfig(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        # Results storage
        self.results: List[QuestionResult] = []
        self.checkpoint_path = self.output_dir / "checkpoint.json"

        # Load checkpoint if resuming
        if config.resume and config.checkpoint_path:
            self._load_checkpoint(config.checkpoint_path)

    def _load_checkpoint(self, path: str):
        """Load results from checkpoint file."""
        checkpoint_path = Path(path)
        if checkpoint_path.exists():
            with open(checkpoint_path, "r") as f:
                data = json.load(f)
            self.results = [
                QuestionResult(
                    question_id=r["question_id"],
                    question=r["question"],
                    ground_truth=r["ground_truth"],
                    method_results={
                        k: MethodResult(**v)
                        for k, v in r["method_results"].items()
                    },
                )
                for r in data["results"]
            ]
            logger.info(f"Loaded {len(self.results)} results from checkpoint")

    def _save_checkpoint(self):
        """Save current results to checkpoint file."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "config": asdict(self.config),
            "results": [
                {
                    "question_id": r.question_id,
                    "question": r.question,
                    "ground_truth": r.ground_truth,
                    "method_results": {
                        k: asdict(v) for k, v in r.method_results.items()
                    },
                }
                for r in self.results
            ],
        }
        with open(self.checkpoint_path, "w") as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)
        logger.debug(f"Saved checkpoint with {len(self.results)} results")

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load dataset from JSON file."""
        with open(self.config.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if self.config.num_questions:
            data = data[: self.config.num_questions]

        logger.info(f"Loaded {len(data)} questions from {self.config.dataset_path}")
        return data

    # ------------------------------------------------------------------
    # Method 1: Zero-Shot
    # ------------------------------------------------------------------
    def run_zero_shot(self, question: str) -> MethodResult:
        """
        Zero-Shot baseline: Direct answer without explicit reasoning chain.

        The model receives only the question and must answer immediately.
        This tests the model's inherent knowledge without any reasoning support.

        Args:
            question: The yes/no question to answer

        Returns:
            MethodResult with prediction and metadata
        """
        start_time = time.time()

        messages = [
            {
                "role": "system",
                "content": "Answer the following question with only 'Yes' or 'No'. No explanation needed.",
            },
            {"role": "user", "content": question},
        ]

        response = self.client.generate(messages, self.gen_config)
        latency = (time.time() - start_time) * 1000

        extracted = self.extractor.extract(response.text)

        return MethodResult(
            method="zero_shot",
            predicted_answer=response.text,
            extracted_answer=extracted.answer,
            is_correct=False,  # Will be set later
            raw_response=response.text,
            latency_ms=latency,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

    # ------------------------------------------------------------------
    # Method 2: Chain-of-Thought (CoT)
    # ------------------------------------------------------------------
    def run_chain_of_thought(self, question: str) -> MethodResult:
        """
        Chain-of-Thought: Step-by-step reasoning before answering.

        The model is prompted to think through the problem step by step,
        then provide a final answer. This tests whether explicit reasoning
        improves accuracy over zero-shot.

        Args:
            question: The yes/no question to answer

        Returns:
            MethodResult with prediction and metadata
        """
        start_time = time.time()

        messages = self.prompt_builder.build_baseline_prompt(
            question, question_type="reasoning"
        )

        response = self.client.generate(messages, self.gen_config)
        latency = (time.time() - start_time) * 1000

        extracted = self.extractor.extract(response.text)

        return MethodResult(
            method="chain_of_thought",
            predicted_answer=response.text,
            extracted_answer=extracted.answer,
            is_correct=False,
            raw_response=response.text,
            latency_ms=latency,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

    # ------------------------------------------------------------------
    # Method 3: Self-Consistency (SC)
    # ------------------------------------------------------------------
    def run_self_consistency(self, question: str) -> MethodResult:
        """
        Self-Consistency with majority voting.

        Generates multiple reasoning chains at temperature > 0, extracts
        answers from each, then selects the majority answer. This tests
        whether sampling multiple reasoning paths improves reliability.

        Args:
            question: The yes/no question to answer

        Returns:
            MethodResult with majority-voted prediction
        """
        start_time = time.time()
        sample_answers = []
        total_input_tokens = 0
        total_output_tokens = 0
        raw_responses = []

        # Use higher temperature for diversity
        sc_config = GenerationConfig(
            model=self.gen_config.model,
            temperature=0.8,  # Higher temp for diversity
            max_tokens=self.gen_config.max_tokens,
        )

        for _ in range(self.config.sc_samples):
            messages = self.prompt_builder.build_baseline_prompt(
                question, question_type="reasoning"
            )
            response = self.client.generate(messages, sc_config)
            extracted = self.extractor.extract(response.text)

            sample_answers.append(extracted.answer)
            raw_responses.append(response.text)
            total_input_tokens += response.input_tokens
            total_output_tokens += response.output_tokens

        latency = (time.time() - start_time) * 1000

        # Majority voting
        answer_counts = Counter(sample_answers)
        majority_answer = answer_counts.most_common(1)[0][0]

        return MethodResult(
            method="self_consistency",
            predicted_answer=majority_answer,
            extracted_answer=majority_answer,
            is_correct=False,
            raw_response="\n---\n".join(raw_responses[:3]),  # Keep first 3 for brevity
            latency_ms=latency,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            num_samples=self.config.sc_samples,
            sample_answers=sample_answers,
        )

    # ------------------------------------------------------------------
    # Method 4: RAG (Knowledge Injection)
    # ------------------------------------------------------------------
    def run_rag(self, question: str) -> MethodResult:
        """
        RAG: Knowledge injection without structured prompting.

        Retrieves relevant facts from the knowledge base and prepends
        them to a CoT prompt. This tests whether raw knowledge injection
        (without structured reasoning guidance) helps.

        Args:
            question: The yes/no question to answer

        Returns:
            MethodResult with RAG-augmented prediction
        """
        start_time = time.time()

        # Get base CoT prompt
        messages = self.prompt_builder.build_baseline_prompt(
            question, question_type="reasoning"
        )

        # Inject knowledge into user message
        user_content = messages[-1]["content"]
        augmented_content, facts = self.knowledge_retriever.inject_knowledge_into_prompt(
            question, user_content
        )
        messages[-1]["content"] = augmented_content

        response = self.client.generate(messages, self.gen_config)
        latency = (time.time() - start_time) * 1000

        extracted = self.extractor.extract(response.text)

        return MethodResult(
            method="rag",
            predicted_answer=response.text,
            extracted_answer=extracted.answer,
            is_correct=False,
            raw_response=response.text,
            latency_ms=latency,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

    # ------------------------------------------------------------------
    # Method 5: KB+SC+Step1/2 (Proposed Method)
    # ------------------------------------------------------------------
    def run_kb_sc_step(self, question: str) -> MethodResult:
        """
        KB+SC+Step1/2: Structured knowledge prompting with self-consistency.

        This is the proposed method that combines:
        1. Knowledge retrieval (KB) for factual grounding
        2. Structured two-step prompting (Step1: analyze, Step2: conclude)
        3. Self-consistency (SC) with majority voting for robustness

        The structured approach forces the model to:
        - First analyze the question with relevant facts
        - Then draw a conclusion from that analysis
        This prevents the model from jumping to conclusions.

        Args:
            question: The yes/no question to answer

        Returns:
            MethodResult with structured KB+SC prediction
        """
        start_time = time.time()
        sample_answers = []
        total_input_tokens = 0
        total_output_tokens = 0
        raw_responses = []

        sc_config = GenerationConfig(
            model=self.gen_config.model,
            temperature=0.8,
            max_tokens=self.gen_config.max_tokens,
        )

        # Retrieve knowledge
        facts = self.knowledge_retriever.retrieve_relevant_facts(question)
        fact_text = ""
        if facts:
            fact_text = "\n".join(f"• {f.fact}" for f in facts[:3])

        for _ in range(self.config.sc_samples):
            # Step 1: Analyze with knowledge
            knowledge_section = ""
            if fact_text:
                knowledge_section = "Relevant facts:\n" + fact_text + "\n"

            step1_prompt = (
                f"Question: {question}\n\n"
                f"{knowledge_section}\n"
                "Analyze this question. Consider what facts are relevant and "
                "what logical steps are needed to answer it. Be specific about your reasoning."
            )

            step1_messages = [
                {
                    "role": "system",
                    "content": "You are a careful reasoner. Analyze questions thoroughly before answering.",
                },
                {"role": "user", "content": step1_prompt},
            ]

            step1_response = self.client.generate(step1_messages, sc_config)
            total_input_tokens += step1_response.input_tokens
            total_output_tokens += step1_response.output_tokens

            # Step 2: Conclude from analysis
            step2_prompt = f"""Based on this analysis:

{step1_response.text}

What is the final answer to: {question}

Answer with only 'Yes' or 'No'."""

            step2_messages = [
                {
                    "role": "system",
                    "content": "Provide a clear Yes/No answer based on the analysis.",
                },
                {"role": "user", "content": step2_prompt},
            ]

            step2_response = self.client.generate(step2_messages, sc_config)
            total_input_tokens += step2_response.input_tokens
            total_output_tokens += step2_response.output_tokens

            extracted = self.extractor.extract(step2_response.text)
            sample_answers.append(extracted.answer)
            raw_responses.append(
                f"Step1: {step1_response.text[:200]}...\nStep2: {step2_response.text}"
            )

        latency = (time.time() - start_time) * 1000

        # Majority voting
        answer_counts = Counter(sample_answers)
        majority_answer = answer_counts.most_common(1)[0][0]

        return MethodResult(
            method="kb_sc_step",
            predicted_answer=majority_answer,
            extracted_answer=majority_answer,
            is_correct=False,
            raw_response="\n---\n".join(raw_responses[:3]),
            latency_ms=latency,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            num_samples=self.config.sc_samples,
            sample_answers=sample_answers,
        )

    # ------------------------------------------------------------------
    # Main Run Loop
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        """
        Run the full benchmark across all methods.

        Returns:
            Dictionary with all results and statistics
        """
        dataset = self.load_dataset()
        methods = [
            ("zero_shot", self.run_zero_shot),
            ("chain_of_thought", self.run_chain_of_thought),
            ("self_consistency", self.run_self_consistency),
            ("rag", self.run_rag),
            ("kb_sc_step", self.run_kb_sc_step),
        ]

        # Get already-processed question IDs from checkpoint
        processed_ids = {r.question_id for r in self.results}
        questions_to_process = [
            q for q in dataset if q["id"] not in processed_ids
        ]

        if not questions_to_process:
            logger.info("All questions already processed (from checkpoint)")
        else:
            logger.info(
                f"Processing {len(questions_to_process)} questions "
                f"({len(processed_ids)} already done)"
            )

        # Run benchmark
        for question_data in tqdm(questions_to_process, desc="Benchmarking"):
            qid = question_data["id"]
            question = question_data["question"]
            ground_truth = question_data["answer"].lower().strip()

            result = QuestionResult(
                question_id=qid,
                question=question,
                ground_truth=ground_truth,
            )

            # Run all methods on this question
            for method_name, method_fn in methods:
                try:
                    method_result = method_fn(question)
                    method_result.is_correct = self.extractor.check_answer(
                        method_result.extracted_answer, ground_truth
                    )
                    result.method_results[method_name] = method_result
                except Exception as e:
                    logger.error(f"Error in {method_name} for {qid}: {e}")
                    result.method_results[method_name] = MethodResult(
                        method=method_name,
                        predicted_answer="ERROR",
                        extracted_answer="",
                        is_correct=False,
                        raw_response=str(e),
                        latency_ms=0,
                        input_tokens=0,
                        output_tokens=0,
                    )

            self.results.append(result)

            # Save checkpoint every 10 questions
            if len(self.results) % 10 == 0:
                self._save_checkpoint()

        # Final checkpoint save
        self._save_checkpoint()

        # Compute statistics
        stats = self.compute_statistics()

        # Save final results
        self._save_results(stats)

        return stats

    # ------------------------------------------------------------------
    # Statistical Analysis
    # ------------------------------------------------------------------
    def compute_statistics(self) -> Dict[str, Any]:
        """
        Compute comprehensive statistics for all methods.

        Returns:
            Dictionary with accuracy, confidence intervals, and tests
        """
        method_names = [
            "zero_shot",
            "chain_of_thought",
            "self_consistency",
            "rag",
            "kb_sc_step",
        ]

        stats = {
            "num_questions": len(self.results),
            "methods": {},
            "mcnemar_tests": {},
        }

        # Compute per-method statistics
        for method in method_names:
            correct = sum(
                1
                for r in self.results
                if method in r.method_results and r.method_results[method].is_correct
            )
            total = sum(
                1
                for r in self.results if method in r.method_results
            )
            accuracy = correct / total if total > 0 else 0.0

            # 95% confidence interval (Wilson score interval)
            if total > 0:
                z = 1.96
                n = total
                p_hat = correct / n
                denominator = 1 + z**2 / n
                center = (p_hat + z**2 / (2 * n)) / denominator
                margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denominator
                ci_low = max(0, center - margin)
                ci_high = min(1, center + margin)
            else:
                ci_low, ci_high = 0.0, 0.0

            # Latency stats
            latencies = [
                r.method_results[method].latency_ms
                for r in self.results
                if method in r.method_results
            ]

            stats["methods"][method] = {
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "ci_95_low": ci_low,
                "ci_95_high": ci_high,
                "mean_latency_ms": np.mean(latencies) if latencies else 0,
                "std_latency_ms": np.std(latencies) if latencies else 0,
            }

        # McNemar's test between all pairs
        for i, m1 in enumerate(method_names):
            for m2 in method_names[i + 1 :]:
                # Build contingency table
                a = b = c = d = 0
                for r in self.results:
                    if m1 in r.method_results and m2 in r.method_results:
                        m1_correct = r.method_results[m1].is_correct
                        m2_correct = r.method_results[m2].is_correct
                        if m1_correct and m2_correct:
                            a += 1
                        elif m1_correct and not m2_correct:
                            b += 1
                        elif not m1_correct and m2_correct:
                            c += 1
                        else:
                            d += 1

                # McNemar's test (with continuity correction)
                if b + c > 0:
                    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
                    p_value = 1 - chi2_dist.cdf(chi2, df=1)
                else:
                    chi2 = 0.0
                    p_value = 1.0

                stats["mcnemar_tests"][f"{m1}_vs_{m2}"] = {
                    "chi2": chi2,
                    "p_value": p_value,
                    "significant_005": p_value < 0.05,
                    "contingency": {"a": a, "b": b, "c": c, "d": d},
                }

        return stats

    # ------------------------------------------------------------------
    # Output Generation
    # ------------------------------------------------------------------
    def _save_results(self, stats: Dict[str, Any]):
        """Save results to JSON and generate report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON results
        json_path = self.output_dir / f"benchmark_results_{timestamp}.json"
        output = {
            "timestamp": timestamp,
            "config": asdict(self.config),
            "statistics": stats,
            "per_question": [
                {
                    "question_id": r.question_id,
                    "question": r.question,
                    "ground_truth": r.ground_truth,
                    "methods": {
                        k: {
                            "extracted": v.extracted_answer,
                            "correct": v.is_correct,
                            "latency_ms": v.latency_ms,
                        }
                        for k, v in r.method_results.items()
                    },
                }
                for r in self.results
            ],
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        logger.info(f"Results saved to {json_path}")

        # Save latest symlink
        latest_path = self.output_dir / "benchmark_results_latest.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

        # Generate markdown report
        report_path = self.output_dir / "benchmark_report.md"
        self._generate_report(stats, report_path)
        logger.info(f"Report saved to {report_path}")

        # Print summary table
        self._print_summary(stats)

    def _generate_report(self, stats: Dict[str, Any], path: Path):
        """Generate markdown benchmark report."""
        lines = [
            "# Benchmark Report",
            "",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Model:** {self.config.model_name}",
            f"**Dataset:** {self.config.dataset_path}",
            f"**Questions:** {stats['num_questions']}",
            f"**SC Samples:** {self.config.sc_samples}",
            "",
            "## Results Summary",
            "",
            "| Method | Accuracy | 95% CI | Correct | Latency (ms) |",
            "|--------|----------|--------|---------|--------------|",
        ]

        method_labels = {
            "zero_shot": "Zero-Shot",
            "chain_of_thought": "Chain-of-Thought",
            "self_consistency": "Self-Consistency",
            "rag": "RAG",
            "kb_sc_step": "KB+SC+Step1/2",
        }

        for method, label in method_labels.items():
            s = stats["methods"][method]
            lines.append(
                f"| {label} | {s['accuracy']:.1%} | "
                f"[{s['ci_95_low']:.1%}, {s['ci_95_high']:.1%}] | "
                f"{s['correct']}/{s['total']} | "
                f"{s['mean_latency_ms']:.0f} ± {s['std_latency_ms']:.0f} |"
            )

        lines.extend([
            "",
            "## Statistical Tests (McNemar's)",
            "",
            "| Comparison | χ² | p-value | Significant (α=0.05) |",
            "|------------|-----|---------|---------------------|",
        ])

        for test_name, test_data in stats["mcnemar_tests"].items():
            sig = "Yes" if test_data["significant_005"] else "No"
            lines.append(
                f"| {test_name} | {test_data['chi2']:.3f} | "
                f"{test_data['p_value']:.4f} | {sig} |"
            )

        lines.extend([
            "",
            "## Methodology",
            "",
            "- **Zero-Shot:** Direct answer without reasoning chain",
            "- **Chain-of-Thought:** Step-by-step reasoning before answering",
            "- **Self-Consistency:** 5 samples with majority voting (temp=0.8)",
            "- **RAG:** Knowledge retrieval + CoT prompt",
            "- **KB+SC+Step1/2:** Structured 2-step prompting with knowledge + SC",
            "",
            "All methods use the same UnifiedAnswerExtractor for fair evaluation.",
            "Confidence intervals use Wilson score interval.",
            "",
            "## Per-Question Results",
            "",
        ])

        for r in self.results:
            lines.append(f"### {r.question_id}: {r.question}")
            lines.append(f"**Ground Truth:** {r.ground_truth}")
            lines.append("")
            lines.append("| Method | Extracted | Correct |")
            lines.append("|--------|-----------|---------|")
            for method in method_labels:
                if method in r.method_results:
                    mr = r.method_results[method]
                    check = "✓" if mr.is_correct else "✗"
                    lines.append(
                        f"| {method_labels[method]} | {mr.extracted_answer} | {check} |"
                    )
            lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _print_summary(self, stats: Dict[str, Any]):
        """Print summary table to console."""
        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 70)
        print(f"Model: {self.config.model_name}")
        print(f"Questions: {stats['num_questions']}")
        print(f"Mock Mode: {self.config.mock}")
        print("-" * 70)

        method_labels = {
            "zero_shot": "Zero-Shot",
            "chain_of_thought": "Chain-of-Thought",
            "self_consistency": "Self-Consistency",
            "rag": "RAG",
            "kb_sc_step": "KB+SC+Step1/2",
        }

        print(f"\n{'Method':<20} {'Accuracy':>10} {'95% CI':>20} {'Latency':>15}")
        print("-" * 70)

        for method, label in method_labels.items():
            s = stats["methods"][method]
            ci = f"[{s['ci_95_low']:.1%}, {s['ci_95_high']:.1%}]"
            latency = f"{s['mean_latency_ms']:.0f}ms"
            print(f"{label:<20} {s['accuracy']:>9.1%} {ci:>20} {latency:>15}")

        print("\n" + "=" * 70)
        print("McNemar's Tests (p < 0.05 = significant)")
        print("-" * 70)

        for test_name, test_data in stats["mcnemar_tests"].items():
            sig = "***" if test_data["p_value"] < 0.001 else (
                "**" if test_data["p_value"] < 0.01 else (
                    "*" if test_data["p_value"] < 0.05 else "ns"
                )
            )
            print(f"{test_name:<30} p={test_data['p_value']:.4f} {sig}")

        print("=" * 70 + "\n")


# ============================================================================
# CLI Interface
# ============================================================================

def parse_args() -> BenchmarkConfig:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run benchmark evaluation for LLM reasoning strategies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Mock mode (no API needed)
    python scripts/run_benchmark.py --mock --num-questions 5

    # Full run with API
    python scripts/run_benchmark.py --dataset data/datasets/benchmark_dataset.json

    # Resume from checkpoint
    python scripts/run_benchmark.py --resume --checkpoint benchmark_results/checkpoint.json
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/datasets/benchmark_dataset.json",
        help="Path to dataset JSON file (default: data/datasets/benchmark_dataset.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mimo-v2.5-pro",
        help="Model name (default: mimo-v2.5-pro)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation (default: 0.7)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Max tokens for generation (default: 1024)",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=None,
        help="Limit number of questions (default: all)",
    )
    parser.add_argument(
        "--sc-samples",
        type=int,
        default=5,
        help="Number of samples for self-consistency (default: 5)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock client (no API credits needed)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Output directory (default: benchmark_results)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file for resuming",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="mimo",
        choices=["mimo", "nvidia"],
        help="API provider: 'mimo' (Xiaomi MiMo) or 'nvidia' (NVIDIA NIM)",
    )

    args = parser.parse_args()

    return BenchmarkConfig(
        dataset_path=args.dataset,
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        num_questions=args.num_questions,
        sc_samples=args.sc_samples,
        mock=args.mock,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        resume=args.resume,
        seed=args.seed,
        provider=args.provider,
    )


def main():
    """Main entry point."""
    config = parse_args()

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(
        Path(config.output_dir) / "benchmark.log",
        level="DEBUG",
        rotation="10 MB",
    )

    logger.info("Starting benchmark run")
    logger.info(f"Config: {asdict(config)}")

    runner = BenchmarkRunner(config)
    stats = runner.run()

    logger.info("Benchmark complete!")
    return stats


if __name__ == "__main__":
    main()
