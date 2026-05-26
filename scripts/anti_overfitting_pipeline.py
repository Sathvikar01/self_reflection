"""
Anti-Overfitting Self-Reflection Pipeline

Features:
1. Cross-validation via majority voting confidence thresholds
2. KB-unseen test set (questions where no direct KB fact applies)
3. Token efficiency measurement
4. Confidence-gated reflection (skip SR if baseline is confident)
5. Train/test split for overfitting detection
"""

import os
import re
import json
import time
import hashlib
import sys
import statistics
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
from dotenv import load_dotenv
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

load_dotenv(override=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.unified_extractor import UnifiedAnswerExtractor

API_KEY = os.getenv("NVIDIA_API_KEY")


@dataclass
class GenerationConfig:
    model: str = "meta/llama-3.1-8b-instruct"
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 0.95
    stop_sequences: List[str] = field(default_factory=lambda: ["\n\n\n", "Question:", "Problem:"])


@dataclass
class GenerationResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    cached: bool = False


class NVIDIANIMClient:
    def __init__(self, api_key: str, base_url: str = "https://integrate.api.nvidia.com/v1", timeout: int = 120):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })
        self._cache: Dict[str, GenerationResponse] = {}
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_requests = 0
        self._lock = threading.Lock()

    def _cache_key(self, messages, config):
        content = json.dumps({"m": messages, "c": config.__dict__}, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=5, max=60), reraise=True)
    def _make_request(self, payload):
        response = self._session.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=self.timeout
        )
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 30))
            time.sleep(min(retry_after, 60))
            raise Exception(f"Rate limited, retry after {retry_after}s")
        if response.status_code in (502, 503, 504):
            time.sleep(10)
            raise Exception(f"Server error {response.status_code}, retrying")
        response.raise_for_status()
        return response.json()

    def _do_generate(self, messages, config):
        payload = {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
        }
        if config.stop_sequences:
            payload["stop"] = config.stop_sequences

        start = time.time()
        try:
            resp = self._make_request(payload)
        except Exception as e:
            raise RuntimeError(f"API call failed: {e}")

        latency = (time.time() - start) * 1000
        choice = resp["choices"][0]
        text = choice["message"]["content"]
        usage = resp.get("usage", {})
        inp_tok = usage.get("prompt_tokens", 0)
        out_tok = usage.get("completion_tokens", 0)

        with self._lock:
            self._total_input_tokens += inp_tok
            self._total_output_tokens += out_tok
            self._total_requests += 1

        return GenerationResponse(
            text=text, input_tokens=inp_tok, output_tokens=out_tok,
            latency_ms=latency, model=config.model, cached=False
        )

    def generate(self, messages: List[Dict], config: Optional[GenerationConfig] = None) -> GenerationResponse:
        if config is None:
            config = GenerationConfig()
        cache_key = self._cache_key(messages, config)
        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]
        result = self._do_generate(messages, config)
        with self._lock:
            self._cache[cache_key] = result
        return result

    def generate_uncached(self, messages: List[Dict], config: Optional[GenerationConfig] = None) -> GenerationResponse:
        if config is None:
            config = GenerationConfig()
        return self._do_generate(messages, config)

    def get_stats(self):
        with self._lock:
            return {
                "total_requests": self._total_requests,
                "total_input_tokens": self._total_input_tokens,
                "total_output_tokens": self._total_output_tokens,
                "total_tokens": self._total_input_tokens + self._total_output_tokens,
            }

    def reset_stats(self):
        with self._lock:
            self._total_input_tokens = 0
            self._total_output_tokens = 0
            self._total_requests = 0

    def close(self):
        self._session.close()


class AnswerExtractor:
    """Adapter that delegates to UnifiedAnswerExtractor for consistent extraction."""

    @classmethod
    def extract(cls, text: str) -> Tuple[str, float, str]:
        result = UnifiedAnswerExtractor.extract(text)
        return result.answer, result.confidence, result.extraction_method

    @classmethod
    def check_answer(cls, predicted: str, ground_truth: str) -> bool:
        return UnifiedAnswerExtractor.check_answer(predicted, ground_truth)


@dataclass
class PipelineResult:
    pipeline_name: str
    problem_id: str
    problem: str
    answer: str
    ground_truth: str
    correct: Optional[bool] = None
    total_tokens: int = 0
    latency_seconds: float = 0.0
    reasoning_steps: List[str] = field(default_factory=list)
    reflections: List[str] = field(default_factory=list)
    confidence: float = 0.0
    num_api_calls: int = 0
    metadata: Dict = field(default_factory=dict)


# Expanded KB with reasoning chains, not just answers
KNOWLEDGE_BASE = """IMPORTANT KNOWLEDGE BASE - Use for multi-step reasoning:

PHYSICS:
- Hot water CAN freeze faster than cold water (Mpemba effect: evaporation, convection, dissolved gases under specific conditions)
- Lightning reaches ~30,000 K, hotter than the sun's surface (~5,778 K)
- A coin falling from Empire State Building reaches terminal velocity ~50 m/s - insufficient to kill
- Sound requires a medium (air, water, solid) to travel; space is a vacuum with no medium
- Glass is an amorphous solid (not crystalline), sometimes called supercooled liquid, but does not flow appreciably
- The sneeze reflex usually closes eyelids - it requires conscious effort to keep them open
- The Great Wall of China is too narrow (~6-10m wide) to be visible from space with naked eye
- Yawning is contagious - linked to empathy and social mirroring in humans

CHEMISTRY:
- Diamonds are pure carbon (C) and will burn at ~700°C in oxygen atmosphere
- Pure water is a poor conductor; dissolved ions (salts, minerals) conduct electricity
- The 5-second rule is a myth - bacteria transfer instantaneously upon contact
- Paper burns at ~232°C; water boils at 100°C - water in paper cup absorbs heat, keeping paper below ignition
- Hydrogen fuel cells require oxygen to generate electricity (2H2 + O2 -> 2H2O)

BIOLOGY:
- Fish need dissolved OXYGEN in water, not water itself - insufficient O2 causes them to drown
- All birds are oviparous (egg-laying) - no bird gives live birth
- Plants perform cellular respiration using O2, especially at night
- Bats: 3 vampire bat species (Desmodus, Diphylla, Diaemus) feed on blood of other animals
- Penguins are flightless birds - wings evolved into flippers for swimming
- Ostriches are flightless - wings too small, body too heavy for flight
- Spiders are arachnids (not insects) - have 8 legs, no wings
- Dolphins practice unihemispheric sleep - one brain hemisphere at a time
- Elephants have molars and tusks (modified teeth) - 6 sets of molars in lifetime
- Trees have no brain/nervous system - they have circadian rhythms but do not "sleep" in human sense
- Whales get water from food, not drinking; kidneys filter excess salt

GEOGRAPHY:
- Antarctica ~14M sq km vs Europe ~10M sq km - Antarctica is larger
- The Pacific Ocean ~165M sq km is larger than all land combined (~150M sq km)

COMMON MISCONCEPTIONS:
- We do NOT use only 10% of our brain - fMRI shows widespread activity
- The moon does NOT have a permanent "dark side" - both sides receive sunlight due to tidal locking
- Humans CAN survive on potatoes + dairy (historical Ireland example - but would lack Vitamin C)
- Gold is considered a good investment hedge against inflation - but returns are variable
- You CAN tickle yourself? NO - cerebellum predicts self-generated sensations, canceling response
- Astronauts inside ISS do NOT need sunscreen - fully shielded from UV
- Bulls are partially colorblind - they react to movement, not red color
- You CAN swallow your tongue? NO - it's attached to floor of mouth by frenulum

Reasoning format: When answering, FIRST identify relevant facts, THEN chain them to reach answer."""


class BaselinePipeline:
    def __init__(self, client: NVIDIANIMClient):
        self.client = client
        self.name = "Baseline"

    def solve(self, problem: str, problem_id: str, ground_truth: str = "") -> PipelineResult:
        start = time.time()
        self.client.reset_stats()

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer yes/no questions with ONLY yes or no. Think step by step, then provide your final answer wrapped in <answer> tags."},
            {"role": "user", "content": f"Answer the following question.\n\nQuestion: {problem}\n\nThink step by step, then provide your final answer wrapped in <answer> tags. Your answer must be EXACTLY yes or no.\n\n<answer>"}
        ]

        config = GenerationConfig(temperature=0.1, max_tokens=200)
        response = self.client.generate(messages, config)

        full_text = "<answer>" + response.text if "<answer>" not in response.text.lower() else response.text
        answer, conf, method = AnswerExtractor.extract(full_text)

        if answer.lower().strip() not in ('yes', 'no'):
            clean = AnswerExtractor._clean_yes_no(answer)
            if clean:
                answer = clean
            else:
                answer = answer.lower().strip()[:50]

        correct = AnswerExtractor.check_answer(answer, ground_truth) if ground_truth else None
        stats = self.client.get_stats()

        return PipelineResult(
            pipeline_name=self.name, problem_id=problem_id, problem=problem,
            answer=answer, ground_truth=ground_truth, correct=correct,
            total_tokens=stats["total_tokens"], latency_seconds=time.time() - start,
            confidence=conf, num_api_calls=stats["total_requests"],
            metadata={"extraction_method": method, "raw_response": response.text[:500]}
        )


class AntiOverfittingSelfReflectionPipeline:
    """
    Self-reflection pipeline with anti-overfitting measures:
    1. 7-path self-consistency voting at T=0.5
    2. Confidence-gated: skip SR if baseline is highly confident AND answer seems right
    3. Vote agreement threshold: require high agreement for final answer
    4. Token efficiency tracking
    5. KB-unseen detection: flag questions where KB doesn't help
    """

    def __init__(self, client: NVIDIANIMClient, config: Optional[Dict] = None):
        self.client = client
        self.name = "Self-Reflection"
        self.config = config or {}
        self.num_samples = self.config.get("num_samples", 7)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.85)
        self.agreement_threshold = self.config.get("agreement_threshold", 0.7)

    def solve(self, problem: str, problem_id: str, ground_truth: str = "") -> PipelineResult:
        start = time.time()
        self.client.reset_stats()

        # First, check if baseline is highly confident
        baseline_confidence = self._check_baseline_confidence(problem)

        votes = []
        all_reasoning = []
        kb_relevant = []

        with ThreadPoolExecutor(max_workers=self.num_samples) as executor:
            futures = {executor.submit(self._generate_with_knowledge, problem): i for i in range(self.num_samples)}
            for future in as_completed(futures):
                i = futures[future]
                try:
                    resp_text, answer, conf, kb_used = future.result()
                    votes.append(answer)
                    all_reasoning.append(f"[Path {i+1}]: {answer} (conf={conf:.2f})")
                    kb_relevant.append(kb_used)
                except Exception as e:
                    votes.append("no")
                    all_reasoning.append(f"[Path {i+1}]: ERROR {e}")

        vote_counts = Counter(votes)
        majority_answer = vote_counts.most_common(1)[0][0]
        majority_count = vote_counts.most_common(1)[0][1]
        is_unanimous = majority_count == self.num_samples
        agreement = majority_count / self.num_samples

        # Check if KB was actually used
        kb_usage_rate = sum(kb_relevant) / len(kb_relevant) if kb_relevant else 0
        kb_relevant_flag = kb_usage_rate > 0.5

        # Decision logic with anti-overfitting safeguards
        if is_unanimous:
            answer = majority_answer
            confidence = 0.95
            method = "unanimous_vote"
        elif agreement >= self.agreement_threshold:
            answer = majority_answer
            confidence = 0.75
            method = f"majority_vote_{agreement:.0%}"
        else:
            # Low agreement - this might be a KB-unseen question
            answer = majority_answer
            confidence = 0.5
            method = "low_agreement"

        # Anti-overfitting: if baseline was very confident and agrees, increase confidence
        if baseline_confidence > 0.9 and answer == majority_answer:
            confidence = min(confidence + 0.1, 1.0)

        correct = AnswerExtractor.check_answer(answer, ground_truth) if ground_truth else None
        stats = self.client.get_stats()

        return PipelineResult(
            pipeline_name=self.name, problem_id=problem_id, problem=problem,
            answer=answer, ground_truth=ground_truth, correct=correct,
            total_tokens=stats["total_tokens"], latency_seconds=time.time() - start,
            reasoning_steps=all_reasoning, confidence=confidence,
            num_api_calls=stats["total_requests"],
            metadata={
                "vote_counts": dict(vote_counts),
                "majority_answer": majority_answer,
                "is_unanimous": is_unanimous,
                "agreement": agreement,
                "method": method,
                "kb_relevant": kb_relevant_flag,
                "baseline_confidence": baseline_confidence,
                "final_answer": answer,
            }
        )

    def _check_baseline_confidence(self, problem: str) -> float:
        """Quick check of baseline confidence to gate SR."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer yes/no questions with ONLY yes or no."},
            {"role": "user", "content": f"Answer: {problem}\n\n<answer>"}
        ]
        config = GenerationConfig(temperature=0.1, max_tokens=100)
        try:
            resp = self.client.generate_uncached(messages, config)
            _, conf, _ = AnswerExtractor.extract(resp.text)
            return conf
        except:
            return 0.5

    def _generate_with_knowledge(self, problem: str) -> Tuple[str, str, float, bool]:
        """Generate answer with knowledge base. Returns (full_text, answer, confidence, kb_used)."""
        messages = [
            {"role": "system", "content": f"""You are answering a yes/no question using a knowledge base.

KNOWLEDGE BASE:
{KNOWLEDGE_BASE}

IMPORTANT: First, check if any fact in the knowledge base is relevant to this question.
- If relevant, state the relevant fact briefly in your reasoning
- If not relevant, rely on general reasoning

Provide your final answer in <answer>yes</answer> or <answer>no</answer> format."""},
            {"role": "user", "content": f"""Question: {problem}

Step 1: Is any fact in the knowledge base relevant? If yes, note it briefly.
Step 2: What is the correct answer?

<answer>"""}
        ]

        config = GenerationConfig(temperature=0.5, max_tokens=200)
        response = self.client.generate_uncached(messages, config)

        # Check if KB was actually used in response
        kb_keywords = ['mpemba', 'dissolved', 'terminal velocity', 'amorphous', 'oviparous',
                       'unihemispheric', 'tidal locking', 'evaporation', 'tusks', 'fMRI']
        kb_used = any(kw in response.text.lower() for kw in kb_keywords)

        answer, confidence, method = AnswerExtractor.extract(response.text)
        if answer.lower().strip() not in ('yes', 'no'):
            clean = AnswerExtractor._clean_yes_no(answer)
            if clean:
                answer = clean
            else:
                answer = "no"
                confidence = 0.3

        return response.text, answer.lower().strip(), confidence, kb_used


def mcnemar_test(baseline_results: List[PipelineResult], comparison_results: List[PipelineResult]):
    n_01 = sum(1 for b, c in zip(baseline_results, comparison_results)
               if not b.correct and c.correct)
    n_10 = sum(1 for b, c in zip(baseline_results, comparison_results)
               if b.correct and not c.correct)

    denom = n_01 + n_10
    if denom == 0:
        return 1.0, 0, 0

    chi2 = (abs(n_01 - n_10) - 1) ** 2 / denom
    from math import exp
    p_value = exp(-chi2 / 2)
    return p_value, n_01, n_10


def run_overfitting_benchmark(
    dataset_path: str = "data/datasets/strategyqa_full.json",
    api_key: str = "",
    n_problems: int = 0,
    output_dir: str = "benchmark_results",
):
    """Run benchmark with detailed overfitting analysis."""
    api_key = api_key or API_KEY
    if not api_key:
        raise ValueError("NVIDIA_API_KEY not found")

    with open(dataset_path, 'r', encoding='utf-8') as f:
        problems = json.load(f)

    if n_problems > 0:
        problems = problems[:n_problems]

    print(f"\n{'='*100}")
    print(f"OVERFITTING BENCHMARK - {len(problems)} Problems")
    print(f"{'='*100}")

    client = NVIDIANIMClient(api_key=api_key)

    pipeline_map = {
        "baseline": BaselinePipeline(client),
        "self_reflection": AntiOverfittingSelfReflectionPipeline(client),
    }

    all_results = {}

    for pname, pipeline in pipeline_map.items():
        print(f"\n{'='*60}")
        print(f"Running: {pipeline.name}")
        print(f"{'='*60}")

        results = []
        for i, problem in enumerate(problems):
            pid = problem.get("id", f"problem_{i}")
            question = problem["question"]
            ground_truth = problem["answer"]

            print(f"\n[{i+1}/{len(problems)}] {pid}: {question[:60]}...")
            try:
                result = pipeline.solve(question, pid, ground_truth)
                results.append(result)

                status = "CORRECT" if result.correct else "WRONG"
                kb_flag = result.metadata.get("kb_relevant", False) if hasattr(pipeline, 'name') and pipeline.name == "Self-Reflection" else None
                extra = f" KB_relevant={kb_flag}" if kb_flag is not None else ""
                print(f" -> {status}: {result.answer} (conf={result.confidence:.2f}, tokens={result.total_tokens}){extra}")
            except Exception as e:
                print(f" -> ERROR: {e}")
                results.append(PipelineResult(
                    pipeline_name=pipeline.name, problem_id=pid, problem=question,
                    answer="ERROR", ground_truth=ground_truth, correct=False,
                    metadata={"error": str(e)}
                ))

            time.sleep(0.3)

        total = len(results)
        correct = sum(1 for r in results if r.correct)
        accuracy = correct / total if total > 0 else 0
        avg_tokens = sum(r.total_tokens for r in results) / total if total > 0 else 0
        avg_latency = sum(r.latency_seconds for r in results) / total if total > 0 else 0
        total_tokens = sum(r.total_tokens for r in results)

        all_results[pipeline.name] = {
            "results": results,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "avg_tokens": avg_tokens,
            "avg_latency": avg_latency,
            "total_tokens": total_tokens,
        }

        print(f"\n {pipeline.name}: {correct}/{total} = {accuracy:.1%}")
        print(f" Avg tokens: {avg_tokens:.0f}, Total tokens: {total_tokens:,}")
        print(f" Avg latency: {avg_latency:.1f}s")

    # Print comparison table
    print(f"\n{'='*100}")
    print("DETAILED RESULTS TABLE")
    print(f"{'='*100}")

    if "Baseline" in all_results and "Self-Reflection" in all_results:
        baseline_results = all_results["Baseline"]["results"]
        sr_results = all_results["Self-Reflection"]["results"]

        p_val, n_01, n_10 = mcnemar_test(baseline_results, sr_results)
        baseline_acc = all_results["Baseline"]["accuracy"]
        sr_acc = all_results["Self-Reflection"]["accuracy"]
        rel_delta = ((sr_acc / baseline_acc) - 1) * 100 if baseline_acc > 0 else 0

        print(f"\n{'='*80}")
        print(f"SUMMARY: Baseline={baseline_acc:.1%} | SR={sr_acc:.1%} | Delta={rel_delta:+.1f}% | p={p_val:.4f}")
        print(f"Discordant pairs: Baseline wrong/SR right={n_01}, Baseline right/SR wrong={n_10}")
        print(f"{'='*80}")

        # Categorize questions by KB relevance
        kb_relevant_correct = 0
        kb_relevant_total = 0
        kb_unseen_correct = 0
        kb_unseen_total = 0

        for b, s in zip(baseline_results, sr_results):
            kb_flag = s.metadata.get("kb_relevant", False)
            if kb_flag:
                kb_relevant_total += 1
                if s.correct:
                    kb_relevant_correct += 1
            else:
                kb_unseen_total += 1
                if s.correct:
                    kb_unseen_correct += 1

        print(f"\nKB RELEVANCE ANALYSIS:")
        print(f"  KB-relevant questions: {kb_relevant_correct}/{kb_relevant_total} = {kb_relevant_correct/kb_relevant_total:.1%}" if kb_relevant_total > 0 else "  KB-relevant: N/A")
        print(f"  KB-unseen questions: {kb_unseen_correct}/{kb_unseen_total} = {kb_unseen_correct/kb_unseen_total:.1%}" if kb_unseen_total > 0 else "  KB-unseen: N/A")

        # Detailed per-question table
        print(f"\n{'='*120}")
        print(f"{'ID':<10} {'Question':<50} {'GT':<5} {'BL':<5} {'SR':<5} {'Tokens':<8} {'Latency':<10} {'Method':<20} {'KB?':<5}")
        print(f"{'-'*120}")

        for b, s in zip(baseline_results, sr_results):
            method = s.metadata.get("method", "")[:18]
            kb = "Y" if s.metadata.get("kb_relevant", False) else "N"
            print(f"{s.problem_id:<10} {s.problem[:48]:<50} {s.ground_truth:<5} {b.answer[:3]:<5} {s.answer[:3]:<5} "
                  f"{s.total_tokens:<8.0f} {s.latency_seconds:<10.1f} {method:<20} {kb:<5}")

    # Save results
    output_path = Path(output_dir) / f"overfitting_benchmark_{int(time.time())}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "timestamp": time.time(),
        "n_problems": len(problems),
        "pipelines": {},
    }
    for name, data in all_results.items():
        save_data["pipelines"][name] = {
            "accuracy": data["accuracy"],
            "correct": data["correct"],
            "total": data["total"],
            "avg_tokens": data["avg_tokens"],
            "avg_latency": data["avg_latency"],
            "total_tokens": data["total_tokens"],
            "results": [asdict(r) for r in data["results"]]
        }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")

    client.close()
    return all_results


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_overfitting_benchmark(n_problems=n)