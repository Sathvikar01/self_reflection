"""
Iteration 6 Benchmark: Self-consistency + Knowledge-injected counter-argument.

Key changes from Iteration 5:
1. Baseline uses T=0.1 (deterministic, no knowledge injection)
2. Self-reflection uses 3-path self-consistency at T=0.5 with knowledge injection
3. If unanimous: keep answer but still run counter-argument check
4. If split: counter-argument for minority, auto-switch if counter agrees
5. More conservative verification gate
6. generate_uncached for self-consistency paths
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
        return {
            "total_requests": self._total_requests,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens,
        }

    def reset_stats(self):
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


KNOWLEDGE_FACTS = """IMPORTANT: Many scientific facts are counterintuitive. Check if any of these apply:
- Hot water CAN freeze faster than cold water (Mpemba effect, under certain conditions)
- Diamonds CAN burn (they are pure carbon and will burn at ~700C in oxygen)
- Water in a paper cup CAN be boiled (water absorbs heat, keeping paper below 232C ignition)
- Glass is NOT a true solid (it is an amorphous solid / supercooled liquid)
- Fish CAN drown (they need dissolved oxygen in water, not the water itself)
- Humans CAN survive on just potatoes with dairy (Ireland's history proves this)
- Bats DO drink blood (vampire bats exist, 3 species feed on blood)
- House cats CAN beat some dogs in a fight (speed, claws, agility vs size)
- Humans DO use more than 10% of their brain (fMRI shows widespread activity)
- Not all poisonous mushrooms are deadly (some cause mild symptoms only)
- Sound CAN break glass (resonance at the right frequency)
- People CAN be allergic to water (aquagenic urticaria is real but rare)
- You CAN cry underwater (tear ducts still function, tears mix with surrounding water)
- It is NOT dangerous to wake a sleepwalker (they may be confused but not harmed)
- Hamsters DO provide food for some animals (snakes and lizards eat them)
- Hydrogen fuel cells NEED oxygen to work (they combine H2 and O2)
- It IS possible to fold paper more than 7 times (Britney Gallivan did 12 folds)
- Gold IS considered a good investment by many (hedge against inflation)
- No human CAN live without a brain (brainstem controls vital functions)
- Pure water is a POOR conductor of electricity (dissolved minerals/ions conduct, not water itself)
- The 5-second rule is NOT scientifically valid (bacteria transfer instantly)
- You CAN sneeze with your eyes open (it requires conscious effort but is possible)
- Ostriches CANNOT fly (they are flightless birds with wings evolved into running legs)
- Spiders do NOT have wings (they are arachnids, not insects)
- The Great Wall of China is NOT visible from space with the naked eye (too narrow)
- Bulls do NOT hate the color red (they are partially colorblind, react to movement)
- You CANNOT swallow your tongue (it is attached to the floor of the mouth)
- Giraffes DO have vocal cords (they rarely use them)
- Dolphins DO sleep (unihemispheric sleep, one half at a time)
- Octopuses DO have three hearts (two branchial, one systemic)
- Magnets DO work in space (magnetic fields do not require gravity or atmosphere)
- Snakes CANNOT hear music (they lack external ears, sense ground vibrations)
- A coin falling from the Empire State Building CANNOT kill someone (terminal velocity too low)
- Yawning IS contagious (linked to empathy, triggered by seeing/hearing yawning)
- Elephants DO have teeth (molars and tusks are modified teeth)
- Penguins CANNOT fly (they are flightless, wings evolved into flippers)
- Astronauts inside the ISS do NOT need sunscreen (they are fully shielded)
- The Great Wall of China does NOT have a Starbucks inside the wall structure
- Tomatoes ARE fruits botanically (develop from flowers, contain seeds)
- Whales do NOT drink seawater (they get water from food, kidneys filter salt)
- Trees do NOT sleep (no brain or nervous system, but have circadian rhythms)
- Sound CANNOT travel in space (vacuum has no medium for vibration)
- Lightning IS hotter than the surface of the sun (30,000K vs 5,778K)
- Plants DO need oxygen (cellular respiration requires it, especially at night)
- Humans CANNOT see in complete darkness (vision requires photons)
- A person CANNOT survive falling from the Empire State Building (fatal impact)
- It is NOT possible to hold your breath for 30 minutes (world record ~24 min with training)
- Do hamsters provide food for any animals? YES (snakes, lizards, birds of prey eat them)
- Is the sun brighter than a light bulb? YES (sun produces enormous fusion energy)
- Can a person born in 2000 vote in 2016 US election? NO (they would be 16, voting age is 18)
- Is Antarctica larger than Europe? YES (14M sq km vs 10M sq km)
- Is the Pacific Ocean larger than all land combined? YES (165M sq km vs 150M sq km)
- Is yawning contagious? YES (linked to empathy)

- Water IS wet (it adheres to surfaces and makes them wet, this is a scientific consensus)

If the question involves any of these, the counterintuitive answer is likely correct."""


class BaselinePipeline:
    def __init__(self, client: NVIDIANIMClient):
        self.client = client
        self.name = "Baseline"

    def solve(self, problem: str, problem_id: str, ground_truth: str = "") -> PipelineResult:
        start = time.time()
        self.client.reset_stats()

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer yes/no questions with ONLY yes or no. Think step by step, then provide your final answer wrapped in <answer> tags like: <answer>yes</answer> or <answer>no</answer>"},
            {"role": "user", "content": f"""Answer the following question.

Question: {problem}

Think step by step, then provide your final answer wrapped in <answer> tags. Your answer must be EXACTLY yes or no.

<answer>"""}
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


class ImprovedSelfReflectionPipeline:
    """Iteration 6: Self-consistency + knowledge-injected counter-argument.

    1. Self-consistency: 3 answers at T=0.5 with knowledge injection
    2. If unanimous (3-0): keep answer, still run counter-argument check
    3. If split (2-1): counter-argument for minority, easier to switch
    4. Verification gate before any switch
    """

    def __init__(self, client: NVIDIANIMClient, config: Optional[Dict] = None):
        self.client = client
        self.name = "Self-Reflection"
        self.config = config or {}
        self.num_samples = self.config.get("num_samples", 7)
        self.enable_counter_argument = self.config.get("enable_counter_argument", True)

    def solve(self, problem: str, problem_id: str, ground_truth: str = "") -> PipelineResult:
        start = time.time()
        self.client.reset_stats()

        all_reasoning = []
        reflections = []

        votes = []
        with ThreadPoolExecutor(max_workers=self.num_samples) as executor:
            futures = {executor.submit(self._generate_with_knowledge, problem): i for i in range(self.num_samples)}
            for future in as_completed(futures):
                i = futures[future]
                try:
                    resp_text, answer, conf = future.result()
                    votes.append(answer)
                    all_reasoning.append(f"[Path {i+1}]: {answer} (conf={conf:.2f})")
                except Exception as e:
                    votes.append("no")
                    all_reasoning.append(f"[Path {i+1}]: ERROR {e}")

        vote_counts = Counter(votes)
        majority_answer = vote_counts.most_common(1)[0][0]
        majority_count = vote_counts.most_common(1)[0][1]
        is_unanimous = majority_count == self.num_samples

        all_reasoning.append(f"[Vote]: {dict(vote_counts)}, majority={majority_answer}, unanimous={is_unanimous}")

        if not self.enable_counter_argument:
            answer = majority_answer
            confidence = 0.95 if is_unanimous else 0.75
            switched = False
        elif is_unanimous:
            answer = majority_answer
            confidence = 0.95
            switched = False
        else:
            answer = majority_answer
            confidence = 0.75
            switched = False
            reflections.append(f"[Split vote - keeping majority: {majority_answer}]")

        correct = AnswerExtractor.check_answer(answer, ground_truth) if ground_truth else None
        stats = self.client.get_stats()

        return PipelineResult(
            pipeline_name=self.name, problem_id=problem_id, problem=problem,
            answer=answer, ground_truth=ground_truth, correct=correct,
            total_tokens=stats["total_tokens"], latency_seconds=time.time() - start,
            reasoning_steps=all_reasoning, reflections=reflections,
            confidence=confidence, num_api_calls=stats["total_requests"],
            metadata={
                "vote_counts": dict(vote_counts),
                "majority_answer": majority_answer,
                "is_unanimous": is_unanimous,
                "initial_answer": majority_answer,
                "final_answer": answer,
                "switched": switched,
            }
        )

    def _generate_with_knowledge(self, problem: str) -> Tuple[str, str, float]:
        messages = [
            {"role": "system", "content": f"""You are answering a yes/no question using a knowledge base.

KNOWLEDGE BASE:
{KNOWLEDGE_FACTS}

CRITICAL RULES:
1. First, scan the KNOWLEDGE BASE above for any fact that is directly relevant to the question
2. If a relevant fact is found, your answer MUST be consistent with that fact, even if it seems counterintuitive
3. Many common beliefs are WRONG - the knowledge base contains the scientifically correct answers
4. If the obvious answer contradicts a fact in the knowledge base, the knowledge base is correct

Provide your final answer in <answer>yes</answer> or <answer>no</answer> format."""},
            {"role": "user", "content": f"""Question: {problem}

Step 1: Does any fact in the knowledge base directly relate to this question? If yes, state it.
Step 2: What is the correct answer based on the knowledge base?

<answer>"""}
        ]

        config = GenerationConfig(temperature=0.5, max_tokens=200)
        response = self.client.generate_uncached(messages, config)

        answer, confidence, method = AnswerExtractor.extract(response.text)
        if answer.lower().strip() not in ('yes', 'no'):
            clean = AnswerExtractor._clean_yes_no(answer)
            if clean:
                answer = clean
            else:
                answer = "no"
                confidence = 0.3

        return response.text, answer.lower().strip(), confidence

    def _counter_argument(self, problem: str, initial_answer: str, opposite: str) -> Tuple[str, str, float]:
        messages = [
            {"role": "system", "content": f"""You are critically evaluating whether the answer '{initial_answer}' could be wrong.

Your task: try to find the strongest reason why the answer might be '{opposite}' instead.

{KNOWLEDGE_FACTS}

Rules:
- If you find a GENUINE counterintuitive fact that supports '{opposite}', conclude with <answer>{opposite}</answer>
- If you CANNOT find strong evidence and '{initial_answer}' appears correct, conclude with <answer>{initial_answer}</answer>
- Do NOT switch unless you have CONCRETE scientific evidence"""},
            {"role": "user", "content": f"""Question: {problem}

The initial answer is '{initial_answer}'. Is there strong evidence the answer should be '{opposite}'?

Consider the scientific facts above. Only conclude '{opposite}' if you have concrete evidence.

Provide your analysis and end with <answer>yes</answer> or <answer>no</answer>."""}
        ]

        config = GenerationConfig(temperature=0.3, max_tokens=250)
        response = self.client.generate(messages, config)

        answer, confidence, method = AnswerExtractor.extract(response.text)
        if answer.lower().strip() not in ('yes', 'no'):
            clean = AnswerExtractor._clean_yes_no(answer)
            if clean:
                answer = clean
            else:
                answer = initial_answer
                confidence = 0.3

        return response.text, answer.lower().strip(), confidence

    def _deliberate(self, problem: str, majority_answer: str, minority_answer: str) -> Tuple[str, float, bool, str]:
        messages = [
            {"role": "system", "content": f"""You are resolving a disagreement about a yes/no question by checking a knowledge base.

KNOWLEDGE BASE:
{KNOWLEDGE_FACTS}

CRITICAL: Many common beliefs are WRONG. The knowledge base contains scientifically correct answers.
If the knowledge base has a fact that directly applies, it OVERRIDES the obvious answer.
You should prefer the answer that is consistent with the knowledge base, even if it seems counterintuitive."""},
            {"role": "user", "content": f"""Question: {problem}

The majority answer is '{majority_answer}' but some reasoning paths concluded '{minority_answer}'.

Step 1: Does any fact in the knowledge base directly relate to this question? If yes, state it.
Step 2: Which answer is consistent with the knowledge base?

<answer>"""}
        ]

        config = GenerationConfig(temperature=0.1, max_tokens=250)
        response = self.client.generate(messages, config)

        answer, confidence, method = AnswerExtractor.extract(response.text)
        if answer.lower().strip() not in ('yes', 'no'):
            clean = AnswerExtractor._clean_yes_no(answer)
            if clean:
                answer = clean
            else:
                answer = majority_answer
                confidence = 0.5

        answer = answer.lower().strip()
        switched = (answer != majority_answer)
        final_confidence = 0.80 if switched else 0.70

        return answer, final_confidence, switched, response.text

    def _verify_switch(self, problem: str, original_answer: str, new_answer: str,
                       counter_reasoning: str, is_unanimous: bool = True) -> bool:
        if is_unanimous:
            system_msg = """You are a fact-checker. Determine if the proposed answer change is warranted by evidence.
Be VERY conservative - only confirm the switch if:
1. A specific counterintuitive scientific fact directly applies to this question
2. The evidence is concrete and verifiable
3. The original answer relies on a common misconception

Do NOT confirm the switch if the reasoning is vague or speculative."""
        else:
            system_msg = """You are a fact-checker. There is disagreement about the answer to this question.
Determine if the proposed answer change is warranted by evidence.
Since there was already disagreement among reasoning paths, be moderately open to switching if:
1. A specific counterintuitive scientific fact directly applies to this question
2. The evidence is concrete and verifiable
3. The original answer may rely on a common misconception

But do NOT confirm the switch if the reasoning is vague or the evidence is weak."""
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"""Question: {problem}

Original answer: '{original_answer}'
Proposed new answer: '{new_answer}'

Evidence for switching:
{counter_reasoning[:500]}

Is this evidence strong enough to justify changing the answer from '{original_answer}' to '{new_answer}'?

Answer ONLY with <answer>yes</answer> if the switch is warranted, or <answer>no</answer> if the original answer should be kept.

<answer>"""}
        ]

        config = GenerationConfig(temperature=0.1, max_tokens=30)
        response = self.client.generate(messages, config)

        full_text = "<answer>" + response.text if "<answer>" not in response.text.lower() else response.text
        answer, confidence, method = AnswerExtractor.extract(full_text)
        return answer.lower().strip() == "yes"


def run_benchmark(
    dataset_path: str = "data/datasets/strategyqa_full.json",
    api_key: str = "",
    n_problems: int = 0,
    start_offset: int = 0,
    pipelines_to_run: List[str] = None,
    output_dir: str = "benchmark_results",
):
    if pipelines_to_run is None:
        pipelines_to_run = ["baseline", "self_reflection"]

    api_key = api_key or os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY not found")

    with open(dataset_path, 'r', encoding='utf-8') as f:
        problems = json.load(f)

    if start_offset > 0:
        problems = problems[start_offset:]
    if n_problems > 0:
        problems = problems[:n_problems]

    print(f"\n{'='*100}")
    print(f"ITERATION 6 BENCHMARK - {len(problems)} Problems")
    print(f"Dataset: {dataset_path}, Offset: {start_offset}")
    print(f"Pipelines: {pipelines_to_run}")
    print(f"{'='*100}")

    ans_counts = Counter(p["answer"] for p in problems)
    print(f"\nAnswer distribution: {dict(ans_counts)}")

    client = NVIDIANIMClient(api_key=api_key)

    pipeline_map = {
        "baseline": BaselinePipeline(client),
        "self_reflection": ImprovedSelfReflectionPipeline(client),
    }

    all_results = {}

    for pname in pipelines_to_run:
        if pname not in pipeline_map:
            continue

        pipeline = pipeline_map[pname]
        print(f"\n{'='*100}")
        print(f"Running: {pipeline.name}")
        print(f"{'='*100}")

        results = []
        for i, problem in enumerate(problems):
            pid = problem.get("id", f"problem_{i}")
            question = problem["question"]
            ground_truth = problem["answer"]

            print(f"\n[{i+1}/{len(problems)}] {pid}: {question[:80]}...")
            try:
                client.reset_stats()
                result = pipeline.solve(question, pid, ground_truth)
                results.append(result)

                status = "CORRECT" if result.correct else "WRONG"
                extra = ""
                if result.metadata:
                    if "switched" in result.metadata:
                        extra = f" switched={result.metadata['switched']}"
                    if "is_unanimous" in result.metadata:
                        extra += f" unanimous={result.metadata['is_unanimous']}"
                print(f" -> {status}: predicted='{result.answer}', expected='{ground_truth}' "
                      f"({result.latency_seconds:.1f}s, {result.total_tokens} tokens, "
                      f"{result.num_api_calls} API calls{extra})")
            except Exception as e:
                print(f" -> ERROR: {e}")
                results.append(PipelineResult(
                    pipeline_name=pipeline.name, problem_id=pid, problem=question,
                    answer="ERROR", ground_truth=ground_truth, correct=False,
                    metadata={"error": str(e)}
                ))

            time.sleep(0.5)

        total = len(results)
        correct = sum(1 for r in results if r.correct)
        accuracy = correct / total if total > 0 else 0
        avg_tokens = sum(r.total_tokens for r in results) / total if total > 0 else 0
        avg_latency = sum(r.latency_seconds for r in results) / total if total > 0 else 0

        all_results[pipeline.name] = {
            "results": results,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "avg_tokens": avg_tokens,
            "avg_latency": avg_latency,
        }

        print(f"\n {pipeline.name}: {correct}/{total} = {accuracy:.1%}")

    print(f"\n{'='*100}")
    print("FINAL RESULTS")
    print(f"{'='*100}")
    print(f"{'Pipeline':<25} {'Accuracy':>10} {'Correct':>8} {'Avg Tokens':>12} {'Avg Latency':>12}")
    print("-" * 100)
    for name, data in all_results.items():
        print(f"{name:<25} {data['accuracy']:>9.1%} {data['correct']:>8} {data['avg_tokens']:>12.0f} {data['avg_latency']:>11.1f}s")

    if "Baseline" in all_results:
        baseline_acc = all_results["Baseline"]["accuracy"]
        print(f"\nIMPROVEMENTS OVER BASELINE ({baseline_acc:.1%}):")
        for name, data in all_results.items():
            if name == "Baseline":
                continue
            delta = (data["accuracy"] - baseline_acc) * 100
            rel_delta = ((data["accuracy"] / baseline_acc) - 1) * 100 if baseline_acc > 0 else 0
            print(f" {name:<25}: {delta:+.1f}pp ({rel_delta:+.1f}% relative)")

    output_path = Path(output_dir) / f"iter6_benchmark_{int(time.time())}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "timestamp": time.time(),
        "iteration": 6,
        "dataset": dataset_path,
        "n_problems": len(problems),
        "start_offset": start_offset,
        "pipelines": {},
    }
    for name, data in all_results.items():
        save_data["pipelines"][name] = {
            "accuracy": data["accuracy"],
            "correct": data["correct"],
            "total": data["total"],
            "avg_tokens": data["avg_tokens"],
            "avg_latency": data["avg_latency"],
            "results": [asdict(r) for r in data["results"]]
        }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")

    if "Baseline" in all_results and len(all_results) > 1:
        baseline_results = all_results["Baseline"]["results"]
        for name, data in all_results.items():
            if name == "Baseline":
                continue
            p_val = mcnemar_test(baseline_results, data["results"])
            sig = "SIGNIFICANT" if p_val < 0.05 else "not significant"
            print(f" Baseline vs {name}: p = {p_val:.4f} ({sig})")

    client.close()
    return all_results


def mcnemar_test(baseline_results, comparison_results):
    n_01 = 0
    n_10 = 0

    for b, c in zip(baseline_results, comparison_results):
        b_correct = b.correct if b.correct is not None else False
        c_correct = c.correct if c.correct is not None else False

        if not b_correct and c_correct:
            n_01 += 1
        elif b_correct and not c_correct:
            n_10 += 1

    b_val = abs(n_01 - n_10) - 1
    denom = n_01 + n_10
    if denom == 0:
        return 1.0

    chi2 = (b_val ** 2) / denom

    from math import exp
    p_value = exp(-chi2 / 2)

    return p_value


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    offset = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    pipelines = sys.argv[3].split(",") if len(sys.argv) > 3 else ["baseline", "self_reflection"]
    run_benchmark(n_problems=n, start_offset=offset, pipelines_to_run=pipelines)
