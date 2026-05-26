"""Validate benchmark result JSONs for consistency and integrity.

Checks:
1. All files parse as valid JSON
2. Same questions appear across methods (consistency)
3. No fabricated results (detects simulation functions, synthetic patterns)
4. Reports anomalies: duplicate questions, missing fields, suspicious patterns

Usage:
    python scripts/validate_results.py [--results-dir benchmark_results]
"""

import json
import os
import sys
import argparse
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Any


# ---------------------------------------------------------------------------
# Suspicious pattern detection
# ---------------------------------------------------------------------------

SIMULATION_INDICATORS = [
    "simulate",
    "simulate_result",
    "mock_result",
    "fake_answer",
    "random.choice",
    "np.random",
    "random.random",
    "generate_fake",
    "synthetic",
    "dummy_result",
]

SUSPICIOUS_RESPONSE_PATTERNS = [
    r"^yes\.?\s*$",
    r"^no\.?\s*$",
    r"^this step follows",
    r"^the relevant fact",
    r"^first, i need to",
    r"^n scenario",
    r"^against the constraints",
]


def detect_simulation_artifacts(data: Any, filename: str) -> list[str]:
    """Scan raw JSON text-level content for simulation indicators."""
    warnings = []
    text = json.dumps(data).lower()

    for indicator in SIMULATION_INDICATORS:
        if indicator in text:
            warnings.append(f"[{filename}] Simulation indicator found: '{indicator}'")

    return warnings


def detect_suspicious_patterns(results: list[dict], filename: str) -> list[str]:
    """Detect suspicious answer patterns that may indicate fabricated results."""
    warnings = []

    # Check for repetitive answers
    answers = [str(r.get("answer", r.get("extracted_answer", ""))).strip().lower()
               for r in results]
    if answers:
        counter = Counter(answers)
        most_common_answer, most_common_count = counter.most_common(1)[0]
        if most_common_count > len(answers) * 0.8 and len(answers) > 5:
            warnings.append(
                f"[{filename}] Suspicious: {most_common_count}/{len(answers)} "
                f"answers are identical ('{most_common_answer}')"
            )

    # Check for all-error results (API failures)
    error_count = sum(1 for r in results
                      if str(r.get("answer", "")).upper() == "ERROR"
                      or r.get("metadata", {}).get("error", ""))
    if error_count > 0 and error_count == len(results):
        warnings.append(
            f"[{filename}] All {error_count} results are errors (likely API failure)"
        )

    # Check for identical correct/incorrect patterns across methods
    return warnings


def check_required_fields(item: dict, required: list[str], filename: str, idx: int) -> list[str]:
    """Check that required fields exist and are non-null."""
    warnings = []
    for field in required:
        if field not in item:
            warnings.append(f"[{filename}] Item {idx}: missing field '{field}'")
        elif item[field] is None:
            warnings.append(f"[{filename}] Item {idx}: field '{field}' is null")
    return warnings


# ---------------------------------------------------------------------------
# Consistency checker
# ---------------------------------------------------------------------------


class ConsistencyChecker:
    """Check that questions are consistent across methods and files."""

    def __init__(self):
        # method -> set of question ids
        self.method_questions: dict[str, set[str]] = defaultdict(set)
        # method -> list of (source_file, question_id)
        self.method_sources: dict[str, list[tuple[str, str]]] = defaultdict(list)
        # question_id -> set of ground truth answers
        self.ground_truths: dict[str, set[str]] = defaultdict(set)
        # question_id -> question text (for reporting)
        self.question_texts: dict[str, str] = {}

    def add_result(self, method: str, question_id: str, question: str,
                   ground_truth: str, source_file: str) -> None:
        self.method_questions[method].add(question_id)
        self.method_sources[method].append((source_file, question_id))
        if ground_truth:
            self.ground_truths[question_id].add(ground_truth.strip().lower())
        if question:
            self.question_texts[question_id] = question

    def check(self) -> list[str]:
        """Run all consistency checks, return list of warnings."""
        warnings = []

        # 1. Check that each method has consistent question sets across files
        for method, entries in self.method_sources.items():
            file_questions: dict[str, set[str]] = defaultdict(set)
            for src, qid in entries:
                file_questions[os.path.basename(src)].add(qid)

            if len(file_questions) > 1:
                all_qids = set()
                for qs in file_questions.values():
                    all_qids.update(qs)
                for fname, qs in file_questions.items():
                    missing = all_qids - qs
                    if missing:
                        warnings.append(
                            f"[{method}] File '{fname}' is missing "
                            f"{len(missing)} questions present in other files: "
                            f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
                        )

        # 2. Check ground truth consistency
        for qid, truths in self.ground_truths.items():
            if len(truths) > 1:
                warnings.append(
                    f"Question '{qid}' has conflicting ground truths: {truths}"
                )

        # 3. Report method coverage
        methods = sorted(self.method_questions.keys())
        if len(methods) > 1:
            all_qids = set()
            for qs in self.method_questions.values():
                all_qids.update(qs)

            for method in methods:
                covered = self.method_questions[method]
                missing = all_qids - covered
                if missing and len(missing) < len(all_qids) * 0.5:
                    warnings.append(
                        f"[{method}] Missing {len(missing)} questions "
                        f"that other methods have results for"
                    )

        return warnings


# ---------------------------------------------------------------------------
# Main validator
# ---------------------------------------------------------------------------


class ResultsValidator:
    """Validate all benchmark result files."""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.info: list[str] = []
        self.files_checked: list[str] = []
        self.checker = ConsistencyChecker()

    def _extract_results(self, data: Any, filename: str) -> list[dict]:
        """Extract per-question results from various JSON formats."""
        results = []

        if isinstance(data, list):
            # comp_*.json / v*.json format
            for item in data:
                if isinstance(item, dict) and "question" in item:
                    results.append(item)

        elif isinstance(data, dict):
            # statistics_format: has results array
            for item in data.get("results", []):
                for method_raw, mdata in item.get("method_results", {}).items():
                    results.append({
                        "question_id": item.get("question_id", ""),
                        "question": item.get("question", ""),
                        "ground_truth": item.get("ground_truth", ""),
                        "answer": mdata.get("extracted_answer", ""),
                        "correct": mdata.get("is_correct", False),
                        "method": method_raw,
                    })

            # pipeline_format
            pipelines = data.get("pipelines", {})
            if isinstance(pipelines, dict):
                for pipeline_name, pdata in pipelines.items():
                    if isinstance(pdata, dict):
                        for item in pdata.get("results", []):
                            results.append({
                                "question_id": item.get("problem_id", ""),
                                "question": item.get("problem", ""),
                                "ground_truth": item.get("ground_truth", ""),
                                "answer": item.get("answer", ""),
                                "correct": item.get("correct", False),
                                "method": pipeline_name,
                            })

            # full_benchmark format
            for section in ["baseline", "self_reflection"]:
                for item in data.get(section, []):
                    results.append({
                        "question_id": item.get("id", ""),
                        "question": item.get("question", ""),
                        "ground_truth": item.get("ground_truth", ""),
                        "answer": item.get("answer", ""),
                        "correct": item.get("correct", False),
                        "method": section,
                    })

            # checkpoint_format
            for item in data.get("results", []):
                for method_raw, mdata in item.get("method_results", {}).items():
                    results.append({
                        "question_id": item.get("question_id", ""),
                        "question": item.get("question", ""),
                        "ground_truth": item.get("ground_truth", ""),
                        "answer": mdata.get("extracted_answer", ""),
                        "correct": mdata.get("is_correct", False),
                        "method": method_raw,
                    })

        return results

    def validate_file(self, filepath: str) -> None:
        """Validate a single JSON file."""
        fname = os.path.basename(filepath)

        # 1. Parse JSON
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw_text = f.read()
            data = json.loads(raw_text)
        except json.JSONDecodeError as e:
            self.errors.append(f"[{fname}] Invalid JSON: {e}")
            return
        except OSError as e:
            self.errors.append(f"[{fname}] Cannot read: {e}")
            return

        self.files_checked.append(fname)

        # 2. Check for simulation artifacts in raw text
        sim_warnings = detect_simulation_artifacts(data, fname)
        self.warnings.extend(sim_warnings)

        # 3. Extract per-question results
        results = self._extract_results(data, fname)

        if not results:
            # It might be an aggregate-only file (like comprehensive_summary.json)
            self.info.append(f"[{fname}] No per-question results (aggregate-only file)")
            return

        # 4. Check suspicious patterns
        pattern_warnings = detect_suspicious_patterns(results, fname)
        self.warnings.extend(pattern_warnings)

        # 5. Register with consistency checker
        for item in results:
            method = item.get("method", "unknown")
            qid = str(item.get("question_id", item.get("id", "")))
            question = item.get("question", "")
            gt = str(item.get("ground_truth", item.get("correct", "")))
            self.checker.add_result(method, qid, question, gt, filepath)

        # 6. Check for duplicate question IDs within the file
        qids = [str(r.get("question_id", r.get("id", ""))) for r in results]
        qid_counts = Counter(qids)
        for qid, count in qid_counts.items():
            if count > 1:
                self.warnings.append(
                    f"[{fname}] Duplicate question ID '{qid}' appears {count} times"
                )

        self.info.append(f"[{fname}] {len(results)} results extracted")

    def validate_all(self) -> None:
        """Validate all JSON files in the results directory."""
        if not self.results_dir.exists():
            self.errors.append(f"Results directory '{self.results_dir}' does not exist")
            return

        for fname in sorted(os.listdir(self.results_dir)):
            if fname.endswith(".json"):
                self.validate_file(str(self.results_dir / fname))

        # Run consistency checks
        consistency_warnings = self.checker.check()
        self.warnings.extend(consistency_warnings)

    def report(self) -> str:
        """Generate a validation report."""
        lines = []
        lines.append("=" * 65)
        lines.append("BENCHMARK RESULTS VALIDATION REPORT")
        lines.append("=" * 65)
        lines.append(f"Results directory: {self.results_dir}")
        lines.append(f"Files checked: {len(self.files_checked)}")
        lines.append("")

        # Errors
        if self.errors:
            lines.append(f"ERRORS ({len(self.errors)}):")
            lines.append("-" * 40)
            for err in self.errors:
                lines.append(f"  ERROR: {err}")
            lines.append("")
        else:
            lines.append("ERRORS: None")
            lines.append("")

        # Warnings
        if self.warnings:
            lines.append(f"WARNINGS ({len(self.warnings)}):")
            lines.append("-" * 40)
            for warn in self.warnings:
                lines.append(f"  WARN: {warn}")
            lines.append("")
        else:
            lines.append("WARNINGS: None")
            lines.append("")

        # Info
        if self.info:
            lines.append(f"INFO ({len(self.info)}):")
            lines.append("-" * 40)
            for msg in self.info:
                lines.append(f"  {msg}")
            lines.append("")

        # Method coverage summary
        if self.checker.method_questions:
            lines.append("METHOD COVERAGE:")
            lines.append("-" * 40)
            for method in sorted(self.checker.method_questions.keys()):
                n = len(self.checker.method_questions[method])
                sources = set(s for s, _ in self.checker.method_sources[method])
                lines.append(f"  {method:<25} {n:>4} questions from {len(sources)} file(s)")
            lines.append("")

        # Ground truth conflicts
        conflicts = {qid: truths for qid, truths in self.checker.ground_truths.items()
                     if len(truths) > 1}
        if conflicts:
            lines.append(f"GROUND TRUTH CONFLICTS ({len(conflicts)}):")
            lines.append("-" * 40)
            for qid, truths in sorted(conflicts.items()):
                text = self.checker.question_texts.get(qid, "")[:60]
                lines.append(f"  '{qid}' ({text}): {truths}")
            lines.append("")

        # Overall verdict
        lines.append("=" * 65)
        if self.errors:
            lines.append("VERDICT: FAIL - errors found")
        elif self.warnings:
            lines.append(f"VERDICT: PASS WITH WARNINGS ({len(self.warnings)} warnings)")
        else:
            lines.append("VERDICT: PASS - all checks passed")
        lines.append("=" * 65)

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Validate benchmark results.")
    parser.add_argument("--results-dir", default="benchmark_results",
                        help="Directory containing result JSON files")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / args.results_dir

    validator = ResultsValidator(str(results_dir))
    validator.validate_all()
    report = validator.report()
    print(report)

    # Exit with appropriate code
    if validator.errors:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
