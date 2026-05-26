"""Aggregate benchmark results and generate publication-ready outputs.

Scans benchmark_results/ for all JSON files, normalizes heterogeneous formats,
computes aggregate statistics with confidence intervals, runs McNemar's tests,
and writes LaTeX tables, CSV exports, and a summary Markdown report.

Usage:
    python scripts/aggregate_results.py [--results-dir benchmark_results] [--output-dir paper/tables]
"""

import json
import os
import sys
import csv
import argparse
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Any

import numpy as np

try:
    from scipy.stats import chi2
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHOD_DISPLAY = {
    "zero_shot": "Zero-Shot",
    "zeroshot": "Zero-Shot",
    "chain_of_thought": "CoT",
    "cot": "CoT",
    "self_consistency": "SC",
    "sc_only": "SC",
    "rag": "RAG",
    "kb_sc_step": "KB+SC",
    "kbsc": "KB+SC",
    "kb_sc": "KB+SC",
    "Baseline": "Baseline",
    "SelfReflection": "Self-Reflection",
}

SKIP_FILES = {"checkpoint.json"}

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _classify_file(path: str, data: Any) -> str:
    """Return a format tag for a JSON file."""
    name = os.path.basename(path)
    if name in SKIP_FILES:
        return "skip"

    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        if "answer" in data[0] and "correct" in data[0]:
            return "per_question_list"
        return "unknown_list"

    if isinstance(data, dict):
        if "statistics" in data and "methods" in data["statistics"]:
            return "statistics_format"
        if "pipelines" in data and isinstance(data["pipelines"], dict):
            return "pipeline_format"
        if "methods" in data and "study" in data:
            return "summary_format"
        if "metrics" in data and "baseline" in data:
            return "full_benchmark"
        if "results" in data and "config" in data:
            return "checkpoint_format"
        if "methods" in data:
            return "methods_dict"

    return "unknown"


def _normalize_method(name: str) -> str:
    """Map raw method/pipeline names to canonical short names."""
    low = name.lower().replace(" ", "_").replace("-", "_")
    for raw, canon in [
        ("zero_shot", "zero_shot"),
        ("zeroshot", "zero_shot"),
        ("chain_of_thought", "chain_of_thought"),
        ("cot", "chain_of_thought"),
        ("self_consistency", "self_consistency"),
        ("sc_only", "self_consistency"),
        ("sc", "self_consistency"),
        ("rag", "rag"),
        ("kb_sc_step", "kb_sc"),
        ("kbsc", "kb_sc"),
        ("kb_sc", "kb_sc"),
        ("baseline", "baseline"),
        ("selfreflection", "self_reflection"),
        ("self_reflection", "self_reflection"),
    ]:
        if low == raw:
            return canon
    return low


def _display_name(canonical: str) -> str:
    return METHOD_DISPLAY.get(canonical, canonical.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Result classes
# ---------------------------------------------------------------------------


class QuestionResult:
    __slots__ = ("question_id", "question", "method", "correct", "latency_ms", "source_file")

    def __init__(self, question_id: str, question: str, method: str, correct: bool,
                 latency_ms: float = 0.0, source_file: str = ""):
        self.question_id = question_id
        self.question = question
        self.method = method
        self.correct = correct
        self.latency_ms = latency_ms
        self.source_file = source_file


# ---------------------------------------------------------------------------
# File parsers
# ---------------------------------------------------------------------------


def parse_per_question_list(path: str, data: list) -> list[QuestionResult]:
    """Parse comp_*.json / v*.json style: list of {id, question, answer, correct, ...}."""
    results = []
    fname = os.path.basename(path)
    # Infer method from filename
    method_raw = fname
    for prefix in ["comp_", "v\\d+_", "_s\\d+_e\\d+"]:
        method_raw = re.sub(prefix, "", method_raw)
    method_raw = method_raw.replace(".json", "")
    # Try to extract method from known patterns like comp_zeroshot_s0_e50
    for m in ["zeroshot", "cot", "rag", "sc_only", "kbsc", "kb_sc"]:
        if m in fname.lower().replace("-", "_"):
            method_raw = m
            break
    method = _normalize_method(method_raw)

    for item in data:
        qid = str(item.get("id", ""))
        question = item.get("question", "")
        answer = str(item.get("answer", "")).strip().lower()
        correct_val = str(item.get("correct", "")).strip().lower()
        is_correct = answer == correct_val
        elapsed = float(item.get("elapsed", 0)) * 1000  # seconds -> ms
        results.append(QuestionResult(qid, question, method, is_correct, elapsed, path))
    return results


def parse_statistics_format(path: str, data: dict) -> list[QuestionResult]:
    """Parse benchmark_results_*.json with statistics.methods and results array."""
    results = []
    # Per-question results
    for item in data.get("results", []):
        qid = str(item.get("question_id", item.get("id", "")))
        question = item.get("question", "")
        for method_raw, mdata in item.get("method_results", {}).items():
            method = _normalize_method(method_raw)
            is_correct = bool(mdata.get("is_correct", False))
            latency = float(mdata.get("latency_ms", 0))
            results.append(QuestionResult(qid, question, method, is_correct, latency, path))
    return results


def parse_pipeline_format(path: str, data: dict) -> list[QuestionResult]:
    """Parse iter6/real benchmark format with pipelines dict."""
    results = []
    pipelines = data.get("pipelines", {})
    if not isinstance(pipelines, dict):
        return results
    for pipeline_name, pdata in pipelines.items():
        if not isinstance(pdata, dict):
            continue
        method = _normalize_method(pipeline_name)
        for item in pdata.get("results", []):
            qid = str(item.get("problem_id", item.get("id", "")))
            question = item.get("problem", item.get("question", ""))
            is_correct = bool(item.get("correct", False))
            latency = float(item.get("latency_seconds", 0)) * 1000
            results.append(QuestionResult(qid, question, method, is_correct, latency, path))
    return results


def parse_checkpoint_format(path: str, data: dict) -> list[QuestionResult]:
    """Parse checkpoint.json with results array and method_results."""
    results = []
    for item in data.get("results", []):
        qid = str(item.get("question_id", ""))
        question = item.get("question", "")
        for method_raw, mdata in item.get("method_results", {}).items():
            method = _normalize_method(method_raw)
            is_correct = bool(mdata.get("is_correct", False))
            latency = float(mdata.get("latency_ms", 0))
            results.append(QuestionResult(qid, question, method, is_correct, latency, path))
    return results


def parse_full_benchmark(path: str, data: dict) -> list[QuestionResult]:
    """Parse full_benchmark.json with baseline/self_reflection arrays."""
    results = []
    for section, method in [("baseline", "baseline"), ("self_reflection", "self_reflection")]:
        for item in data.get(section, []):
            qid = str(item.get("id", ""))
            question = item.get("question", "")
            is_correct = bool(item.get("correct", False))
            results.append(QuestionResult(qid, question, method, is_correct, 0.0, path))
    return results


def parse_summary_format(path: str, data: dict) -> list[QuestionResult]:
    """Parse comprehensive_summary.json – only has aggregate numbers, skip per-question."""
    return []


def parse_methods_dict(path: str, data: dict) -> list[QuestionResult]:
    """Parse a dict with 'methods' key mapping to aggregate stats only."""
    return []


PARSERS = {
    "per_question_list": parse_per_question_list,
    "statistics_format": parse_statistics_format,
    "pipeline_format": parse_pipeline_format,
    "checkpoint_format": parse_checkpoint_format,
    "full_benchmark": parse_full_benchmark,
    "summary_format": parse_summary_format,
    "methods_dict": parse_methods_dict,
}


# ---------------------------------------------------------------------------
# Aggregation engine
# ---------------------------------------------------------------------------


class ResultsAggregator:
    """Collect, aggregate, and analyse benchmark results."""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.all_results: list[QuestionResult] = []
        self.files_parsed: list[str] = []
        self.files_skipped: list[str] = []
        self.parse_errors: list[str] = []

    # ---- loading ----------------------------------------------------------

    def load_all(self) -> None:
        if not self.results_dir.exists():
            print(f"Warning: results directory '{self.results_dir}' does not exist.")
            return
        for fname in sorted(os.listdir(self.results_dir)):
            if not fname.endswith(".json"):
                continue
            if fname in SKIP_FILES:
                self.files_skipped.append(fname)
                continue
            fpath = str(self.results_dir / fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                self.parse_errors.append(f"{fname}: {e}")
                continue

            fmt = _classify_file(fpath, data)
            if fmt == "skip" or fmt == "unknown" or fmt == "unknown_list":
                self.files_skipped.append(fname)
                continue

            parser = PARSERS.get(fmt)
            if parser is None:
                self.files_skipped.append(fname)
                continue

            try:
                items = parser(fpath, data)
                if items:
                    self.all_results.extend(items)
                    self.files_parsed.append(fname)
                else:
                    self.files_skipped.append(fname)
            except Exception as e:
                self.parse_errors.append(f"{fname}: {e}")

    # ---- aggregation ------------------------------------------------------

    def method_names(self) -> list[str]:
        return sorted({r.method for r in self.all_results})

    def results_for_method(self, method: str) -> list[QuestionResult]:
        return [r for r in self.all_results if r.method == method]

    def question_ids_for_method(self, method: str) -> set[str]:
        return {r.question_id for r in self.all_results if r.method == method}

    def accuracy(self, method: str) -> tuple[int, int, float]:
        rs = self.results_for_method(method)
        n = len(rs)
        if n == 0:
            return 0, 0, 0.0
        c = sum(1 for r in rs if r.correct)
        return c, n, c / n

    def ci_95_wilson(self, method: str) -> tuple[float, float]:
        """Wilson score 95% CI for a proportion."""
        c, n, p = self.accuracy(method)
        if n == 0:
            return 0.0, 0.0
        z = 1.96
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
        return max(0.0, center - margin), min(1.0, center + margin)

    def avg_latency(self, method: str) -> tuple[float, float]:
        rs = self.results_for_method(method)
        if not rs:
            return 0.0, 0.0
        latencies = [r.latency_ms for r in rs]
        return float(np.mean(latencies)), float(np.std(latencies))

    # ---- statistical tests ------------------------------------------------

    @staticmethod
    def _mcnemar_p(b: int, c: int) -> float:
        """McNemar's test p-value (with continuity correction)."""
        if b + c == 0:
            return 1.0
        if HAS_SCIPY:
            chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
            return float(max(0.0, min(1.0, 1.0 - chi2.cdf(chi2_stat, 1))))
        # Fallback without scipy
        chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
        # Approximate p-value using survival function of chi2 with df=1
        import math
        p = math.exp(-chi2_stat / 2)
        return float(max(0.0, min(1.0, p)))

    def mcnemar_pairwise(self, method_a: str, method_b: str) -> dict:
        """Compute McNemar's test between two methods on their common questions."""
        ids_a = {r.question_id: r for r in self.results_for_method(method_a)}
        ids_b = {r.question_id: r for r in self.results_for_method(method_b)}
        common = set(ids_a.keys()) & set(ids_b.keys())
        if not common:
            return {"n_common": 0, "b": 0, "c": 0, "chi2": 0.0, "p_value": 1.0,
                    "significant_005": False, "significant_01": False}

        b = c = 0
        for qid in common:
            a_ok = ids_a[qid].correct
            b_ok = ids_b[qid].correct
            if not a_ok and b_ok:
                b += 1
            elif a_ok and not b_ok:
                c += 1

        if b + c == 0:
            chi2_val, p = 0.0, 1.0
        else:
            chi2_val = (abs(b - c) - 1) ** 2 / (b + c)
            p = self._mcnemar_p(b, c)

        return {
            "n_common": len(common),
            "b": b,  # a wrong, b right
            "c": c,  # a right, b wrong
            "chi2": round(chi2_val, 4),
            "p_value": round(p, 6),
            "significant_005": p < 0.05,
            "significant_01": p < 0.10,
        }

    def all_pairwise_mcnemar(self, reference: str = "zero_shot") -> list[dict]:
        """Run McNemar's test for every method pair; return sorted list."""
        methods = self.method_names()
        pairs = []
        for i, m1 in enumerate(methods):
            for m2 in methods[i + 1:]:
                res = self.mcnemar_pairwise(m1, m2)
                res["method_a"] = m1
                res["method_b"] = m2
                pairs.append(res)
        return pairs

    # ---- per-source run grouping ------------------------------------------

    def group_by_source(self) -> dict[str, list[QuestionResult]]:
        groups: dict[str, list[QuestionResult]] = defaultdict(list)
        for r in self.all_results:
            groups[r.source_file].append(r)
        return dict(groups)

    def detect_multi_runs(self) -> dict[str, list[str]]:
        """Detect methods that have results from multiple source files."""
        method_sources: dict[str, set[str]] = defaultdict(set)
        for r in self.all_results:
            method_sources[r.method].add(os.path.basename(r.source_file))
        return {m: sorted(s) for m, s in method_sources.items() if len(s) > 1}

    # ---- summary table builder -------------------------------------------

    def build_summary_table(self, baseline: str = "zero_shot") -> list[dict]:
        """Build one row per method with accuracy, CI, p-value vs baseline."""
        rows = []
        methods = self.method_names()
        for method in methods:
            c, n, acc = self.accuracy(method)
            lo, hi = self.ci_95_wilson(method)
            mean_lat, std_lat = self.avg_latency(method)

            p_vs_base = None
            if method != baseline and baseline in methods:
                p_vs_base = self.mcnemar_pairwise(baseline, method)["p_value"]

            rows.append({
                "method": method,
                "display": _display_name(method),
                "correct": c,
                "total": n,
                "accuracy": acc,
                "ci_low": lo,
                "ci_high": hi,
                "mean_latency_ms": mean_lat,
                "std_latency_ms": std_lat,
                "p_value_vs_baseline": p_vs_base,
            })

        # Sort: baseline first, then by accuracy descending
        rows.sort(key=lambda r: (r["method"] != baseline, -r["accuracy"]))
        return rows


# ---------------------------------------------------------------------------
# Output generators
# ---------------------------------------------------------------------------


def generate_latex_table(rows: list[dict], baseline: str = "zero_shot") -> str:
    """Generate a publication-ready LaTeX table."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Benchmark results: accuracy and statistical significance.}")
    lines.append(r"\label{tab:results}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l c c c c}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Method} & \textbf{Accuracy} & \textbf{95\% CI} & \textbf{$N$} & \textbf{$p$-value (vs " + _display_name(baseline) + r")} \\")
    lines.append(r"\midrule")

    for row in rows:
        name = row["display"]
        acc = f"{row['accuracy']:.1%}"
        ci = f"[{row['ci_low']:.1%}, {row['ci_high']:.1%}]"
        n = str(row["total"])
        if row["method"] == baseline:
            pval = "---"
        elif row["p_value_vs_baseline"] is not None:
            p = row["p_value_vs_baseline"]
            if p < 0.001:
                pval = r"$<$0.001"
            elif p < 0.05:
                pval = f"{p:.3f}" + r"$^{*}$"
            elif p < 0.10:
                pval = f"{p:.3f}" + r"$^{\dagger}$"
            else:
                pval = f"{p:.3f}"
        else:
            pval = "---"

        # Bold best
        marker = ""
        if row["accuracy"] == max(r["accuracy"] for r in rows):
            marker = r"\textbf{"

        line = f"{name} & {acc} & {ci} & {n} & {pval} \\\\"
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_csv(rows: list[dict], path: str) -> None:
    """Write results as CSV."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_markdown(aggregator: ResultsAggregator, rows: list[dict],
                      mcnemar_pairs: list[dict], output_path: str) -> None:
    """Write a full summary Markdown report."""
    lines = []
    lines.append("# Benchmark Results Summary\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"Results directory: `{aggregator.results_dir}`\n")
    lines.append(f"Files parsed: {len(aggregator.files_parsed)} | "
                 f"Skipped: {len(aggregator.files_skipped)} | "
                 f"Errors: {len(aggregator.parse_errors)}\n")

    if aggregator.parse_errors:
        lines.append("\n## Parse Errors\n")
        for err in aggregator.parse_errors:
            lines.append(f"- `{err}`")
        lines.append("")

    # Main table
    lines.append("\n## Main Results\n")
    lines.append("| Method | Accuracy | 95% CI | N | p-value (vs Zero-Shot) |")
    lines.append("|--------|----------|--------|---|----------------------|")
    for row in rows:
        pval = "---" if row["p_value_vs_baseline"] is None else f"{row['p_value_vs_baseline']:.4f}"
        ci = f"[{row['ci_low']:.1%}, {row['ci_high']:.1%}]"
        lines.append(f"| {row['display']} | {row['accuracy']:.1%} | {ci} | {row['total']} | {pval} |")

    # Multi-run detection
    multi = aggregator.detect_multi_runs()
    if multi:
        lines.append("\n## Multi-Run Sources\n")
        lines.append("Methods with results from multiple files (potential cross-validation):\n")
        for method, sources in multi.items():
            lines.append(f"- **{_display_name(method)}**: {', '.join(f'`{s}`' for s in sources)}")
        lines.append("")

    # Per-source breakdown (ablation)
    groups = aggregator.group_by_source()
    if len(groups) > 1:
        lines.append("\n## Per-Source Breakdown\n")
        for source, qresults in sorted(groups.items()):
            method_acc: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
            for r in qresults:
                c, n = method_acc[r.method]
                method_acc[r.method] = (c + (1 if r.correct else 0), n + 1)
            lines.append(f"### `{source}`\n")
            lines.append("| Method | Correct | Total | Accuracy |")
            lines.append("|--------|---------|-------|----------|")
            for m, (c, n) in sorted(method_acc.items()):
                lines.append(f"| {_display_name(m)} | {c} | {n} | {c/n:.1%} |")
            lines.append("")

    # McNemar pairwise
    if mcnemar_pairs:
        lines.append("\n## McNemar's Test (Pairwise)\n")
        lines.append("| Method A | Method B | N (common) | b | c | chi2 | p-value | Sig (0.05) |")
        lines.append("|----------|----------|------------|---|---|------|---------|------------|")
        for p in mcnemar_pairs:
            sig = "Yes" if p["significant_005"] else "No"
            lines.append(
                f"| {_display_name(p['method_a'])} | {_display_name(p['method_b'])} | "
                f"{p['n_common']} | {p['b']} | {p['c']} | {p['chi2']:.3f} | "
                f"{p['p_value']:.4f} | {sig} |"
            )

    # Latency table
    lines.append("\n## Latency Statistics\n")
    lines.append("| Method | Mean (ms) | Std (ms) |")
    lines.append("|--------|-----------|----------|")
    for row in rows:
        lines.append(f"| {row['display']} | {row['mean_latency_ms']:.1f} | {row['std_latency_ms']:.1f} |")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Aggregate benchmark results.")
    parser.add_argument("--results-dir", default="benchmark_results",
                        help="Directory containing result JSON files")
    parser.add_argument("--output-dir", default="paper/tables",
                        help="Directory for output files")
    parser.add_argument("--baseline", default="zero_shot",
                        help="Baseline method for p-value comparisons")
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / args.results_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {results_dir}")
    agg = ResultsAggregator(str(results_dir))
    agg.load_all()

    print(f"Parsed {len(agg.files_parsed)} files, skipped {len(agg.files_skipped)}, "
          f"errors {len(agg.parse_errors)}")
    print(f"Total question-method results: {len(agg.all_results)}")
    print(f"Methods found: {', '.join(agg.method_names())}")

    if not agg.all_results:
        print("No per-question results found. Nothing to aggregate.")
        return

    # Build summary
    rows = agg.build_summary_table(baseline=args.baseline)

    # Print summary to console
    print("\n" + "=" * 65)
    print(f"{'Method':<20} {'Acc':>7} {'95% CI':>18} {'N':>5} {'p (vs BS)':>10}")
    print("-" * 65)
    for row in rows:
        ci = f"[{row['ci_low']:.1%}, {row['ci_high']:.1%}]"
        pval = "---" if row["p_value_vs_baseline"] is None else f"{row['p_value_vs_baseline']:.4f}"
        print(f"{row['display']:<20} {row['accuracy']:>6.1%} {ci:>18} {row['total']:>5} {pval:>10}")
    print("=" * 65)

    # McNemar pairwise
    mcnemar_pairs = agg.all_pairwise_mcnemar()

    # Generate outputs
    latex_path = output_dir / "results_table.tex"
    csv_path = output_dir / "results_table.csv"
    md_path = output_dir / "results_summary.md"

    latex = generate_latex_table(rows, baseline=args.baseline)
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex + "\n")
    print(f"\nLaTeX table: {latex_path}")

    generate_csv(rows, str(csv_path))
    print(f"CSV export:  {csv_path}")

    generate_markdown(agg, rows, mcnemar_pairs, str(md_path))
    print(f"Markdown:    {md_path}")

    # Also write a raw JSON aggregation
    json_path = output_dir / "aggregated_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated": datetime.now().isoformat(),
            "results_dir": str(results_dir),
            "files_parsed": agg.files_parsed,
            "files_skipped": agg.files_skipped,
            "parse_errors": agg.parse_errors,
            "methods": rows,
            "mcnemar_pairwise": mcnemar_pairs,
        }, f, indent=2)
    print(f"JSON:        {json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
