"""Tests for metrics collection module."""

import pytest
import json
from src.utils.metrics import (
    MetricsCollector,
    ProblemMetrics,
    ExperimentMetrics,
    ComparisonReport,
)


class TestProblemMetrics:
    """Tests for ProblemMetrics dataclass."""

    def test_creation(self):
        m = ProblemMetrics(
            problem_id="p1",
            solved=True,
            correct=True,
            score=0.9,
            input_tokens=100,
            output_tokens=50,
            latency_seconds=1.5,
        )
        assert m.problem_id == "p1"
        assert m.correct is True
        assert m.score == 0.9

    def test_defaults(self):
        m = ProblemMetrics(
            problem_id="p1", solved=True, correct=False, score=0.0,
            input_tokens=10, output_tokens=5, latency_seconds=0.1,
        )
        assert m.num_expansions == 0
        assert m.num_reflections == 0
        assert m.num_backtracks == 0
        assert m.ground_truth is None


class TestExperimentMetrics:
    """Tests for ExperimentMetrics dataclass."""

    def test_creation(self):
        m = ExperimentMetrics(
            experiment_name="test",
            num_problems=10,
            accuracy=0.8,
            avg_score=0.75,
            total_tokens=1000,
            avg_tokens_per_problem=100,
            total_latency_seconds=10.0,
            avg_latency_seconds=1.0,
            total_expansions=20,
            total_reflections=10,
            total_backtracks=5,
            avg_path_length=3.5,
        )
        assert m.experiment_name == "test"
        assert m.accuracy == 0.8


def _make_problem(problem_id="p1", correct=True, score=0.8, input_tokens=100,
                  output_tokens=50, latency_seconds=1.0, num_expansions=3,
                  num_reflections=2, num_backtracks=1, path_length=5):
    return ProblemMetrics(
        problem_id=problem_id,
        solved=True,
        correct=correct,
        score=score,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_seconds=latency_seconds,
        num_expansions=num_expansions,
        num_reflections=num_reflections,
        num_backtracks=num_backtracks,
        path_length=path_length,
    )


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    def test_init(self):
        c = MetricsCollector(name="test")
        assert c.name == "test"

    def test_record_problem(self):
        c = MetricsCollector()
        c.record_problem(_make_problem())
        assert len(c._problem_metrics) == 1

    def test_get_aggregate_empty(self):
        c = MetricsCollector()
        agg = c.get_aggregate_metrics()
        assert agg.num_problems == 0
        assert agg.accuracy == 0.0

    def test_get_aggregate_single(self):
        c = MetricsCollector()
        c.record_problem(_make_problem(correct=True, score=0.9))
        agg = c.get_aggregate_metrics()
        assert agg.num_problems == 1
        assert agg.accuracy == 1.0
        assert agg.avg_score == pytest.approx(0.9)

    def test_get_aggregate_multiple(self):
        c = MetricsCollector()
        c.record_problem(_make_problem(correct=True, score=0.9, input_tokens=100, output_tokens=50))
        c.record_problem(_make_problem(correct=False, score=0.3, input_tokens=200, output_tokens=100))
        agg = c.get_aggregate_metrics()
        assert agg.num_problems == 2
        assert agg.accuracy == pytest.approx(0.5)
        assert agg.total_tokens == 450  # 150 + 300

    def test_compare_with_baseline(self):
        ours = MetricsCollector(name="rl")
        ours.record_problem(_make_problem(correct=True))
        ours.record_problem(_make_problem(correct=True))

        baseline = MetricsCollector(name="baseline")
        baseline.record_problem(_make_problem(correct=True))
        baseline.record_problem(_make_problem(correct=False))

        comparison = ours.compare_with_baseline(baseline)
        assert comparison["accuracy_improvement"] == pytest.approx(0.5)
        assert comparison["accuracy_ours"] == pytest.approx(1.0)
        assert comparison["accuracy_baseline"] == pytest.approx(0.5)

    def test_compare_with_empty_baseline(self):
        ours = MetricsCollector()
        comparison = ours.compare_with_baseline(MetricsCollector())
        assert "error" in comparison

    def test_get_error_analysis_no_errors(self):
        c = MetricsCollector()
        c.record_problem(_make_problem(correct=True))
        analysis = c.get_error_analysis()
        assert analysis["total_errors"] == 0

    def test_get_error_analysis_with_errors(self):
        c = MetricsCollector()
        c.record_problem(_make_problem(correct=True))
        c.record_problem(_make_problem(correct=False, score=0.3))
        c.record_problem(_make_problem(correct=False, score=0.8))
        analysis = c.get_error_analysis()
        assert analysis["total_errors"] == 2
        assert analysis["low_score_errors"] == 1
        assert analysis["high_score_errors"] == 1

    def test_get_action_distribution(self):
        c = MetricsCollector()
        c.record_problem(_make_problem(num_expansions=3, num_reflections=2, num_backtracks=1))
        c.record_problem(_make_problem(num_expansions=1, num_reflections=0, num_backtracks=2))
        dist = c.get_action_distribution()
        assert dist["expansions"] == 4
        assert dist["reflections"] == 2
        assert dist["backtracks"] == 3

    def test_export_results(self, tmp_path):
        c = MetricsCollector(name="test_export")
        c.record_problem(_make_problem())
        filepath = str(tmp_path / "metrics.json")
        c.export_results(filepath)
        with open(filepath) as f:
            data = json.load(f)
        assert data["name"] == "test_export"
        assert len(data["problems"]) == 1

    def test_clear(self):
        c = MetricsCollector()
        c.record_problem(_make_problem())
        c._timestamps["p1"] = 1.0
        c.clear()
        assert len(c._problem_metrics) == 0
        assert len(c._timestamps) == 0

    def test_start_problem(self):
        c = MetricsCollector()
        c.start_problem("p1")
        assert "p1" in c._timestamps


class TestComparisonReport:
    """Tests for ComparisonReport class."""

    @pytest.fixture
    def report(self):
        baseline = MetricsCollector(name="baseline")
        baseline.record_problem(_make_problem(correct=True, score=0.7, input_tokens=50, output_tokens=30, latency_seconds=0.5))
        baseline.record_problem(_make_problem(correct=False, score=0.4, input_tokens=60, output_tokens=40, latency_seconds=0.6))

        rl = MetricsCollector(name="rl")
        rl.record_problem(_make_problem(correct=True, score=0.9, input_tokens=100, output_tokens=60, latency_seconds=1.0))
        rl.record_problem(_make_problem(correct=True, score=0.85, input_tokens=120, output_tokens=70, latency_seconds=1.2))
        return ComparisonReport(baseline, rl)

    def test_generate_report(self, report):
        text = report.generate_report()
        assert "EXPERIMENT COMPARISON REPORT" in text
        assert "ACCURACY" in text
        assert "COST ANALYSIS" in text

    def test_get_summary_dict(self, report):
        summary = report.get_summary_dict()
        assert "accuracy" in summary
        assert "tokens" in summary
        assert "latency" in summary
        assert "rl_actions" in summary
        assert summary["accuracy"]["rl"] == pytest.approx(1.0)
        assert summary["accuracy"]["baseline"] == pytest.approx(0.5)
