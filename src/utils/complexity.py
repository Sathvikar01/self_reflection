"""Query complexity analysis for adaptive pipelines."""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ComplexityScore:
    """Complexity analysis of a query."""
    overall_score: float
    factors: Dict[str, float]
    recommended_depth: int
    recommended_tree_depth: int = 0
    reasoning: str = ""


class QueryComplexityAnalyzer:
    """Analyzes query complexity to determine search parameters."""

    FACTUAL_INDICATORS = [
        "what is", "what are", "who is", "who was", "where is",
        "when did", "how many", "how much", "define", "name the"
    ]

    REASONING_INDICATORS = [
        "why", "because", "therefore", "thus", "so", "hence",
        "if", "then", "would", "could", "should", "might",
        "compare", "contrast", "difference between", "similar"
    ]

    STRATEGIC_INDICATORS = [
        "best way", "optimal", "strategy", "should i",
        "which would", "choose", "decide", "evaluate",
        "pros and cons", "trade-off", "alternative"
    ]

    COMPLEXITY_MARKERS = [
        "multiple", "several", "various", "both", "each",
        "all of the", "none of the", "some but not all",
        "except", "unless", "however", "although", "despite"
    ]

    def analyze(self, query: str) -> ComplexityScore:
        """Analyze query complexity."""
        query_lower = query.lower()

        factors = {}

        has_factual = any(ind in query_lower for ind in self.FACTUAL_INDICATORS)
        has_reasoning = any(ind in query_lower for ind in self.REASONING_INDICATORS)
        has_strategic = any(ind in query_lower for ind in self.STRATEGIC_INDICATORS)

        if has_strategic:
            type_score = 0.8
        elif has_reasoning:
            type_score = 0.6
        elif has_factual:
            type_score = 0.3
        else:
            type_score = 0.5
        factors["question_type"] = type_score

        marker_count = sum(1 for m in self.COMPLEXITY_MARKERS if m in query_lower)
        factors["complexity_markers"] = min(marker_count * 0.15, 1.0)

        word_count = len(query.split())
        factors["length"] = min(word_count / 30, 1.0)

        negation_words = ["not", "never", "no ", "n't", "cannot", "can't"]
        has_negation = any(neg in query_lower for neg in negation_words)
        factors["negation"] = 0.2 if has_negation else 0.0

        separators = [" and ", " or ", ";", ",", " also "]
        part_count = sum(1 for sep in separators if sep in query_lower)
        factors["multi_part"] = min(part_count * 0.2, 0.5)

        weights = {
            "question_type": 0.35,
            "complexity_markers": 0.25,
            "length": 0.15,
            "negation": 0.15,
            "multi_part": 0.10
        }

        overall_score = sum(factors[k] * weights[k] for k in weights)

        if overall_score < 0.3:
            recommended_depth = 1
            recommended_tree_depth = 2
            reasoning = "Low complexity: simple factual query"
        elif overall_score < 0.5:
            recommended_depth = 2
            recommended_tree_depth = 3
            reasoning = "Medium-low complexity: standard reasoning"
        elif overall_score < 0.7:
            recommended_depth = 3
            recommended_tree_depth = 4
            reasoning = "Medium-high complexity: multi-step reasoning"
        else:
            recommended_depth = 4
            recommended_tree_depth = 5
            reasoning = "High complexity: strategic or multi-faceted query"

        return ComplexityScore(
            overall_score=overall_score,
            factors=factors,
            recommended_depth=recommended_depth,
            recommended_tree_depth=recommended_tree_depth,
            reasoning=reasoning
        )
