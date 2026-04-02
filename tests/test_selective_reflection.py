"""Tests for selective reflection feature."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestration.self_reflection_pipeline import (
    SelfReflectionPipeline,
    SelfReflectionConfig,
)


def test_problem_classification():
    config = SelfReflectionConfig(enable_selective_reflection=True)
    pipeline = SelfReflectionPipeline(config=config)

    factual_questions = [
        "Is the sun brighter than a light bulb?",
        "What is the capital of France?",
        "How many planets are in the solar system?",
    ]

    reasoning_questions = [
        "Do hamsters provide food for any animals?",
        "If John is taller than Mary and Mary is taller than Sue, who is the shortest?",
        "Why does ice float on water?",
    ]

    strategic_questions = [
        "What is the best opening move in chess?",
        "How should I invest my retirement savings?",
        "What's the optimal strategy for tic-tac-toe?",
    ]

    print("\n=== Testing Problem Classification ===\n")

    print("Factual questions:")
    for q in factual_questions:
        result = pipeline._classify_problem_type(q)
        print(f"  '{q[:50]}...' -> {result}")

    print("\nReasoning questions:")
    for q in reasoning_questions:
        result = pipeline._classify_problem_type(q)
        print(f"  '{q[:50]}...' -> {result}")

    print("\nStrategic questions:")
    for q in strategic_questions:
        result = pipeline._classify_problem_type(q)
        print(f"  '{q[:50]}...' -> {result}")


def test_baseline_confidence():
    config = SelfReflectionConfig(enable_selective_reflection=True)
    pipeline = SelfReflectionPipeline(config=config)

    high_confidence_reasoning = [
        "First, I need to understand what the question is asking about the sun's brightness.",
        "The key facts are: the sun is definitely a star, and light bulbs are artificial sources.",
        "Therefore, the sun is clearly much brighter than any light bulb.",
    ]

    low_confidence_reasoning = [
        "I'm not sure but maybe hamsters could be food for some animals.",
        "Perhaps cats or snakes might eat hamsters, but it's unclear.",
        "Possibly, hamsters might be prey in some situations.",
    ]

    print("\n=== Testing Baseline Confidence ===\n")

    high_conf = pipeline._calculate_baseline_confidence(high_confidence_reasoning)
    print(f"High confidence reasoning: {high_conf:.2f}")
    print(f"  Should be > 0.5: {high_conf > 0.5}")

    low_conf = pipeline._calculate_baseline_confidence(low_confidence_reasoning)
    print(f"\nLow confidence reasoning: {low_conf:.2f}")
    print(f"  Should be < high_conf: {low_conf < high_conf}")


def test_reflection_depths():
    config = SelfReflectionConfig(
        enable_selective_reflection=True,
        reflection_depths={
            "factual": 1,
            "reasoning": 2,
            "strategic": 3,
        },
    )

    print("\n=== Testing Reflection Depths Config ===\n")
    print(f"Factual problems: {config.reflection_depths['factual']} reflection pass")
    print(f"Reasoning problems: {config.reflection_depths['reasoning']} reflection passes")
    print(f"Strategic problems: {config.reflection_depths['strategic']} reflection passes")
    print(f"Confidence threshold for skipping: {config.confidence_threshold_skip}")


def test_selective_reflection_integration():
    config = SelfReflectionConfig(
        enable_selective_reflection=True,
        confidence_threshold_skip=0.9,
    )

    print("\n=== Selective Reflection Integration Test ===\n")
    print(f"Selective reflection enabled: {config.enable_selective_reflection}")
    print(f"Confidence threshold: {config.confidence_threshold_skip}")
    print(f"Reflection depths by type: {config.reflection_depths}")

    print("\nExpected behavior:")
    print("1. Factual questions: 1 reflection pass (knowledge retrieval focused)")
    print("2. Reasoning questions: 2 reflection passes (default)")
    print("3. Strategic questions: 3 reflection passes (needs deeper analysis)")
    print("4. High confidence (>0.9): Skip reflection entirely")
    print("5. No issues in first pass: Early stop")


if __name__ == "__main__":
    test_reflection_depths()
    test_problem_classification()
    test_baseline_confidence()
    test_selective_reflection_integration()
    print("\n=== All tests completed ===\n")
