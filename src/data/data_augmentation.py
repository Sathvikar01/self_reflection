"""Data augmentation pipeline for expanding training datasets."""

import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
from pathlib import Path
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class AugmentedProblem:
    """Augmented problem with metadata."""
    id: str
    question: str
    answer: str
    answer_type: str
    original_id: str
    augmentation_type: str
    metadata: Dict[str, Any] = None


class QuestionParaphraser:
    """Paraphrase questions using templates."""

    @staticmethod
    def paraphrase(question: str) -> List[str]:
        """Generate paraphrased versions.

        Args:
            question: Original question

        Returns:
            List of paraphrased questions
        """
        paraphrases = []

        # Template-based paraphrasing
        if question.startswith("Do "):
            paraphrases.append(f"Are there {question[3:]}")
            paraphrases.append(f"Is it true that {question[3:].rstrip('?')}?")
            paraphrases.append(f"Can we say that {question[3:].rstrip('?')}?")

        elif question.startswith("Is "):
            paraphrases.append(f"Does {question[3:]}")
            paraphrases.append(f"Would you say that {question[3:].rstrip('?')}?")

        elif question.startswith("What "):
            # What questions
            paraphrases.append(f"Can you tell me {question[5:]}")
            paraphrases.append(f"I want to know {question[5:]}")

        # Add question mark if missing
        paraphrases = [
            p if p.endswith('?') else f"{p}?"
            for p in paraphrases
        ]

        return paraphrases


class CounterfactualGenerator:
    """Generate counterfactual problems."""

    @staticmethod
    def generate(problem: str, answer: str) -> List[Dict[str, str]]:
        """Generate counterfactual versions.

        Args:
            problem: Original problem
            answer: Original answer

        Returns:
            List of counterfactual problems
        """
        counterfactuals = []

        # For yes/no questions
        if answer.lower() in ["yes", "no"]:
            opposite = "no" if answer.lower() == "yes" else "yes"

            # Negate the question
            if "Do " in problem:
                counterfactual = problem.replace("Do ", "Do not ", 1)
            elif "Is " in problem:
                counterfactual = problem.replace("Is ", "Is not ", 1)
            elif "Are " in problem:
                counterfactual = problem.replace("Are ", "Are not ", 1)
            else:
                counterfactual = f"Is it false that {problem.rstrip('?')}?"

            counterfactuals.append({
                "question": counterfactual,
                "answer": opposite,
                "type": "negation"
            })

        return counterfactuals


class QuestionDecomposer:
    """Decompose complex questions into sub-questions."""

    @staticmethod
    def decompose(problem: str) -> List[Dict[str, Any]]:
        """Decompose complex question.

        Args:
            problem: Complex problem

        Returns:
            List of sub-problems
        """
        # Simple decomposition based on conjunctions
        subproblems = []

        if " and " in problem:
            parts = problem.split(" and ", 1)
            subproblems.append({
                "question": parts[0].rstrip('?') + "?",
                "type": "first_part"
            })
            subproblems.append({
                "question": parts[1] if parts[1].endswith('?') else parts[1] + "?",
                "type": "second_part"
            })

        return subproblems


class DataAugmentationPipeline:
    """Complete data augmentation pipeline."""

    def __init__(
        self,
        paraphrase: bool = True,
        counterfactual: bool = True,
        decompose: bool = True
    ):
        """Initialize augmentation pipeline.

        Args:
            paraphrase: Enable paraphrasing
            counterfactual: Enable counterfactual generation
            decompose: Enable decomposition
        """
        self.paraphrase_enabled = paraphrase
        self.counterfactual_enabled = counterfactual
        self.decompose_enabled = decompose

        self.paraphraser = QuestionParaphraser()
        self.counterfactual_gen = CounterfactualGenerator()
        self.decomposer = QuestionDecomposer()

        logger.info("DataAugmentationPipeline initialized")

    def augment_problem(
        self,
        problem: Dict[str, Any],
        n_augmentations: int = 3
    ) -> List[AugmentedProblem]:
        """Augment a single problem.

        Args:
            problem: Original problem dict
            n_augmentations: Number of augmentations per type

        Returns:
            List of augmented problems
        """
        augmented = []

        question = problem.get("question", "")
        answer = problem.get("answer", "")
        problem_id = problem.get("id", "unknown")

        # Paraphrasing
        if self.paraphrase_enabled:
            paraphrases = self.paraphraser.paraphrase(question)[:n_augmentations]
            for i, para in enumerate(paraphrases):
                aug = AugmentedProblem(
                    id=f"{problem_id}_para_{i}",
                    question=para,
                    answer=answer,
                    answer_type=problem.get("answer_type", "text"),
                    original_id=problem_id,
                    augmentation_type="paraphrase"
                )
                augmented.append(aug)

        # Counterfactual
        if self.counterfactual_enabled:
            counterfactuals = self.counterfactual_gen.generate(question, answer)
            for i, cf in enumerate(counterfactuals[:n_augmentations]):
                aug = AugmentedProblem(
                    id=f"{problem_id}_cf_{i}",
                    question=cf["question"],
                    answer=cf["answer"],
                    answer_type=problem.get("answer_type", "text"),
                    original_id=problem_id,
                    augmentation_type="counterfactual"
                )
                augmented.append(aug)

        # Decomposition
        if self.decompose_enabled:
            subproblems = self.decomposer.decompose(question)
            for i, sub in enumerate(subproblems[:n_augmentations]):
                aug = AugmentedProblem(
                    id=f"{problem_id}_sub_{i}",
                    question=sub["question"],
                    answer="unknown",  # Would need human labeling
                    answer_type="text",
                    original_id=problem_id,
                    augmentation_type="decomposition",
                    metadata={"subproblem_type": sub["type"]}
                )
                augmented.append(aug)

        return augmented

    def augment_dataset(
        self,
        problems: List[Dict[str, Any]],
        n_augmentations_per_problem: int = 2
    ) -> List[Dict[str, Any]]:
        """Augment entire dataset.

        Args:
            problems: List of problems
            n_augmentations_per_problem: Augmentations per problem

        Returns:
            List including original + augmented problems
        """
        augmented_dataset = []

        for problem in problems:
            # Keep original
            augmented_dataset.append(problem)

            # Add augmentations
            augmented = self.augment_problem(problem, n_augmentations_per_problem)
            for aug in augmented:
                augmented_dataset.append({
                    "id": aug.id,
                    "question": aug.question,
                    "answer": aug.answer,
                    "answer_type": aug.answer_type,
                    "original_id": aug.original_id,
                    "augmentation_type": aug.augmentation_type,
                    "metadata": aug.metadata
                })

        logger.info(f"Augmented {len(problems)} problems to {len(augmented_dataset)} total")

        return augmented_dataset

    def save_augmented_dataset(
        self,
        dataset: List[Dict],
        output_path: str
    ):
        """Save augmented dataset.

        Args:
            dataset: Dataset to save
            output_path: Output file path
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)

        logger.info(f"Augmented dataset saved to {output_path}")


def main():
    """Main function for data augmentation."""
    import argparse

    parser = argparse.ArgumentParser(description="Data augmentation")
    parser.add_argument("--input-file", type=str, required=True,
                        help="Input dataset file")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Output file for augmented dataset")
    parser.add_argument("--n-augmentations", type=int, default=2,
                        help="Augmentations per problem")
    parser.add_argument("--disable-paraphrase", action="store_true",
                        help="Disable paraphrasing")
    parser.add_argument("--disable-counterfactual", action="store_true",
                        help="Disable counterfactual")
    parser.add_argument("--disable-decompose", action="store_true",
                        help="Disable decomposition")

    args = parser.parse_args()

    # Load dataset
    with open(args.input_file, 'r') as f:
        dataset = json.load(f)

    # Initialize pipeline
    pipeline = DataAugmentationPipeline(
        paraphrase=not args.disable_paraphrase,
        counterfactual=not args.disable_counterfactual,
        decompose=not args.disable_decompose
    )

    # Augment
    augmented = pipeline.augment_dataset(
        dataset,
        n_augmentations_per_problem=args.n_augmentations
    )

    # Save
    pipeline.save_augmented_dataset(augmented, args.output_file)

    logger.info(f"Data augmentation complete: {len(dataset)} -> {len(augmented)}")


if __name__ == "__main__":
    main()
