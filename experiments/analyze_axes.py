from typing import Dict, List, Optional, Tuple
import json
import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

try:
    from .evaluation_utils import extract_answer_letter
except ImportError:
    from evaluation_utils import extract_answer_letter

QUESTION_TYPES = {
    "Detailed Finding Analysis": ["detection", "localization", "characterization"],
    "Pattern Recognition & Relations": ["detection", "classification", "relationship"],
    "Spatial Understanding": ["localization", "comparison", "relationship"],
    "Clinical Decision Making": ["classification", "comparison", "diagnosis"],
    "Diagnostic Classification": ["classification", "characterization", "diagnosis"],
}




def analyze_gpt4_results(
    results_file: str, max_questions: Optional[int] = None
) -> Tuple[float, Dict, Dict, List[str], List[str]]:
    """
    Analyze results in GPT-4 format.

    Args:
        results_file: Path to results file
        max_questions: Maximum number of questions to analyze

    Returns:
        Tuple containing:
            - overall_accuracy (float)
            - category_accuracies (Dict)
            - question_type_stats (Dict)
            - correct_ids (List[str])
            - incorrect_ids (List[str])
    """
    category_performance = defaultdict(lambda: {"total": 0, "correct": 0})
    all_questions = 0
    all_correct = 0
    correct_ids = []
    incorrect_ids = []

    with open(results_file, "r") as f:
        lines = f.readlines()

    processed_questions = 0

    for line in tqdm(lines, desc="Analyzing Benchmark Results"):
        # Check if we've hit the maximum questions
        if max_questions is not None and processed_questions >= max_questions:
            break
        if line.startswith("HTTP Request:"):
            continue

        try:
            entry = json.loads(line)
            metadata = entry.get("input", {}).get("question_data", {}).get("metadata", {})
            question_id = entry.get("question_id")

            model_letter = extract_answer_letter(entry.get("model_answer"))
            correct_letter = extract_answer_letter(entry.get("correct_answer"))

            if model_letter and correct_letter:
                all_questions += 1
                processed_questions += 1
                is_correct = model_letter == correct_letter

                if is_correct:
                    all_correct += 1
                    correct_ids.append(question_id)
                else:
                    incorrect_ids.append(question_id)

                for category in metadata.get("categories", []):
                    category_performance[category]["total"] += 1
                    if is_correct:
                        category_performance[category]["correct"] += 1

        except json.JSONDecodeError:
            continue

    return process_results(
        category_performance, all_questions, all_correct, correct_ids, incorrect_ids
    )


def load_chestagentbench_manifest(benchmark_path: str) -> Dict[str, Dict]:
    """Load released JSONL or generated question files as the evaluation manifest."""
    path = Path(benchmark_path)
    entries: Dict[str, Dict] = {}

    if path.is_file():
        question_records = [json.loads(line) for line in path.read_text().splitlines() if line]
    elif path.is_dir():
        question_records = []
        for question_file in path.glob("*/*.json"):
            record = json.loads(question_file.read_text())
            record.setdefault("question_id", question_file.stem)
            question_records.append(record)
    else:
        raise FileNotFoundError(f"Benchmark manifest not found: {benchmark_path}")

    for record in question_records:
        question_id = record.get("question_id")
        if not question_id or question_id in entries:
            raise ValueError(f"Missing or duplicate benchmark question ID: {question_id}")

        metadata = record.get("metadata", {})
        categories = record.get("categories", metadata.get("categories", []))
        if isinstance(categories, str):
            categories = [category.strip() for category in categories.split(",")]

        answer = extract_answer_letter(record.get("answer"))
        if answer is None:
            raise ValueError(f"Invalid benchmark answer for question ID: {question_id}")

        entries[question_id] = {
            "answer": answer,
            "categories": [category for category in categories if category != "reasoning"],
        }

    if len(entries) != 2500:
        raise ValueError(f"ChestAgentBench must contain exactly 2500 questions, found {len(entries)}")
    return entries


def analyze_medrax_results(
    results_file: str, benchmark_path: str, max_questions: Optional[int] = None
) -> Tuple[float, Dict, Dict, List[str], List[str]]:
    """Analyze MedRAX with the paper's fixed 2,500-question denominator."""
    if max_questions is not None:
        raise ValueError("--max-questions is incompatible with the MedRAX paper protocol")

    manifest = load_chestagentbench_manifest(benchmark_path)
    predictions: Dict[str, Dict] = {}
    with open(results_file, "r") as results:
        for line in results:
            if not line.strip() or line.startswith("HTTP Request:"):
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            question_id = entry.get("question_id")
            if not question_id:
                continue
            if question_id in predictions:
                raise ValueError(f"Duplicate terminal result for question ID: {question_id}")
            if question_id not in manifest:
                raise ValueError(f"Result question ID is absent from benchmark: {question_id}")
            predictions[question_id] = entry

    category_performance = defaultdict(lambda: {"total": 0, "correct": 0})
    correct_ids = []
    incorrect_ids = []

    for question_id, expected in manifest.items():
        entry = predictions.get(question_id, {})
        model_letter = extract_answer_letter(entry.get("model_answer"))
        correct_letter = extract_answer_letter(expected["answer"])
        is_correct = model_letter is not None and model_letter == correct_letter

        if is_correct:
            correct_ids.append(question_id)
        else:
            incorrect_ids.append(question_id)

        for category in expected["categories"]:
            category_performance[category]["total"] += 1
            if is_correct:
                category_performance[category]["correct"] += 1

    return process_results(
        category_performance,
        len(manifest),
        len(correct_ids),
        correct_ids,
        incorrect_ids,
    )


def analyze_llama_results(
    results_file: str, max_questions: Optional[int] = None
) -> Tuple[float, Dict, Dict, List[str], List[str]]:
    """
    Analyze results in Llama format.

    Args:
        results_file: Path to results file
        max_questions: Maximum number of questions to analyze

    Returns:
        Tuple containing:
            - overall_accuracy (float)
            - category_accuracies (Dict)
            - question_type_stats (Dict)
            - correct_ids (List[str])
            - incorrect_ids (List[str])
    """
    category_performance = defaultdict(lambda: {"total": 0, "correct": 0})
    all_questions = 0
    all_correct = 0
    correct_ids = []
    incorrect_ids = []

    with open(results_file, "r") as f:
        lines = f.readlines()

    # If max_questions is set, limit the number of lines processed
    if max_questions is not None:
        lines = lines[:max_questions]

    for line in tqdm(lines, desc="Analyzing Benchmark Results"):
        if line.startswith("HTTP Request:"):
            continue

        try:
            entry = json.loads(line)
            metadata = entry.get("input", {}).get("question_data", {}).get("metadata", {})
            question_id = entry.get("question_id")

            model_letter = extract_answer_letter(entry.get("model_answer"))
            correct_letter = extract_answer_letter(entry.get("correct_answer"))

            if model_letter and correct_letter:
                all_questions += 1
                is_correct = model_letter == correct_letter

                if is_correct:
                    all_correct += 1
                    correct_ids.append(question_id)
                else:
                    incorrect_ids.append(question_id)

                for category in metadata.get("categories", []):
                    category_performance[category]["total"] += 1
                    if is_correct:
                        category_performance[category]["correct"] += 1

        except json.JSONDecodeError:
            continue

    return process_results(
        category_performance, all_questions, all_correct, correct_ids, incorrect_ids
    )


def analyze_chexagent_results(
    results_file: str, max_questions: Optional[int] = None
) -> Tuple[float, Dict, Dict, List[str], List[str]]:
    """
    Analyze results in CheXagent format.

    Args:
        results_file: Path to results file
        max_questions: Maximum number of questions to analyze

    Returns:
        Tuple containing:
            - overall_accuracy (float)
            - category_accuracies (Dict)
            - question_type_stats (Dict)
            - correct_ids (List[str])
            - incorrect_ids (List[str])
    """
    category_performance = defaultdict(lambda: {"total": 0, "correct": 0})
    all_questions = 0
    all_correct = 0
    correct_ids = []
    incorrect_ids = []

    with open(results_file, "r") as f:
        lines = f.readlines()

    # If max_questions is set, limit the number of lines processed
    if max_questions is not None:
        lines = lines[:max_questions]

    for line in tqdm(lines, desc="Analyzing Benchmark Results"):
        try:
            entry = json.loads(line)
            metadata = entry.get("input", {}).get("question_data", {}).get("metadata", {})
            question_id = entry.get("question_id")

            model_letter = extract_answer_letter(entry.get("model_answer"))
            correct_letter = extract_answer_letter(entry.get("correct_answer"))

            if model_letter and correct_letter:
                all_questions += 1
                is_correct = model_letter == correct_letter

                if is_correct:
                    all_correct += 1
                    correct_ids.append(question_id)
                else:
                    incorrect_ids.append(question_id)

                for category in metadata.get("categories", []):
                    category_performance[category]["total"] += 1
                    if is_correct:
                        category_performance[category]["correct"] += 1

        except json.JSONDecodeError:
            continue

    return process_results(
        category_performance, all_questions, all_correct, correct_ids, incorrect_ids
    )


def process_results(
    category_performance: Dict,
    all_questions: int,
    all_correct: int,
    correct_ids: Optional[List[str]] = None,
    incorrect_ids: Optional[List[str]] = None,
) -> Tuple[float, Dict, Dict, List[str], List[str]]:
    """
    Process raw results into final statistics.

    Args:
        category_performance: Dict containing performance by category
        all_questions: Total number of questions
        all_correct: Total number of correct answers
        correct_ids: List of IDs for correctly answered questions
        incorrect_ids: List of IDs for incorrectly answered questions

    Returns:
        Tuple containing:
            - overall_accuracy (float)
            - category_accuracies (Dict)
            - question_type_stats (Dict)
            - correct_ids (List[str])
            - incorrect_ids (List[str])
    """
    category_accuracies = {
        category: {
            "accuracy": stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0,
            "total": stats["total"],
            "correct": stats["correct"],
        }
        for category, stats in category_performance.items()
    }

    question_type_stats = {}
    for qtype, categories in QUESTION_TYPES.items():
        total = sum(
            category_performance[cat]["total"] for cat in categories if cat in category_performance
        )
        correct = sum(
            category_performance[cat]["correct"]
            for cat in categories
            if cat in category_performance
        )

        question_type_stats[qtype] = {
            "accuracy": (correct / total * 100) if total > 0 else 0,
            "total": total,
            "correct": correct,
        }

    overall_accuracy = (all_correct / all_questions * 100) if all_questions > 0 else 0

    return (
        overall_accuracy,
        category_accuracies,
        question_type_stats,
        correct_ids or [],
        incorrect_ids or [],
    )


def print_analysis(
    overall_accuracy: float,
    category_accuracies: Dict,
    question_type_stats: Dict,
    correct_ids: List[str],
    incorrect_ids: List[str],
    model_name: str,
) -> None:
    """
    Print analysis results.

    Args:
        overall_accuracy: Overall accuracy percentage
        category_accuracies: Dict containing accuracy metrics by category
        question_type_stats: Dict containing stats by question type
        correct_ids: List of IDs for correctly answered questions
        incorrect_ids: List of IDs for incorrectly answered questions
        model_name: Name of the model being analyzed
    """
    total_questions = len(correct_ids) + len(incorrect_ids)
    print(
        f"\nOverall Accuracy: {overall_accuracy:.2f}% ({len(correct_ids)} correct out of {total_questions} questions)"
    )

    print("\nCategory Performance:")
    sorted_categories = sorted(
        category_accuracies.items(), key=lambda x: x[1]["accuracy"], reverse=True
    )
    for category, metrics in sorted_categories:
        print(f"{category}:")
        print(f"  Accuracy: {metrics['accuracy']:.2f}%")
        print(f"  Total Questions: {metrics['total']}")
        print(f"  Correct Questions: {metrics['correct']}")

    print("\nQuestion Type Performance:")
    sorted_types = sorted(question_type_stats.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    for qtype, metrics in sorted_types:
        print(f"\n{qtype}:")
        print(f"  Accuracy: {metrics['accuracy']:.2f}%")
        print(f"  Total Questions: {metrics['total']}")
        print(f"  Correct Questions: {metrics['correct']}")
        print(f"  Categories: {', '.join(QUESTION_TYPES[qtype])}")

    # Save question IDs to JSON
    question_ids = {"correct_ids": correct_ids, "incorrect_ids": incorrect_ids}

    output_filename = f"{model_name}_question_ids.json"
    with open(output_filename, "w") as f:
        json.dump(question_ids, f, indent=2)

    print(f"\nQuestion IDs have been saved to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("results_file", help="Path to results file")
    parser.add_argument("benchmark_dir", nargs="?", help="Path to benchmark questions directory")
    parser.add_argument(
        "--model",
        choices=["llava-med", "chexagent", "llama", "gpt4", "medrax"],
        default="gpt4",
        help="Specify model format (default: gpt4)",
    )
    parser.add_argument("--max-questions", type=int, help="Maximum number of questions to analyze")
    args = parser.parse_args()

    if args.model == "gpt4":
        results = analyze_gpt4_results(args.results_file, args.max_questions)
    elif args.model == "llama":
        results = analyze_llama_results(args.results_file, args.max_questions)
    elif args.model == "chexagent":
        results = analyze_chexagent_results(args.results_file, args.max_questions)
    elif args.model == "medrax":
        if not args.benchmark_dir:
            parser.error("--model medrax requires benchmark_dir or metadata.jsonl")
        results = analyze_medrax_results(
            args.results_file, args.benchmark_dir, args.max_questions
        )
    else:
        parser.error(f"Unsupported model: {args.model}")

    print_analysis(*results, args.model)
