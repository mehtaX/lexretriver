import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from evaluation.metrics import (
    answer_quality,
    calculate_citation_coverage,
    calculate_source_recall,
)
from rag_pipeline import get_rag_instance

ROOT = Path(__file__).parent.resolve()
GOLDEN_DATA_PATH = ROOT / "golden_dataset.json"


def load_golden_dataset() -> List[Dict[str, Any]]:
    with open(GOLDEN_DATA_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def format_source_items(docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [
        {
            "act": item.get("act", "Unknown"),
            "section": item.get("section", "Unknown"),
            "source": item.get("source", "Unknown"),
            "page": item.get("page", "Unknown"),
        }
        for item in docs
    ]


def evaluate_one_question(rag, question: str, expected_sources: List[Dict[str, str]], ground_truth: str, generate: bool) -> Dict[str, Any]:
    evaluation = {
        "question": question,
        "expected_sources": expected_sources,
        "retrieval_recall": 0.0,
        "citation_coverage": 0.0,
        "answer_score": 0.0,
        "sources": [],
        "answer": "",
    }

    retrieved_docs = rag.retrieve_and_rerank(question)
    retrieved_sources = [
        {
            "act": doc.metadata.get("act", "Unknown"),
            "section": doc.metadata.get("section", "Unknown"),
            "source": doc.metadata.get("source", "Unknown"),
            "page": str(doc.metadata.get("page", "Unknown")),
        }
        for doc in retrieved_docs
    ]

    evaluation["sources"] = format_source_items(retrieved_sources)
    evaluation["retrieval_recall"] = calculate_source_recall(retrieved_sources, expected_sources)

    if generate:
        result = rag.ask(question)
        evaluation["answer"] = result.get("answer", "")
        evaluation["citation_coverage"] = calculate_citation_coverage(result.get("answer", ""), expected_sources)
        evaluation["answer_score"] = answer_quality(result.get("answer", ""), ground_truth)

    return evaluation


def run_evaluation(generate: bool, ci_mode: bool) -> int:
    rag = get_rag_instance()
    dataset = load_golden_dataset()

    records = []
    for item in dataset:
        records.append(
            evaluate_one_question(
                rag,
                item["question"],
                item.get("expected_sources", []),
                item.get("ground_truth", ""),
                generate,
            )
        )

    total = len(records)
    retrieval_hit = sum(1 for row in records if row["retrieval_recall"] >= 1.0) / total
    average_recall = sum(row["retrieval_recall"] for row in records) / total
    average_citation = sum(row["citation_coverage"] for row in records) / total
    average_answer_score = sum(row["answer_score"] for row in records) / total

    metrics = {
        "total_questions": total,
        "retrieval_precision_at_1": retrieval_hit,
        "average_source_recall": average_recall,
        "average_citation_coverage": average_citation,
        "average_answer_similarity": average_answer_score,
        "metrics_version": "1.0",
    }

    print(json.dumps(metrics, indent=2))

    if ci_mode:
        threshold_failures = []
        if retrieval_hit < 0.7:
            threshold_failures.append("retrieval_precision_at_1 < 0.7")
        if average_recall < 0.75:
            threshold_failures.append("average_source_recall < 0.75")
        if generate and average_citation < 0.70:
            threshold_failures.append("average_citation_coverage < 0.70")
        if generate and average_answer_score < 0.45:
            threshold_failures.append("average_answer_similarity < 0.45")

        if threshold_failures:
            print("CI gating failed:", ", ".join(threshold_failures))
            return 1

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Legal RAG retrieval and generation quality.")
    parser.add_argument("--generate", action="store_true", help="Generate answers during evaluation.")
    parser.add_argument("--ci", action="store_true", help="Exit non-zero when quality thresholds are not met.")
    args = parser.parse_args()

    code = run_evaluation(generate=args.generate, ci_mode=args.ci)
    raise SystemExit(code)
