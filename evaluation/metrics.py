import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

SOURCE_PATTERN = re.compile(r"\[SOURCE=(.+?);\s*SECTION=(.+?)\]")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def extract_citations(text: str) -> List[Dict[str, str]]:
    citations: List[Dict[str, str]] = []
    for group in SOURCE_PATTERN.findall(text):
        citations.append({"act": group[0].strip(), "section": group[1].strip()})
    return citations


def is_expected_source_present(actual: Dict[str, Any], expected: Dict[str, str]) -> bool:
    return (
        normalize_text(actual.get("act", "")) == normalize_text(expected.get("act", ""))
        and normalize_text(actual.get("section", "")) == normalize_text(expected.get("section", ""))
    )


def calculate_source_recall(retrieved_sources: List[Dict[str, Any]], expected_sources: List[Dict[str, str]]) -> float:
    if not expected_sources:
        return 1.0
    hits = 0
    for expected in expected_sources:
        if any(is_expected_source_present(doc, expected) for doc in retrieved_sources):
            hits += 1
    return hits / len(expected_sources)


def calculate_citation_coverage(answer: str, expected_sources: List[Dict[str, str]]) -> float:
    if not expected_sources:
        return 1.0
    citations = extract_citations(answer)
    if not citations:
        return 0.0
    hits = 0
    for expected in expected_sources:
        if any(is_expected_source_present(citation, expected) for citation in citations):
            hits += 1
    return hits / len(expected_sources)


def answer_quality(answer: str, ground_truth: str) -> float:
    return similarity(answer, ground_truth)
