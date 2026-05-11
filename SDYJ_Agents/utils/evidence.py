"""Evidence normalization and report-grounding helpers."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence
from urllib.parse import urlparse, urlunparse


EvidenceItem = Dict[str, Any]
CITATION_RE = re.compile(r"\[E\d+\]")


def normalize_url(url: str | None) -> str:
    """Normalize URLs enough for duplicate detection without changing meaning."""
    if not url:
        return ""

    parsed = urlparse(url.strip())
    if not parsed.scheme or not parsed.netloc:
        return url.strip().rstrip("/")

    normalized = parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower(),
        fragment="",
    )
    return urlunparse(normalized).rstrip("/")


def extract_domain(url: str | None) -> str | None:
    """Extract a compact domain for source quality displays."""
    if not url:
        return None
    parsed = urlparse(url)
    return parsed.netloc.lower() or None


def extract_published_date(metadata: Dict[str, Any] | None) -> str | None:
    """Return the best available publication-like date from tool metadata."""
    if not metadata:
        return None
    for key in ("published_date", "published", "updated", "submitted"):
        value = metadata.get(key)
        if value:
            return str(value)
    return None


def evidence_key(item: EvidenceItem) -> str:
    """Build a stable duplicate key for an evidence item."""
    url = normalize_url(item.get("url"))
    if url:
        return f"url:{url}"
    source = item.get("source", "unknown")
    title = str(item.get("title", "")).strip().lower()
    return f"title:{source}:{title}"


def _next_evidence_number(existing: Sequence[EvidenceItem]) -> int:
    numbers = []
    for item in existing:
        evidence_id = str(item.get("evidence_id", ""))
        if evidence_id.startswith("E") and evidence_id[1:].isdigit():
            numbers.append(int(evidence_id[1:]))
    return max(numbers, default=0) + 1


def normalize_search_batch(batch: Dict[str, Any]) -> List[EvidenceItem]:
    """Convert one raw search batch into evidence items."""
    evidence = []
    metadata_source = batch.get("source", "unknown")

    for item in batch.get("results", []) or []:
        metadata = item.get("metadata") or {}
        url = item.get("url") or ""
        evidence.append(
            {
                "evidence_id": None,
                "task_id": batch.get("task_id"),
                "query": batch.get("query"),
                "source": metadata_source,
                "title": item.get("title") or "Untitled",
                "url": url,
                "normalized_url": normalize_url(url),
                "domain": extract_domain(url),
                "snippet": item.get("snippet") or item.get("summary") or "",
                "relevance_score": item.get("relevance_score"),
                "published_date": extract_published_date(metadata),
                "metadata": metadata,
            }
        )

    return evidence


def merge_evidence_items(
    existing: Sequence[EvidenceItem],
    new_items: Iterable[EvidenceItem],
) -> List[EvidenceItem]:
    """Merge new evidence into existing evidence, deduplicating by URL or title."""
    merged = [dict(item) for item in existing]
    seen = {evidence_key(item) for item in merged}
    next_number = _next_evidence_number(merged)

    for raw_item in new_items:
        item = dict(raw_item)
        key = evidence_key(item)
        if key in seen:
            continue
        item["evidence_id"] = item.get("evidence_id") or f"E{next_number}"
        next_number += 1
        seen.add(key)
        merged.append(item)

    return merged


def build_evidence_from_results(results: Sequence[Dict[str, Any]]) -> List[EvidenceItem]:
    """Build a deduplicated evidence list from raw research results."""
    evidence: List[EvidenceItem] = []
    for batch in results:
        evidence = merge_evidence_items(evidence, normalize_search_batch(batch))
    return evidence


def format_evidence_for_prompt(evidence_items: Sequence[EvidenceItem], limit: int = 30) -> str:
    """Render evidence with stable IDs for LLM synthesis prompts."""
    lines = []
    for item in evidence_items[:limit]:
        evidence_id = item.get("evidence_id", "E?")
        title = item.get("title", "Untitled")
        source = item.get("source", "unknown")
        url = item.get("url") or "N/A"
        snippet = str(item.get("snippet", ""))[:350]
        lines.append(f"- [{evidence_id}] ({source}) {title} | {url}\n  {snippet}")
    return "\n".join(lines)


def select_evidence_ids_for_text(
    text: str,
    evidence_items: Sequence[EvidenceItem],
    max_ids: int = 2,
) -> List[str]:
    """Select a small set of likely supporting evidence IDs for a claim."""
    if not evidence_items:
        return []

    text_l = text.lower()
    tokens = re.findall(r"[a-zA-Z0-9_\-]{3,}", text_l)
    token_counts = Counter(tokens)
    scored = []

    for index, item in enumerate(evidence_items):
        haystack = " ".join(
            [
                str(item.get("title", "")),
                str(item.get("snippet", "")),
                str(item.get("query", "")),
                str(item.get("source", "")),
            ]
        ).lower()
        score = 0.0
        for token, count in token_counts.items():
            if token in haystack:
                score += count
        if str(item.get("source", "")).lower() in text_l:
            score += 1.0
        relevance = item.get("relevance_score")
        if isinstance(relevance, (int, float)):
            score += min(float(relevance), 1.0)
        scored.append((score, index, str(item.get("evidence_id", ""))))

    scored.sort(key=lambda row: (-row[0], row[1]))
    selected = [evidence_id for score, _, evidence_id in scored if evidence_id and score > 0]
    if not selected:
        selected = [str(item.get("evidence_id")) for item in evidence_items[:max_ids]]
    return selected[:max_ids]


def append_citations(text: str, evidence_items: Sequence[EvidenceItem], max_ids: int = 2) -> str:
    """Append stable evidence citations to a report claim if missing."""
    if re.search(r"\[E\d+\]", text):
        return text
    ids = select_evidence_ids_for_text(text, evidence_items, max_ids=max_ids)
    if not ids:
        return text
    return f"{text} {' '.join(f'[{evidence_id}]' for evidence_id in ids)}"


def _evidence_id_sort_key(evidence_id: str) -> tuple[int, str]:
    if evidence_id.startswith("E") and evidence_id[1:].isdigit():
        return (int(evidence_id[1:]), evidence_id)
    return (10**9, evidence_id)


def _ordered_unique(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def audit_report_citations(
    report: str,
    evidence_items: Sequence[EvidenceItem],
) -> Dict[str, Any]:
    """Deterministically audit report citations against collected evidence.

    The LLM verifier judges semantic support, but citation syntax and evidence
    coverage should be checked without model discretion. This catches common
    report failures such as citing ``[E99]`` when only ``E1``/``E2`` exist, or
    key-finding bullets that have no valid evidence ID.
    """
    report = report or ""
    valid_evidence_ids = sorted(
        {
            str(item.get("evidence_id"))
            for item in evidence_items
            if item.get("evidence_id")
        },
        key=_evidence_id_sort_key,
    )
    valid_set = set(valid_evidence_ids)
    cited_ids = _ordered_unique(match.strip("[]") for match in CITATION_RE.findall(report))
    valid_cited_ids = [evidence_id for evidence_id in cited_ids if evidence_id in valid_set]
    invalid_citation_ids = [evidence_id for evidence_id in cited_ids if evidence_id not in valid_set]
    unused_evidence_ids = [
        evidence_id for evidence_id in valid_evidence_ids if evidence_id not in set(valid_cited_ids)
    ]

    key_finding_lines = [
        line.strip()
        for line in report.splitlines()
        if line.strip().startswith("- ") and "##" not in line
    ]
    supported_key_finding_lines = []
    unsupported_key_finding_lines = []
    for line in key_finding_lines:
        line_citations = {match.strip("[]") for match in CITATION_RE.findall(line)}
        if line_citations & valid_set:
            supported_key_finding_lines.append(line)
        else:
            unsupported_key_finding_lines.append(line)

    evidence_count = len(valid_evidence_ids)
    cited_count = len(cited_ids)
    valid_cited_count = len(set(valid_cited_ids))
    return {
        "valid_evidence_ids": valid_evidence_ids,
        "cited_evidence_ids": cited_ids,
        "valid_cited_evidence_ids": sorted(set(valid_cited_ids), key=_evidence_id_sort_key),
        "invalid_citation_ids": invalid_citation_ids,
        "unused_evidence_ids": unused_evidence_ids,
        "evidence_count": evidence_count,
        "citation_count": cited_count,
        "valid_citation_count": valid_cited_count,
        "invalid_citation_count": len(invalid_citation_ids),
        "citation_id_coverage": valid_cited_count / evidence_count if evidence_count else 0.0,
        "citation_validity": valid_cited_count / cited_count if cited_count else 1.0,
        "key_finding_count": len(key_finding_lines),
        "supported_key_finding_count": len(supported_key_finding_lines),
        "unsupported_key_finding_count": len(unsupported_key_finding_lines),
        "unsupported_key_finding_examples": unsupported_key_finding_lines[:5],
        "grounded_key_finding_rate": (
            len(supported_key_finding_lines) / len(key_finding_lines)
            if key_finding_lines else 0.0
        ),
        "citation_audit_passed": not invalid_citation_ids and not unsupported_key_finding_lines,
    }


def calculate_evidence_metrics(
    research_results: Sequence[Dict[str, Any]],
    evidence_items: Sequence[EvidenceItem],
    report: str = "",
) -> Dict[str, Any]:
    """Calculate lightweight grounding metrics for reports and evals."""
    raw_urls = []
    successful_batches = 0
    total_batches = len(research_results)

    for batch in research_results:
        if not batch.get("error"):
            successful_batches += 1
        for item in batch.get("results", []) or []:
            url = normalize_url(item.get("url"))
            if url:
                raw_urls.append(url)

    duplicate_urls = len(raw_urls) - len(set(raw_urls))
    citation_audit = audit_report_citations(report, evidence_items)

    return {
        "raw_url_count": len(raw_urls),
        "duplicate_url_count": duplicate_urls,
        "duplicate_url_ratio": duplicate_urls / len(raw_urls) if raw_urls else 0.0,
        "evidence_count": len(evidence_items),
        "citation_count": citation_audit["citation_count"],
        "valid_citation_count": citation_audit["valid_citation_count"],
        "invalid_citation_count": citation_audit["invalid_citation_count"],
        "citation_validity": citation_audit["citation_validity"],
        "citation_evidence_coverage": citation_audit["citation_id_coverage"],
        "unused_evidence_count": len(citation_audit["unused_evidence_ids"]),
        "unsupported_key_finding_count": citation_audit["unsupported_key_finding_count"],
        "citation_audit_passed": citation_audit["citation_audit_passed"],
        "citation_density_per_1k_chars": (
            citation_audit["valid_citation_count"] / max(len(report), 1) * 1000
            if report else 0.0
        ),
        "tool_success_rate": successful_batches / total_batches if total_batches else 0.0,
        "grounded_key_finding_rate": citation_audit["grounded_key_finding_rate"],
    }
