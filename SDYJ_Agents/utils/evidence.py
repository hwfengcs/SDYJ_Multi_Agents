"""Evidence normalization and report-grounding helpers."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence
from urllib.parse import urlparse, urlunparse


EvidenceItem = Dict[str, Any]


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


def format_evidence_for_prompt(
    evidence_items: Sequence[EvidenceItem],
    limit: int = 30,
    char_budget: int = 12000,
) -> str:
    """Render evidence with stable IDs for LLM synthesis prompts."""
    selected = select_evidence_for_prompt(evidence_items, limit=limit, char_budget=char_budget)
    return "\n".join(_render_evidence_line(item) for item in selected)


def _render_evidence_line(item: EvidenceItem, snippet_chars: int = 350) -> str:
    evidence_id = item.get("evidence_id", "E?")
    title = item.get("title", "Untitled")
    source = item.get("source", "unknown")
    url = item.get("url") or "N/A"
    snippet = str(item.get("snippet", ""))[:snippet_chars]
    return f"- [{evidence_id}] ({source}) {title} | {url}\n  {snippet}"


def _relevance_value(item: EvidenceItem) -> float:
    relevance = item.get("relevance_score")
    if isinstance(relevance, (int, float)):
        return float(relevance)
    return 0.0


def select_evidence_for_prompt(
    evidence_items: Sequence[EvidenceItem],
    limit: int = 30,
    char_budget: int = 12000,
) -> List[EvidenceItem]:
    """Pick the evidence subset an LLM prompt should see.

    Ordering is deterministic: relevance score desc, published date desc,
    original insertion order as the final tiebreak. Items are accepted in that
    order until either the item limit or the rendered-character budget is hit,
    so low-relevance tails can no longer crowd out high-relevance evidence.
    """
    indexed = list(enumerate(evidence_items))
    # Stable multi-pass sort: least-significant key first.
    indexed.sort(key=lambda pair: str(pair[1].get("published_date") or ""), reverse=True)
    indexed.sort(key=lambda pair: _relevance_value(pair[1]), reverse=True)

    selected: List[EvidenceItem] = []
    used_chars = 0
    for _, item in indexed:
        if len(selected) >= limit:
            break
        rendered_chars = len(_render_evidence_line(item))
        if selected and used_chars + rendered_chars > char_budget:
            break
        selected.append(item)
        used_chars += rendered_chars
    return selected


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


def valid_evidence_ids(evidence_items: Sequence[EvidenceItem]) -> set:
    """Return the set of evidence ids that actually exist."""
    return {
        str(item.get("evidence_id"))
        for item in evidence_items
        if item.get("evidence_id")
    }


def validate_citations(
    text: str,
    evidence_items: Sequence[EvidenceItem],
) -> tuple[str, Dict[str, Any]]:
    """Strip citation ids that do not exist in the evidence set.

    Returns the cleaned text plus stats about total/valid/invalid mentions, so
    fabricated ids can never survive into the delivered report nor inflate
    grounding metrics.
    """
    valid_ids = valid_evidence_ids(evidence_items)
    stats = {
        "total_citation_mentions": 0,
        "valid_citation_mentions": 0,
        "invalid_citation_ids": [],
    }

    def _check(match: re.Match) -> str:
        stats["total_citation_mentions"] += 1
        evidence_id = f"E{match.group(1)}"
        if evidence_id in valid_ids:
            stats["valid_citation_mentions"] += 1
            return match.group(0)
        if evidence_id not in stats["invalid_citation_ids"]:
            stats["invalid_citation_ids"].append(evidence_id)
        return ""

    cleaned = re.sub(r" ?\[E(\d+)\]", _check, text or "")
    total = stats["total_citation_mentions"]
    stats["citation_validity_rate"] = (
        stats["valid_citation_mentions"] / total if total else 1.0
    )
    return cleaned, stats


def _is_reference_list_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("- [E") or '"citation"' in stripped


def extract_body_citations(report: str) -> List[str]:
    """Collect ``[E#]`` mentions outside the reference list.

    Reference-list lines (``- [E1] title ...`` in Markdown, ``"citation"``
    entries in JSON) enumerate every evidence id by construction, so counting
    them would make citation coverage self-fulfilling.
    """
    mentions: List[str] = []
    for line in (report or "").splitlines():
        if _is_reference_list_line(line):
            continue
        mentions.extend(re.findall(r"\[E\d+\]", line))
    return mentions


def calculate_evidence_metrics(
    research_results: Sequence[Dict[str, Any]],
    evidence_items: Sequence[EvidenceItem],
    report: str = "",
) -> Dict[str, Any]:
    """Calculate lightweight grounding metrics for reports and evals.

    Citation metrics only count ids that exist in the evidence set and only
    look at report-body mentions, so neither a fabricated ``[E99]`` nor the
    auto-generated reference list can inflate coverage.
    """
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
    valid_ids = valid_evidence_ids(evidence_items)
    all_mentions = re.findall(r"\[E\d+\]", report or "")
    valid_mention_count = sum(1 for m in all_mentions if m[1:-1] in valid_ids)
    body_mentions = extract_body_citations(report)
    citation_ids = {m for m in body_mentions if m[1:-1] in valid_ids}
    key_finding_lines = [
        line for line in (report or "").splitlines()
        if line.strip().startswith("- ")
        and "##" not in line
        and not _is_reference_list_line(line)
    ]
    cited_key_finding_lines = [
        line for line in key_finding_lines
        if any(citation in line for citation in citation_ids)
    ]

    return {
        "raw_url_count": len(raw_urls),
        "duplicate_url_count": duplicate_urls,
        "duplicate_url_ratio": duplicate_urls / len(raw_urls) if raw_urls else 0.0,
        "evidence_count": len(evidence_items),
        "citation_count": len(citation_ids),
        "citation_validity_rate": (
            valid_mention_count / len(all_mentions) if all_mentions else 1.0
        ),
        "invalid_citation_count": len(all_mentions) - valid_mention_count,
        "citation_density_per_1k_chars": (
            len(citation_ids) / max(len(report), 1) * 1000 if report else 0.0
        ),
        "tool_success_rate": successful_batches / total_batches if total_batches else 0.0,
        "grounded_key_finding_rate": (
            len(cited_key_finding_lines) / len(key_finding_lines) if key_finding_lines else 0.0
        ),
    }
