"""LLM-as-judge faithfulness scoring for cited report claims.

The judge grades whether each sampled claim is actually supported by the
evidence it cites — the semantic check that deterministic citation metrics
cannot do. It runs dual-track: offline benchmarks use FakeEvalLLM's canned
verdict (CI stays deterministic and key-free), ``--live`` grades with the real
provider. Kept OUT of ``overall_score`` on purpose: it reports as its own
dimension with its own threshold so a fixed weighted blend cannot be gamed.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Sequence

from ..llm.base import BaseLLM
from ..prompts.loader import PromptLoader
from ..utils.evidence import valid_evidence_ids

_CITATION_RE = re.compile(r"\[E\d+\]")

VERDICT_SUPPORTED = "supported"
VERDICT_PARTIAL = "partial"
VERDICT_UNSUPPORTED = "unsupported"
_VALID_VERDICTS = {VERDICT_SUPPORTED, VERDICT_PARTIAL, VERDICT_UNSUPPORTED}


def extract_cited_claims(
    report: str,
    evidence_items: Sequence[Dict[str, Any]],
    max_claims: int = 6,
) -> List[Dict[str, Any]]:
    """Collect key-finding bullets that cite valid evidence, in report order.

    Selection is deterministic (no RNG): body bullets only — reference-list
    lines (``- [E...``) are skipped — capped at ``max_claims``.
    """
    valid_ids = valid_evidence_ids(evidence_items)
    claims: List[Dict[str, Any]] = []
    for line in (report or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith("- ") or stripped.startswith("- [E"):
            continue
        cited = [m[1:-1] for m in _CITATION_RE.findall(stripped) if m[1:-1] in valid_ids]
        if not cited:
            continue
        claims.append(
            {
                "claim_index": len(claims) + 1,
                "text": stripped[2:].strip(),
                "cited_ids": cited,
            }
        )
        if len(claims) >= max_claims:
            break
    return claims


def _format_claims_block(claims: List[Dict[str, Any]]) -> str:
    lines = []
    for claim in claims:
        cited = ", ".join(claim["cited_ids"])
        lines.append(f"{claim['claim_index']}. {claim['text']}（引用：{cited}）")
    return "\n".join(lines)


def _format_evidence_block(
    claims: List[Dict[str, Any]],
    evidence_items: Sequence[Dict[str, Any]],
) -> str:
    cited_ids = {evidence_id for claim in claims for evidence_id in claim["cited_ids"]}
    by_id = {
        str(item.get("evidence_id")): item
        for item in evidence_items
        if item.get("evidence_id")
    }
    lines = []
    for evidence_id in sorted(cited_ids, key=lambda eid: int(eid[1:])):
        item = by_id.get(evidence_id)
        if not item:
            continue
        snippet = str(item.get("snippet", ""))[:400]
        lines.append(f"[{evidence_id}] {item.get('title', 'Untitled')}\n  {snippet}")
    return "\n".join(lines)


def _parse_verdicts(response: str, claims: List[Dict[str, Any]]) -> Dict[int, str]:
    default = VERDICT_UNSUPPORTED
    by_index: Dict[int, str] = {}
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        payload = json.loads(response[start:end])
        parsed_default = str(payload.get("default_verdict", "")).lower()
        if parsed_default in _VALID_VERDICTS:
            default = parsed_default
        for entry in payload.get("verdicts", []):
            verdict = str(entry.get("verdict", "")).lower()
            if verdict in _VALID_VERDICTS:
                by_index[int(entry.get("claim_index", 0))] = verdict
    except (ValueError, TypeError, AttributeError, json.JSONDecodeError):
        pass
    return {
        claim["claim_index"]: by_index.get(claim["claim_index"], default)
        for claim in claims
    }


def run_faithfulness_judge(
    report: str,
    evidence_items: Sequence[Dict[str, Any]],
    llm: BaseLLM,
    max_claims: int = 6,
) -> Dict[str, Any]:
    """Grade sampled cited claims against their evidence with ONE batched call.

    Returns ``faithfulness_score`` ((supported + 0.5*partial) / n),
    ``citation_precision`` ((supported + partial) / n), the judged claim count,
    and per-claim verdicts. Zero judgeable claims scores 0.0 — an uncited
    report must not pass a faithfulness gate by default.
    """
    claims = extract_cited_claims(report, evidence_items, max_claims=max_claims)
    if not claims:
        return {
            "judged_claim_count": 0,
            "faithfulness_score": 0.0,
            "citation_precision": 0.0,
            "judge_verdicts": [],
        }

    prompt = PromptLoader().load(
        "judge_faithfulness",
        claims_block=_format_claims_block(claims),
        evidence_block=_format_evidence_block(claims, evidence_items),
    )
    response = llm.generate(prompt, temperature=0.0, max_tokens=1200)
    verdict_by_index = _parse_verdicts(response, claims)

    supported = sum(1 for v in verdict_by_index.values() if v == VERDICT_SUPPORTED)
    partial = sum(1 for v in verdict_by_index.values() if v == VERDICT_PARTIAL)
    total = len(claims)
    return {
        "judged_claim_count": total,
        "faithfulness_score": round((supported + 0.5 * partial) / total, 4),
        "citation_precision": round((supported + partial) / total, 4),
        "judge_verdicts": [
            {
                "claim_index": claim["claim_index"],
                "claim": claim["text"],
                "cited_ids": claim["cited_ids"],
                "verdict": verdict_by_index[claim["claim_index"]],
            }
            for claim in claims
        ],
    }
