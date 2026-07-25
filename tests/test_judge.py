"""Tests for the LLM-as-judge faithfulness scorer."""

import json

from SDYJ_Agents.evaluation.judge import (
    extract_cited_claims,
    run_faithfulness_judge,
)

EVIDENCE = [
    {"evidence_id": "E1", "title": "Doc one", "snippet": "snippet one"},
    {"evidence_id": "E2", "title": "Doc two", "snippet": "snippet two"},
]

REPORT = (
    "## 核心发现\n"
    "- 论断一 [E1]\n"
    "- 论断二 [E2]\n"
    "- 无引用论断\n"
    "- 假引用论断 [E9]\n"
    "## 参考资料\n"
    "- [E1] Doc one\n"
    "- [E2] Doc two\n"
)


class VerdictLLM:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        return self.response


def test_extract_cited_claims_skips_uncited_invalid_and_reference_lines():
    claims = extract_cited_claims(REPORT, EVIDENCE)

    assert [claim["text"] for claim in claims] == ["论断一 [E1]", "论断二 [E2]"]
    assert claims[0]["cited_ids"] == ["E1"]
    assert claims[1]["claim_index"] == 2


def test_extract_cited_claims_respects_max_claims():
    report = "\n".join(f"- 论断{i} [E1]" for i in range(10))
    claims = extract_cited_claims(report, EVIDENCE, max_claims=3)
    assert len(claims) == 3


def test_judge_scores_supported_partial_unsupported():
    response = json.dumps(
        {
            "default_verdict": "unsupported",
            "verdicts": [
                {"claim_index": 1, "verdict": "supported", "reason": "ok"},
                {"claim_index": 2, "verdict": "partial", "reason": "half"},
            ],
        }
    )
    llm = VerdictLLM(response)

    result = run_faithfulness_judge(REPORT, EVIDENCE, llm)

    assert result["judged_claim_count"] == 2
    assert result["faithfulness_score"] == 0.75  # (1 + 0.5) / 2
    assert result["citation_precision"] == 1.0  # (1 + 1) / 2
    assert result["judge_verdicts"][0]["verdict"] == "supported"
    # One batched call carrying claims and their cited evidence only.
    assert len(llm.prompts) == 1
    assert "[PROMPT_ID: judge_faithfulness]" in llm.prompts[0]
    assert "Doc one" in llm.prompts[0]


def test_judge_uses_default_verdict_for_missing_claims():
    response = json.dumps(
        {
            "default_verdict": "supported",
            "verdicts": [{"claim_index": 1, "verdict": "partial", "reason": "canned"}],
        }
    )
    result = run_faithfulness_judge(REPORT, EVIDENCE, VerdictLLM(response))

    # claim 2 falls back to default_verdict=supported.
    assert result["faithfulness_score"] == 0.75
    assert result["citation_precision"] == 1.0


def test_judge_malformed_response_scores_unsupported():
    result = run_faithfulness_judge(REPORT, EVIDENCE, VerdictLLM("not json at all"))

    assert result["judged_claim_count"] == 2
    assert result["faithfulness_score"] == 0.0
    assert result["citation_precision"] == 0.0


def test_judge_with_no_citable_claims_scores_zero_without_llm_call():
    llm = VerdictLLM("should never be called")
    result = run_faithfulness_judge("完全没有引用的报告", EVIDENCE, llm)

    assert result["judged_claim_count"] == 0
    assert result["faithfulness_score"] == 0.0
    assert llm.prompts == []
