from SDYJ_Agents.utils.evidence import (
    audit_report_citations,
    build_evidence_from_results,
    calculate_evidence_metrics,
    normalize_url,
)


def test_normalize_url_ignores_fragment_and_trailing_slash():
    assert normalize_url("HTTPS://Example.com/path/#section") == "https://example.com/path"


def test_build_evidence_deduplicates_urls_and_assigns_ids():
    results = [
        {
            "task_id": 1,
            "query": "agent eval",
            "source": "tavily",
            "results": [
                {"title": "A", "url": "https://example.com/a", "snippet": "one"},
                {"title": "A copy", "url": "https://example.com/a#frag", "snippet": "two"},
            ],
        },
        {
            "task_id": 1,
            "query": "agent eval",
            "source": "arxiv",
            "results": [
                {"title": "B", "url": "https://arxiv.org/abs/1", "snippet": "three"},
            ],
        },
    ]

    evidence = build_evidence_from_results(results)

    assert [item["evidence_id"] for item in evidence] == ["E1", "E2"]
    assert evidence[0]["domain"] == "example.com"


def test_evidence_metrics_count_citations_and_duplicate_urls():
    results = [
        {
            "source": "tavily",
            "results": [
                {"title": "A", "url": "https://example.com/a"},
                {"title": "A copy", "url": "https://example.com/a"},
            ],
        }
    ]
    evidence = build_evidence_from_results(results)

    metrics = calculate_evidence_metrics(results, evidence, "- claim [E1]\n")

    assert metrics["duplicate_url_count"] == 1
    assert metrics["evidence_count"] == 1
    assert metrics["citation_count"] == 1


def test_audit_report_citations_flags_invalid_and_unsupported_claims():
    evidence = [
        {"evidence_id": "E1", "title": "Trace", "snippet": "trace"},
        {"evidence_id": "E2", "title": "Cost", "snippet": "cost"},
    ]
    report = "- grounded claim [E1]\n- cites a missing source [E99]\n- no citation here\n"

    audit = audit_report_citations(report, evidence)

    assert audit["invalid_citation_ids"] == ["E99"]
    assert audit["unused_evidence_ids"] == ["E2"]
    assert audit["valid_citation_count"] == 1
    assert audit["invalid_citation_count"] == 1
    assert audit["unsupported_key_finding_count"] == 2
    assert audit["grounded_key_finding_rate"] == 1 / 3
    assert audit["citation_audit_passed"] is False


def test_evidence_metrics_use_valid_citations_only():
    evidence = [
        {"evidence_id": "E1", "title": "Trace", "snippet": "trace"},
        {"evidence_id": "E2", "title": "Cost", "snippet": "cost"},
    ]

    metrics = calculate_evidence_metrics([], evidence, "- valid [E1]\n- invalid [E99]\n")

    assert metrics["citation_count"] == 2
    assert metrics["valid_citation_count"] == 1
    assert metrics["invalid_citation_count"] == 1
    assert metrics["citation_validity"] == 0.5
    assert metrics["citation_evidence_coverage"] == 0.5
