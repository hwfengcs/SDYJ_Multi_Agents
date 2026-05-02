from SDYJ_Agents.utils.evidence import (
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
