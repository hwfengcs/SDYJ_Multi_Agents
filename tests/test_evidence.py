from SDYJ_Agents.utils.evidence import (
    build_evidence_from_results,
    calculate_evidence_metrics,
    extract_body_citations,
    format_evidence_for_prompt,
    normalize_url,
    select_evidence_for_prompt,
    validate_citations,
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


def _evidence(evidence_id, relevance=None, published=None, snippet="s"):
    return {
        "evidence_id": evidence_id,
        "title": f"title {evidence_id}",
        "source": "tavily",
        "url": f"https://example.com/{evidence_id}",
        "snippet": snippet,
        "relevance_score": relevance,
        "published_date": published,
    }


def test_select_evidence_prefers_relevance_over_insertion_order():
    items = [
        _evidence("E1", relevance=0.2),
        _evidence("E2", relevance=0.9),
        _evidence("E3", relevance=0.5),
    ]

    selected = select_evidence_for_prompt(items, limit=2)

    assert [item["evidence_id"] for item in selected] == ["E2", "E3"]


def test_select_evidence_breaks_relevance_ties_by_recency_then_order():
    items = [
        _evidence("E1", relevance=0.5, published="2024-01-01"),
        _evidence("E2", relevance=0.5, published="2026-01-01"),
        _evidence("E3", relevance=0.5, published="2026-01-01"),
        _evidence("E4", relevance=None),
    ]

    selected = select_evidence_for_prompt(items, limit=4)

    assert [item["evidence_id"] for item in selected] == ["E2", "E3", "E1", "E4"]


def test_select_evidence_respects_char_budget_but_keeps_first_item():
    items = [
        _evidence("E1", relevance=0.9, snippet="x" * 300),
        _evidence("E2", relevance=0.8, snippet="y" * 300),
    ]

    selected = select_evidence_for_prompt(items, limit=10, char_budget=50)

    assert [item["evidence_id"] for item in selected] == ["E1"]


def test_select_evidence_is_deterministic_across_calls():
    items = [
        _evidence("E1", relevance=0.4),
        _evidence("E2", relevance=0.9, published="2025-05-01"),
        _evidence("E3", relevance=0.9, published="2024-05-01"),
    ]

    first = [item["evidence_id"] for item in select_evidence_for_prompt(items)]
    second = [item["evidence_id"] for item in select_evidence_for_prompt(items)]

    assert first == second == ["E2", "E3", "E1"]


def test_format_evidence_renders_selected_ids():
    items = [
        _evidence("E1", relevance=0.1),
        _evidence("E2", relevance=0.9),
    ]

    rendered = format_evidence_for_prompt(items, limit=1)

    assert "[E2]" in rendered
    assert "[E1]" not in rendered


def test_validate_citations_strips_fabricated_ids():
    items = [_evidence("E1"), _evidence("E2")]

    cleaned, stats = validate_citations("claim one [E1]. claim two [E99].", items)

    assert cleaned == "claim one [E1]. claim two."
    assert stats["total_citation_mentions"] == 2
    assert stats["valid_citation_mentions"] == 1
    assert stats["invalid_citation_ids"] == ["E99"]
    assert stats["citation_validity_rate"] == 0.5


def test_validate_citations_without_mentions_is_perfect():
    cleaned, stats = validate_citations("no citations here", [_evidence("E1")])

    assert cleaned == "no citations here"
    assert stats["citation_validity_rate"] == 1.0


def test_body_citations_exclude_reference_list_lines():
    report = (
        "## 核心发现\n"
        "- claim [E1]\n"
        "## 参考资料\n"
        "- [E1] Title - Tavily - example.com\n"
        "- [E2] Other - Arxiv - arxiv.org\n"
    )

    assert extract_body_citations(report) == ["[E1]"]


def test_metrics_ignore_reference_list_and_fabricated_ids():
    results = [
        {
            "source": "tavily",
            "results": [
                {"title": "A", "url": "https://example.com/a"},
                {"title": "B", "url": "https://example.com/b"},
            ],
        }
    ]
    evidence = build_evidence_from_results(results)
    report = (
        "## 核心发现\n"
        "- claim [E1]\n"
        "- fabricated claim [E9]\n"
        "## 参考资料\n"
        "- [E1] A - Tavily\n"
        "- [E2] B - Tavily\n"
    )

    metrics = calculate_evidence_metrics(results, evidence, report)

    assert metrics["citation_count"] == 1  # E2 only appears in the reference list
    assert metrics["invalid_citation_count"] == 1  # the [E9] mention
    assert metrics["citation_validity_rate"] == 0.75  # 3 of 4 mentions valid
    assert metrics["grounded_key_finding_rate"] == 0.5  # E9 bullet is not grounded
