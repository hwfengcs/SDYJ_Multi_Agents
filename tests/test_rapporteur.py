from SDYJ_Agents.agents.rapporteur import Rapporteur


class FakeLLM:
    def generate(self, prompt: str, **kwargs) -> str:
        return "generated"


def test_source_overview_counts_results_and_errors():
    rapporteur = Rapporteur(FakeLLM())
    results = [
        {
            "source": "tavily",
            "results": [{"title": "A"}, {"title": "B"}],
        },
        {
            "source": "arxiv",
            "results": [],
            "error": "temporary failure",
        },
    ]

    overview = rapporteur._format_source_overview(results)

    assert "| tavily | 1 | 2 | 0 |" in overview
    assert "| arxiv | 1 | 0 | 1 |" in overview


def test_citations_are_deduplicated_by_url():
    rapporteur = Rapporteur(FakeLLM())
    results = [
        {
            "source": "tavily",
            "results": [
                {"title": "Same", "url": "https://example.com"},
                {"title": "Same again", "url": "https://example.com"},
            ],
        }
    ]

    citations = rapporteur._format_citations(results)

    assert citations.count("https://example.com") == 2
    assert "2." not in citations
