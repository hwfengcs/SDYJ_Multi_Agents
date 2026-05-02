from SDYJ_Agents.agents.rapporteur import Rapporteur
import json


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


def test_json_report_is_machine_readable():
    rapporteur = Rapporteur(FakeLLM())
    report = rapporteur._generate_json_report(
        query="agent ops",
        plan={"research_goal": "agent ops"},
        summary="summary",
        organized_info={"themes": [{"name": "Reliability", "key_points": ["Trace every claim"]}]},
        results=[
            {
                "source": "tavily",
                "query": "agent ops",
                "results": [{"title": "Trace", "url": "https://example.com/t", "snippet": "trace"}],
            }
        ],
    )

    payload = json.loads(report)
    assert payload["query"] == "agent ops"
    assert payload["sources"][0]["evidence_id"] == "E1"
    assert payload["references"][0]["citation"] == "[E1]"
