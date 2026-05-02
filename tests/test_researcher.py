from SDYJ_Agents.agents.researcher import Researcher


class FakeLLM:
    def generate(self, prompt: str, **kwargs) -> str:
        return "summary"


def test_execute_task_aggregates_results_and_marks_task_completed(monkeypatch):
    researcher = Researcher(FakeLLM())

    def fake_search(query, source):
        return {
            "query": query,
            "source": source,
            "results": [{"title": "Result", "url": "https://example.com", "snippet": "Body"}],
            "timestamp": "2026-01-01T00:00:00",
        }

    monkeypatch.setattr(researcher, "_search", fake_search)
    state = {
        "research_results": [],
        "research_plan": {
            "sub_tasks": [
                {
                    "task_id": 1,
                    "description": "Search docs",
                    "search_queries": ["agent workflow"],
                    "sources": ["tavily"],
                    "status": "pending",
                }
            ]
        },
    }

    updated = researcher.execute_task(state, state["research_plan"]["sub_tasks"][0])

    assert len(updated["research_results"]) == 1
    assert updated["research_results"][0]["task_id"] == 1
    assert updated["research_plan"]["sub_tasks"][0]["status"] == "completed"
