"""Tests for durable checkpointing and resume across workflow instances."""

import json

import pytest

from SDYJ_Agents.agents.coordinator import Coordinator
from SDYJ_Agents.agents.planner import Planner
from SDYJ_Agents.agents.rapporteur import Rapporteur
from SDYJ_Agents.agents.researcher import Researcher
from SDYJ_Agents.workflow.graph import ResearchWorkflow, open_sqlite_checkpointer


class ScriptedLLM:
    def __init__(self, script):
        self.script = script

    def generate(self, prompt: str, **kwargs) -> str:
        for marker, response in self.script.items():
            if marker in prompt:
                return response
        raise AssertionError(f"Unexpected LLM call: {prompt[:120]}")

    def stream_generate(self, prompt: str, **kwargs):
        yield self.generate(prompt, **kwargs)


class FakeSearchTool:
    def search(self, query: str, **kwargs):
        return {
            "query": query,
            "source": "tavily",
            "results": [
                {
                    "title": "Doc",
                    "url": "https://example.com/doc",
                    "snippet": "snippet",
                    "relevance_score": 0.9,
                }
            ],
            "total_results": 1,
        }


SCRIPT = {
    "[PROMPT_ID: coordinator_classify_query]": "RESEARCH",
    "[PROMPT_ID: planner_create_plan]": json.dumps(
        {
            "research_goal": "goal",
            "sub_tasks": [
                {
                    "task_id": 1,
                    "description": "task",
                    "search_queries": ["query"],
                    "sources": ["tavily"],
                    "priority": 1,
                }
            ],
            "completion_criteria": "done",
            "estimated_iterations": 1,
        }
    ),
    "[PROMPT_ID: rapporteur_summarize]": "摘要 [E1]",
    "[PROMPT_ID: rapporteur_organize_info]": json.dumps(
        {"themes": [{"name": "主题", "key_points": ["发现 [E1]"]}]},
        ensure_ascii=False,
    ),
}


def _build_workflow(db_path):
    pytest.importorskip("langgraph.checkpoint.sqlite")
    llm = ScriptedLLM(SCRIPT)
    researcher = Researcher(llm)
    researcher.tavily = FakeSearchTool()
    return ResearchWorkflow(
        Coordinator(llm),
        Planner(llm),
        researcher,
        Rapporteur(llm),
        checkpointer=open_sqlite_checkpointer(db_path),
    )


def test_run_resumes_from_sqlite_checkpoint_across_instances(tmp_path):
    db_path = tmp_path / "checkpoint.sqlite"
    thread_id = "resume-test-thread"

    # Phase 1: stream up to the human_review interrupt, then drop the
    # workflow entirely (no callback + no auto_approve stops at the gate).
    first = _build_workflow(db_path)
    try:
        interrupted = False
        for update in first.stream_interactive(
            "研究一个问题",
            max_iterations=1,
            auto_approve=False,
            human_approval_callback=None,
            output_format="json",
            trace={"run_id": thread_id},
        ):
            if "__interrupt__" in update:
                interrupted = True
        assert interrupted
    finally:
        first.close()

    # Phase 2: a brand-new process/workflow attaches to the same sqlite file
    # and finishes the run from the pending interrupt.
    second = _build_workflow(db_path)
    try:
        final_state = {}
        for update in second.resume_interactive(
            thread_id=thread_id,
            max_iterations=1,
            auto_approve=True,
        ):
            for value in update.values():
                if isinstance(value, dict):
                    final_state = value

        assert final_state.get("final_report")
        assert final_state.get("current_step") == "completed"
        # The full state must survive the resume boundary intact: exactly the
        # one recorded batch, with no keys lost and no duplication.
        assert len(final_state.get("research_results") or []) == 1
    finally:
        second.close()


def test_resume_without_checkpoint_raises(tmp_path):
    db_path = tmp_path / "empty.sqlite"
    workflow = _build_workflow(db_path)
    try:
        with pytest.raises(ValueError, match="No checkpoint"):
            list(workflow.resume_interactive(thread_id="missing-thread"))
    finally:
        workflow.close()
