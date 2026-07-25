"""Lockstep contract between prompt templates and FakeEvalLLM canned responses.

Every template embeds a stable ``[PROMPT_ID: <name>]`` marker right after its
front matter, and FakeEvalLLM dispatches on that marker. These tests fail when
either side drifts, which is exactly the failure mode that previously left the
``rapporteur_organize_info`` canned response unreachable (the old dispatch
matched template wording that had changed).
"""

import json

import pytest

from SDYJ_Agents.evaluation.runner import FakeEvalLLM
from SDYJ_Agents.prompts.loader import PromptLoader

TEMPLATES_WITH_CANNED_RESPONSES = [
    "coordinator_classify_query",
    "planner_create_plan",
    "planner_modify_plan",
    "planner_evaluate_context",
    "rapporteur_summarize",
    "rapporteur_organize_info",
    "rapporteur_synthesized_analysis",
    "rapporteur_conclusion",
    "rapporteur_generate_html",
    "judge_faithfulness",
]


@pytest.mark.parametrize("template_name", TEMPLATES_WITH_CANNED_RESPONSES)
def test_template_embeds_its_marker(template_name):
    raw = PromptLoader().load_raw(template_name)
    assert f"[PROMPT_ID: {template_name}]" in raw


@pytest.mark.parametrize("template_name", TEMPLATES_WITH_CANNED_RESPONSES)
def test_fake_eval_llm_has_a_response_for_every_marker(template_name):
    llm = FakeEvalLLM()
    response = llm.generate(f"prefix [PROMPT_ID: {template_name}] suffix")
    # The fallback response is "YES"; every listed template must have its own
    # dedicated canned response (planner_evaluate_context's genuinely is YES).
    if template_name != "planner_evaluate_context":
        assert response != "YES"


def test_classify_marker_returns_research():
    assert FakeEvalLLM().generate("[PROMPT_ID: coordinator_classify_query]") == "RESEARCH"


@pytest.mark.parametrize("template_name", ["planner_create_plan", "planner_modify_plan"])
def test_plan_markers_return_parseable_plan_json(template_name):
    response = FakeEvalLLM().generate(f"[PROMPT_ID: {template_name}]")
    plan = json.loads(response)
    assert plan["sub_tasks"]


def test_organize_marker_returns_parseable_cited_themes():
    response = FakeEvalLLM().generate("[PROMPT_ID: rapporteur_organize_info]")
    themes = json.loads(response)["themes"]
    assert len(themes) == 2
    for theme in themes:
        for point in theme["key_points"]:
            assert "[E" in point


@pytest.mark.parametrize(
    "template_name",
    ["rapporteur_summarize", "rapporteur_synthesized_analysis"],
)
def test_cited_section_markers_include_citations(template_name):
    response = FakeEvalLLM().generate(f"[PROMPT_ID: {template_name}]")
    assert "[E1]" in response


def test_conclusion_marker_is_uncited_by_design():
    response = FakeEvalLLM().generate("[PROMPT_ID: rapporteur_conclusion]")
    assert response
    assert "[E" not in response


def test_html_marker_returns_html_document():
    response = FakeEvalLLM().generate("[PROMPT_ID: rapporteur_generate_html]")
    assert response.startswith("<!DOCTYPE html>")
