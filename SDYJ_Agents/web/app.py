"""Streamlit Web UI for SDYJ Multi Agents.

Run with::

    streamlit run SDYJ_Agents/web/app.py

The UI deliberately stays thin — it reuses the same agents, workflow,
``InstrumentedLLM``, and trace pipeline that the CLI uses. Anything we add
here therefore costs us nothing in the CLI path.

Design choices:

* For the MVP we use ``auto_approve=True`` so the run completes in one click.
  Genuine human-in-the-loop approval is tracked as a follow-up in v0.6 because
  it requires LangGraph state to survive a Streamlit rerender, which the
  in-memory checkpointer does not do out of the box.
* Trace, evidence, plan, report, and cost figures are read directly from the
  final state and the ``trace`` dict, exactly the same shapes used by the CLI.
  Components do not own any state — they just render dicts.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Allow running ``streamlit run SDYJ_Agents/web/app.py`` from a checkout
# without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st  # noqa: E402  (must follow sys.path tweak above)
from dotenv import load_dotenv  # noqa: E402

from SDYJ_Agents.agents.coordinator import Coordinator  # noqa: E402
from SDYJ_Agents.agents.planner import Planner  # noqa: E402
from SDYJ_Agents.agents.rapporteur import Rapporteur  # noqa: E402
from SDYJ_Agents.agents.researcher import Researcher  # noqa: E402
from SDYJ_Agents.agents.verifier import DEFAULT_MAX_REVISIONS, Verifier  # noqa: E402
from SDYJ_Agents.llm.factory import LLMFactory  # noqa: E402
from SDYJ_Agents.utils.config import load_config_from_env  # noqa: E402
from SDYJ_Agents.utils.tracing import (  # noqa: E402
    InstrumentedLLM,
    create_run_trace,
    iter_timeline_events,
    merge_trace_state,
    save_trace,
)
from SDYJ_Agents.web.components import (  # noqa: E402
    render_evidence_cards,
    render_llm_calls_table,
    render_metrics_summary,
    render_plan,
    render_run_header,
    render_timeline,
    render_tool_calls_table,
)
from SDYJ_Agents.workflow.graph import ResearchWorkflow  # noqa: E402

PROVIDER_DEFAULT_MODELS = {
    "deepseek": "deepseek-v4-flash",
    "openai": "gpt-4o-mini",
    "claude": "claude-3-5-sonnet-20241022",
    "gemini": "gemini-1.5-pro",
}

PROVIDER_MODELS = {
    "deepseek": ["deepseek-v4-flash", "deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner"],
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"],
    "claude": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
    "gemini": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"],
}

PROVIDER_API_KEY_ENVS = {
    "deepseek": ("DEEPSEEK_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "claude": ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"),
    "gemini": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
}


def _missing_api_key_for(provider: str) -> Optional[str]:
    """Return the env var name to set, or None if a usable key is present."""
    env_names = PROVIDER_API_KEY_ENVS.get(provider, ())
    for name in env_names:
        if os.environ.get(name):
            return None
    return env_names[0] if env_names else None


def _build_workflow(
    provider: str,
    model: str,
    skip_verification: bool,
    enable_reflection: bool,
    enable_plan_refinement: bool,
    enable_parallel_tool_execution: bool,
) -> tuple[ResearchWorkflow, Dict[str, Any]]:
    """Build a fresh workflow + trace for one run.

    A new workflow per run keeps the LangGraph checkpointer simple and avoids
    bleed-through across queries inside a single Streamlit session.
    """
    os.environ["LLM_PROVIDER"] = provider
    env_cfg = load_config_from_env()
    env_cfg.llm.model = model

    trace = create_run_trace(
        query="",  # filled in by caller
        provider=provider,
        model=model,
        mode="web",
    )

    base_llm = LLMFactory.create_llm(
        provider=env_cfg.llm.provider,
        api_key=env_cfg.llm.api_key,
        model=env_cfg.llm.model,
    )
    llm = InstrumentedLLM(base_llm, trace)
    coordinator = Coordinator(llm)
    planner = Planner(llm, enable_plan_refinement=enable_plan_refinement)
    researcher = Researcher(
        llm=llm,
        tavily_api_key=env_cfg.search.tavily_api_key,
        mcp_server_url=env_cfg.search.mcp_server_url,
        mcp_api_key=env_cfg.search.mcp_api_key,
        mcp_transport=env_cfg.search.mcp_transport,
        mcp_tool_name=env_cfg.search.mcp_tool_name,
        mcp_config_path=env_cfg.search.mcp_config_path,
        mcp_server_name=env_cfg.search.mcp_server_name,
        mcp_command=env_cfg.search.mcp_command,
        mcp_args=env_cfg.search.mcp_args,
        mcp_env=env_cfg.search.mcp_env,
        enable_reflection=enable_reflection,
        enable_parallel_tool_execution=enable_parallel_tool_execution,
    )
    rapporteur = Rapporteur(llm)
    verifier = None if skip_verification else Verifier(llm)
    workflow = ResearchWorkflow(coordinator, planner, researcher, rapporteur, verifier)
    return workflow, trace


def _run_research(
    query: str,
    provider: str,
    model: str,
    max_iterations: int,
    output_format: str,
    output_dir: str,
    skip_verification: bool,
    max_revisions: int,
    enable_reflection: bool,
    enable_plan_refinement: bool,
    enable_parallel_tool_execution: bool,
) -> Dict[str, Any]:
    """Run the full research workflow and persist the trace bundle.

    Streamlit will block while this executes; the work is bounded by
    ``max_iterations`` and the LLM calls within. Streaming would be a nicer
    experience but is left as a follow-up — the trace timeline below
    reconstructs the run order, so users can see exactly what happened.
    """
    workflow, trace = _build_workflow(
        provider,
        model,
        skip_verification,
        enable_reflection,
        enable_plan_refinement,
        enable_parallel_tool_execution,
    )
    trace["query"] = query
    trace.setdefault("config", {}).update(
        {
            "enable_reflection": enable_reflection,
            "skip_verification": skip_verification,
            "max_revisions": max_revisions,
            "enable_plan_refinement": enable_plan_refinement,
            "enable_parallel_tool_execution": enable_parallel_tool_execution,
        }
    )

    final_state: Dict[str, Any] = {}
    # ``stream_interactive`` with ``auto_approve=True`` exercises the same
    # human-review interrupt path the CLI uses, just without prompting.
    for update in workflow.stream_interactive(
        query=query,
        max_iterations=max_iterations,
        auto_approve=True,
        output_format=output_format,
        trace=trace,
        skip_verification=skip_verification,
        max_revisions=max_revisions,
    ):
        for value in update.values():
            if isinstance(value, dict):
                final_state = value

    # Reconcile state-side trace mutations with our base trace (LangGraph
    # passes copies through nodes).
    merged_trace = merge_trace_state(trace, final_state.get("trace")) or trace
    save_trace(merged_trace, output_dir, final_state=final_state, report=final_state.get("final_report"))
    return {"final_state": final_state, "trace": merged_trace}


def _set_query_to_example(example: str) -> None:
    """Streamlit callback used by example chips to seed the query box."""
    st.session_state["query"] = example


def main() -> None:
    load_dotenv()

    st.set_page_config(
        page_title="SDYJ Multi Agents",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ---- Sidebar ---------------------------------------------------------
    with st.sidebar:
        st.title("SDYJ Multi Agents")
        st.caption(
            "Self-verifying multi-agent research framework with replayable traces."
        )

        provider = st.selectbox(
            "LLM provider",
            list(PROVIDER_MODELS.keys()),
            index=0,
            help="DeepSeek is the cheapest provider and works well for benchmarks.",
        )
        model = st.selectbox(
            "Model",
            PROVIDER_MODELS[provider],
            index=0,
        )
        max_iterations = st.slider(
            "Max research iterations",
            min_value=1,
            max_value=8,
            value=3,
            help="Upper bound on Researcher loops before the report is generated.",
        )
        output_format = st.selectbox(
            "Report format",
            ["markdown", "html", "json"],
            index=0,
        )
        output_dir = st.text_input(
            "Output directory",
            value="./outputs",
            help="Where the run bundle (trace, events, report) is saved.",
        )

        st.divider()
        enable_verifier = st.toggle(
            "Self-verifying loop (v0.6)",
            value=True,
            help=(
                "When on, the Verifier agent grades the report against the "
                "evidence and may trigger up to N revisions when grounding is "
                "weak. Off = v0.5 behavior (single-pass, no critique)."
            ),
        )
        max_revisions = st.slider(
            "Max revisions",
            min_value=0,
            max_value=4,
            value=DEFAULT_MAX_REVISIONS,
            help="Hard cap on Rapporteur revisions when the verifier is on.",
            disabled=not enable_verifier,
        )
        enable_reflection = st.toggle(
            "Reflexive Researcher (v0.6)",
            value=True,
            help=(
                "When on, the Researcher rewrites and retries queries that "
                "returned empty / low-relevance / failing batches. Off = "
                "single-pass retrieval (v0.5 behavior)."
            ),
        )
        enable_plan_refinement = st.toggle(
            "Plan refinement (v0.6)",
            value=True,
            help=(
                "When on, the Planner revisits the remaining sub-tasks "
                "after the first 2 complete and may delete redundant "
                "ones, tighten queries, or add a follow-up sub-task that "
                "the evidence revealed. One-shot per run."
            ),
        )
        enable_parallel_tool_execution = st.toggle(
            "Parallel tool execution (v0.6)",
            value=True,
            help=(
                "When on, each task runs its query/source lookups concurrently "
                "with a bounded concurrency limit. Off = sequential retrieval "
                "(v0.5 behavior)."
            ),
        )

        st.divider()
        missing_key = _missing_api_key_for(provider)
        if missing_key:
            st.error(f"Missing **{missing_key}** — set it in `.env` or your shell.")
        else:
            st.success(f"`{provider}` API key detected.")
        if not os.environ.get("TAVILY_API_KEY"):
            st.warning("`TAVILY_API_KEY` not set — web search will be skipped.")

        st.divider()
        st.markdown(
            "**Repo:** [hwfengcs/SDYJ_Multi_Agents]"
            "(https://github.com/hwfengcs/SDYJ_Multi_Agents)\n\n"
            "**Release notes:** [v0.6](../docs/release-notes/v0.6.md)"
        )

    # ---- Main area -------------------------------------------------------
    st.title("🔎 Deep research with traceable, replayable multi-agent workflow")
    st.markdown(
        "Enter an open-ended research question. SDYJ will plan, retrieve from "
        "Tavily / arXiv, ground every claim in `E1/E2/...` evidence, and emit "
        "a Trace v2 bundle so the run can be inspected, replayed, or "
        "regression-tested later."
    )

    examples = [
        "How should RAG agents be evaluated for reliability?",
        "Compare the design trade-offs of LangGraph, AutoGen, and CrewAI.",
        "Summarize recent advances in agent self-reflection (2025-2026).",
    ]
    chip_cols = st.columns(len(examples))
    for col, example in zip(chip_cols, examples, strict=True):
        col.button(
            example[:60] + ("…" if len(example) > 60 else ""),
            key=f"example_{hash(example)}",
            on_click=_set_query_to_example,
            args=(example,),
            use_container_width=True,
        )

    query = st.text_area(
        "Your research question",
        key="query",
        placeholder="What would you like SDYJ to research?",
        height=100,
    )

    col_run, col_clear = st.columns([1, 5])
    run_clicked = col_run.button("▶ Run research", type="primary", use_container_width=True)
    if col_clear.button("Clear last result", use_container_width=False):
        st.session_state.pop("last_result", None)

    if run_clicked:
        if not query.strip():
            st.error("Please enter a research question first.")
        elif missing_key:
            st.error(
                f"Cannot run: missing `{missing_key}`. The Web UI uses the same "
                "configuration as the CLI — set the env var and reload the page."
            )
        else:
            with st.status("Running multi-agent workflow…", expanded=True) as status:
                status.write("Initializing LLM and tools…")
                try:
                    result = _run_research(
                        query=query.strip(),
                        provider=provider,
                        model=model,
                        max_iterations=max_iterations,
                        output_format=output_format,
                        output_dir=output_dir,
                        skip_verification=not enable_verifier,
                        max_revisions=max_revisions,
                        enable_reflection=enable_reflection,
                        enable_plan_refinement=enable_plan_refinement,
                        enable_parallel_tool_execution=enable_parallel_tool_execution,
                    )
                    st.session_state["last_result"] = result
                    status.update(
                        label="✅ Research complete — see the tabs below.",
                        state="complete",
                    )
                except Exception as exc:
                    st.session_state["last_result"] = None
                    status.update(label=f"❌ Run failed: {exc}", state="error")
                    raise

    result = st.session_state.get("last_result")
    if not result:
        st.info("Submit a question above to start a run. Sample artifacts ship in `examples/`.")
        return

    final_state: Dict[str, Any] = result.get("final_state") or {}
    trace: Dict[str, Any] = result.get("trace") or {}

    render_run_header(trace)
    render_metrics_summary(trace.get("metrics") or {})

    tab_report, tab_plan, tab_evidence, tab_trace, tab_llm = st.tabs(
        ["📄 Report", "🗺️ Plan", "📚 Evidence", "🕒 Trace", "💸 LLM cost"]
    )

    with tab_report:
        report = final_state.get("final_report") or ""
        if not report:
            st.info("No report generated.")
        elif output_format == "html":
            st.components.v1.html(report, height=900, scrolling=True)
        elif output_format == "json":
            st.json(report)
        else:
            st.markdown(report)

    with tab_plan:
        render_plan(final_state.get("research_plan"))

    with tab_evidence:
        render_evidence_cards(final_state.get("evidence_items") or [])

    with tab_trace:
        st.subheader("Tool calls")
        render_tool_calls_table(trace.get("tool_calls") or [])
        st.subheader("Event timeline")
        render_timeline(iter_timeline_events(trace))

    with tab_llm:
        render_llm_calls_table(trace.get("llm_calls") or [])


if __name__ == "__main__":
    main()
