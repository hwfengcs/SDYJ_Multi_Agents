"""Reusable Streamlit components for the SDYJ Web UI.

Kept in a separate module so ``app.py`` stays focused on flow control. Each
component takes plain dicts (matching the trace and state shapes), not bespoke
classes, so they are easy to test or swap out for a different framework later.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import streamlit as st


def render_metrics_summary(metrics: Dict[str, Any]) -> None:
    """Render the high-level metrics from a finished run as Streamlit metrics."""
    if not metrics:
        st.info("No metrics yet — run a query to see scores, token counts, and cost.")
        return

    col1, col2, col3, col4 = st.columns(4)
    overall = metrics.get("overall_score")
    col1.metric(
        "Overall score",
        f"{overall:.3f}" if isinstance(overall, (int, float)) else "—",
    )
    citation = metrics.get("citation_id_coverage")
    col2.metric(
        "Citation coverage",
        f"{citation:.0%}" if isinstance(citation, (int, float)) else "—",
    )
    tools = metrics.get("tool_success_rate")
    col3.metric(
        "Tool success rate",
        f"{tools:.0%}" if isinstance(tools, (int, float)) else "—",
    )
    cost = metrics.get("total_cost_usd")
    col4.metric(
        "Cost (USD)",
        f"${cost:.4f}" if isinstance(cost, (int, float)) else "—",
    )

    tokens_col1, tokens_col2, tokens_col3 = st.columns(3)
    tokens_col1.metric("Prompt tokens", metrics.get("total_prompt_tokens", "—"))
    tokens_col2.metric("Completion tokens", metrics.get("total_completion_tokens", "—"))
    tokens_col3.metric("Total tokens", metrics.get("total_tokens", "—"))


def render_plan(plan: Optional[Dict[str, Any]]) -> None:
    """Render the structured research plan as a checklist-friendly view."""
    if not plan:
        st.info("Plan not generated yet.")
        return
    st.markdown(f"**Goal:** {plan.get('research_goal', '—')}")
    st.markdown(f"**Completion criteria:** {plan.get('completion_criteria', '—')}")
    st.markdown(f"**Estimated iterations:** {plan.get('estimated_iterations', '—')}")
    sub_tasks = plan.get("sub_tasks") or []
    if not sub_tasks:
        st.warning("No sub-tasks in plan.")
        return
    for task in sorted(sub_tasks, key=lambda t: (t.get("priority", 99), t.get("task_id", 0))):
        with st.expander(
            f"Task {task.get('task_id')} — {task.get('description', '')[:80]} "
            f"({task.get('status', 'pending')})",
            expanded=False,
        ):
            st.markdown(f"**Description:** {task.get('description', '—')}")
            st.markdown(f"**Priority:** {task.get('priority', '—')}")
            queries = task.get("search_queries") or []
            if queries:
                st.markdown("**Search queries:**")
                for q in queries:
                    st.markdown(f"- `{q}`")
            sources = task.get("sources") or []
            if sources:
                st.markdown(f"**Sources:** {', '.join(sources)}")


def render_evidence_cards(evidence_items: List[Dict[str, Any]]) -> None:
    """Render deduplicated evidence items as compact cards."""
    if not evidence_items:
        st.info("No evidence collected yet.")
        return

    st.caption(f"{len(evidence_items)} deduplicated evidence items")
    for item in evidence_items:
        evidence_id = item.get("evidence_id", "E?")
        title = item.get("title") or "Untitled"
        source = (item.get("source") or "unknown").capitalize()
        domain = item.get("domain") or "—"
        published = item.get("published_date") or "—"
        score = item.get("relevance_score")
        score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "—"
        url = item.get("url") or ""
        snippet = (item.get("snippet") or "")[:240]

        with st.container(border=True):
            header = f"**[{evidence_id}] {title}**"
            st.markdown(header)
            meta_parts = [
                f"`{source}`",
                f"domain: `{domain}`",
                f"score: `{score_str}`",
                f"published: `{published}`",
            ]
            st.caption(" · ".join(meta_parts))
            if snippet:
                st.markdown(f"> {snippet}{'…' if len(item.get('snippet') or '') > 240 else ''}")
            if url:
                st.markdown(f"[Open source ↗]({url})")


def render_llm_calls_table(llm_calls: List[Dict[str, Any]]) -> None:
    """Render a table of per-call token + cost numbers."""
    if not llm_calls:
        st.info("No LLM calls recorded yet.")
        return

    rows = []
    for call in llm_calls:
        cost = call.get("cost_usd")
        rows.append(
            {
                "Call": call.get("call_id") or "—",
                "Model": call.get("model") or "—",
                "Prompt tokens": call.get("prompt_tokens_actual") or 0,
                "Completion tokens": call.get("completion_tokens_actual") or 0,
                "Latency (ms)": call.get("latency_ms") or 0,
                "Cost (USD)": f"${cost:.6f}" if isinstance(cost, (int, float)) else "—",
                "Error": call.get("error") or "",
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_tool_calls_table(tool_calls: List[Dict[str, Any]]) -> None:
    """Render a table of retrieval tool calls."""
    if not tool_calls:
        st.info("No tool calls recorded yet.")
        return
    rows = [
        {
            "Source": call.get("source") or "—",
            "Query": call.get("query") or "—",
            "Results": call.get("result_count") or 0,
            "Latency (ms)": call.get("latency_ms") or 0,
            "Error": call.get("error") or "",
        }
        for call in tool_calls
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_timeline(events: Iterable[Dict[str, Any]]) -> None:
    """Render the trace v2 event timeline, lightly formatted.

    Streamlit does not have a native timeline widget, so we fall back to a
    chronologically ordered table. The :func:`render_timeline_chart` helper
    can render the same data as a Plotly Gantt chart when Plotly is available.
    """
    rows = []
    for event in events:
        rows.append(
            {
                "Seq": event.get("seq", "—"),
                "Type": event.get("event_type", "—"),
                "Name": event.get("name", "—"),
                "Node": event.get("node") or "—",
                "Status": event.get("status", "—"),
                "Latency (ms)": event.get("latency_ms") if event.get("latency_ms") is not None else "—",
                "Error": event.get("error") or "",
            }
        )
    if not rows:
        st.info("No timeline events yet.")
        return
    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_run_header(trace: Dict[str, Any]) -> None:
    """Compact header showing run id, provider, and basic counts."""
    cols = st.columns(4)
    cols[0].caption(f"**Run ID**\n`{trace.get('run_id', '—')}`")
    cols[1].caption(f"**Provider**\n`{trace.get('provider', '—')}`")
    cols[2].caption(f"**Model**\n`{trace.get('model', '—')}`")
    cols[3].caption(f"**Mode**\n`{trace.get('mode', '—')}`")
