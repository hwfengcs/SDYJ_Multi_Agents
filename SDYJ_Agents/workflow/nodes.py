"""
Workflow Nodes

This module defines the node functions for the LangGraph workflow.
"""

import time
from typing import Dict, Any
from ..agents.coordinator import Coordinator
from ..agents.planner import Planner
from ..agents.researcher import Researcher
from ..agents.rapporteur import Rapporteur
from ..utils.tracing import (
    record_decision,
    record_degraded_event,
    record_node_event,
    record_trace_event,
)


class WorkflowNodes:
    """
    Container for workflow node functions.
    """

    def __init__(
        self,
        coordinator: Coordinator,
        planner: Planner,
        researcher: Researcher,
        rapporteur: Rapporteur
    ):
        """
        Initialize workflow nodes.

        Args:
            coordinator: Coordinator agent instance
            planner: Planner agent instance
            researcher: Researcher agent instance
            rapporteur: Rapporteur agent instance
        """
        self.coordinator = coordinator
        self.planner = planner
        self.researcher = researcher
        self.rapporteur = rapporteur

    def coordinator_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinator node - entry point of the workflow.

        Args:
            state: Current research state

        Returns:
            Updated state
        """
        started = time.perf_counter()
        record_trace_event(
            state.get('trace'),
            "node_start",
            "coordinator",
            node="coordinator",
            input_snapshot={"query": state.get("query"), "query_type": state.get("query_type")},
        )
        try:
            # Check if this is a simple query that was already handled
            if state.get('query_type') in ['GREETING', 'INAPPROPRIATE']:
                # Simple query already handled in initialize_research
                state['current_step'] = 'completed'
                return state

            # For research queries, delegate to planner
            state['current_step'] = 'coordinating'
            state = self.coordinator.delegate_to_planner(state)
            return state
        finally:
            record_node_event(
                state.get('trace'),
                "coordinator",
                int(round((time.perf_counter() - started) * 1000)),
                metadata={"query_type": state.get("query_type")},
            )

    def planner_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Planner node - creates or updates research plan.

        Args:
            state: Current research state

        Returns:
            Updated state with research plan
        """
        started = time.perf_counter()
        record_trace_event(
            state.get('trace'),
            "node_start",
            "planner",
            node="planner",
            input_snapshot={
                "query": state.get("query"),
                "has_plan": bool(state.get("research_plan")),
                "has_feedback": bool(state.get("user_feedback")),
            },
        )
        try:
            state['current_step'] = 'planning'

            # If there's user feedback and a plan exists, modify it
            if state.get('user_feedback') and state.get('research_plan'):
                state = self.planner.modify_plan(state, state['user_feedback'])
            # Otherwise create a new plan
            elif not state.get('research_plan'):
                state = self.planner.create_research_plan(state)

            return state
        finally:
            record_node_event(
                state.get('trace'),
                "planner",
                int(round((time.perf_counter() - started) * 1000)),
                metadata={
                    "sub_tasks": len((state.get("research_plan") or {}).get("sub_tasks", []))
                },
            )

    def human_review_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Human review node - pauses for user approval.

        Args:
            state: Current research state

        Returns:
            Updated state
        """
        started = time.perf_counter()
        record_trace_event(
            state.get('trace'),
            "node_start",
            "human_review",
            node="human_review",
            input_snapshot={
                "auto_approve": state.get("auto_approve_plan", False),
                "plan_approved": state.get("plan_approved", False),
            },
        )
        try:
            state['current_step'] = 'awaiting_approval'

            # Check if auto-approve is enabled
            if state.get('auto_approve_plan', False):
                state['plan_approved'] = True
                record_decision(
                    state.get('trace'),
                    node="human_review",
                    decision="plan_auto_approved",
                    reason="auto_approve_plan is enabled",
                )

            # In actual implementation, this will pause and wait for user input
            # For now, we just mark the state
            return state
        finally:
            record_node_event(
                state.get('trace'),
                "human_review",
                int(round((time.perf_counter() - started) * 1000)),
                metadata={"approved": state.get("plan_approved", False)},
            )

    def researcher_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Researcher node - executes research tasks.

        Args:
            state: Current research state

        Returns:
            Updated state with research results
        """
        started = time.perf_counter()
        record_trace_event(
            state.get('trace'),
            "node_start",
            "researcher",
            node="researcher",
            input_snapshot={
                "iteration_count": state.get("iteration_count", 0),
                "max_iterations": state.get("max_iterations"),
                "evidence_count": len(state.get("evidence_items") or []),
            },
        )
        try:
            state['current_step'] = 'researching'

            # Get next task from plan
            next_task = self.planner.get_next_task(state)

            if next_task:
                try:
                    state = self.researcher.execute_task(state, next_task)
                except Exception as exc:
                    # A failed task must not kill the run or spin the loop:
                    # mark it failed so get_next_task skips it, keep iterating.
                    for task in (state.get('research_plan') or {}).get('sub_tasks', []):
                        if task.get('task_id') == next_task.get('task_id'):
                            task['status'] = 'failed'
                            break
                    record_degraded_event(
                        state.get('trace'),
                        state,
                        node="researcher",
                        where="researcher_task",
                        error=str(exc),
                    )
                state['current_task'] = next_task
                state['iteration_count'] += 1
            else:
                # No more tasks
                state['needs_more_research'] = False

            return state
        finally:
            record_node_event(
                state.get('trace'),
                "researcher",
                int(round((time.perf_counter() - started) * 1000)),
                metadata={
                    "iteration_count": state.get("iteration_count", 0),
                    "evidence_count": len(state.get("evidence_items") or []),
                },
            )

    def rapporteur_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rapporteur node - generates final report.

        Args:
            state: Current research state

        Returns:
            Updated state with final report
        """
        started = time.perf_counter()
        record_trace_event(
            state.get('trace'),
            "node_start",
            "rapporteur",
            node="rapporteur",
            input_snapshot={
                "research_batches": len(state.get("research_results") or []),
                "evidence_count": len(state.get("evidence_items") or []),
                "output_format": state.get("output_format"),
            },
        )
        try:
            state['current_step'] = 'generating_report'
            state = self.rapporteur.generate_report(state)
            return state
        finally:
            record_node_event(
                state.get('trace'),
                "rapporteur",
                int(round((time.perf_counter() - started) * 1000)),
                metadata=state.get("report_metrics") or {},
            )

    def should_continue_to_planner(self, state: Dict[str, Any]) -> str:
        """
        Conditional edge function - determines if we continue to planner or end.

        Args:
            state: Current research state

        Returns:
            Next node name: "planner" for research queries, "end" for simple queries
        """
        # If this is a simple query (greeting or inappropriate), end workflow
        if state.get('query_type') in ['GREETING', 'INAPPROPRIATE']:
            record_decision(
                state.get('trace'),
                node="coordinator",
                decision="route_end",
                reason=f"query_type={state.get('query_type')}",
            )
            return "end"

        # Otherwise, continue to planner for research
        record_decision(
            state.get('trace'),
            node="coordinator",
            decision="route_planner",
            reason="research query",
        )
        return "planner"

    def should_continue_research(self, state: Dict[str, Any]) -> str:
        """
        Conditional edge function - determines next step after human review.

        Args:
            state: Current research state

        Returns:
            Next node name
        """
        # If plan not approved, go back to planner
        if not state.get('plan_approved'):
            record_decision(
                state.get('trace'),
                node="human_review",
                decision="route_planner",
                reason="plan not approved",
                metadata={"feedback": state.get("user_feedback")},
            )
            return "planner"

        # If plan approved, start research
        record_decision(
            state.get('trace'),
            node="human_review",
            decision="route_researcher",
            reason="plan approved",
        )
        return "researcher"

    def should_generate_report(self, state: Dict[str, Any]) -> str:
        """
        Conditional edge function - determines if we should generate report.

        Args:
            state: Current research state

        Returns:
            Next node name
        """
        # Check if max iterations reached
        if state['iteration_count'] >= state['max_iterations']:
            record_decision(
                state.get('trace'),
                node="researcher",
                decision="route_rapporteur",
                reason="max iterations reached",
                metadata={
                    "iteration_count": state.get("iteration_count"),
                    "max_iterations": state.get("max_iterations"),
                },
            )
            return "rapporteur"

        # Check if context is sufficient. A failing sufficiency check (an LLM
        # call) must never abort the graph from inside a routing decision —
        # degrade to writing the report with whatever evidence exists.
        try:
            sufficient = self.planner.evaluate_context_sufficiency(state)
        except Exception as exc:
            record_degraded_event(
                state.get('trace'),
                state,
                node="researcher",
                where="sufficiency_check",
                error=str(exc),
            )
            record_decision(
                state.get('trace'),
                node="researcher",
                decision="sufficiency_check_failed_default_report",
                reason="sufficiency check failed; defaulting to report generation",
            )
            return "rapporteur"

        if sufficient:
            record_decision(
                state.get('trace'),
                node="researcher",
                decision="route_rapporteur",
                reason="planner judged context sufficient",
                metadata={
                    "iteration_count": state.get("iteration_count"),
                    "evidence_count": len(state.get("evidence_items") or []),
                },
            )
            return "rapporteur"

        # Check if there are more tasks
        next_task = self.planner.get_next_task(state)
        if next_task:
            record_decision(
                state.get('trace'),
                node="researcher",
                decision="route_researcher",
                reason="pending task available",
                metadata={"task_id": next_task.get("task_id")},
            )
            return "researcher"
        else:
            record_decision(
                state.get('trace'),
                node="researcher",
                decision="route_rapporteur",
                reason="no pending tasks",
            )
            return "rapporteur"


def create_node_functions(
    coordinator: Coordinator,
    planner: Planner,
    researcher: Researcher,
    rapporteur: Rapporteur
) -> WorkflowNodes:
    """
    Create workflow node functions.

    Args:
        coordinator: Coordinator agent
        planner: Planner agent
        researcher: Researcher agent
        rapporteur: Rapporteur agent

    Returns:
        WorkflowNodes instance
    """
    return WorkflowNodes(coordinator, planner, researcher, rapporteur)
