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
from ..agents.verifier import Verifier, append_verification_history
from ..utils.tracing import record_decision, record_node_event, record_trace_event


class WorkflowNodes:
    """
    Container for workflow node functions.
    """

    def __init__(
        self,
        coordinator: Coordinator,
        planner: Planner,
        researcher: Researcher,
        rapporteur: Rapporteur,
        verifier: "Verifier | None" = None,
    ):
        """
        Initialize workflow nodes.

        Args:
            coordinator: Coordinator agent instance
            planner: Planner agent instance
            researcher: Researcher agent instance
            rapporteur: Rapporteur agent instance
            verifier: Optional Verifier agent. When ``None``, the workflow
                still has a verifier node but it short-circuits to
                ``accept`` (used by tests / replay where verification is
                irrelevant).
        """
        self.coordinator = coordinator
        self.planner = planner
        self.researcher = researcher
        self.rapporteur = rapporteur
        self.verifier = verifier

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
                # Execute the task
                state = self.researcher.execute_task(state, next_task)
                state['current_task'] = next_task
                state['iteration_count'] += 1
                # After the task is done, give the planner a chance to
                # refine the *remaining* sub-tasks based on the evidence
                # we just collected. Refinement is one-shot per run
                # (guarded by ``state['plan_refined']``) so an unstable
                # planner cannot churn the plan indefinitely.
                self._maybe_refine_plan(state)
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
                    "plan_refined": state.get("plan_refined", False),
                },
            )

    def _maybe_refine_plan(self, state: Dict[str, Any]) -> None:
        """Trigger Planner.refine_plan when we have enough signal."""
        if state.get('plan_refined'):
            return
        if not getattr(self.planner, 'enable_plan_refinement', False):
            return
        plan = state.get('research_plan') or {}
        sub_tasks = plan.get('sub_tasks') or []
        completed = sum(1 for t in sub_tasks if t.get('status') == 'completed')
        remaining = sum(1 for t in sub_tasks if t.get('status') != 'completed')
        if remaining == 0:
            return
        if completed < int(getattr(self.planner, 'refine_after_n_tasks', 2)):
            return

        before_subtask_ids = [t.get('task_id') for t in sub_tasks]
        started = time.perf_counter()
        try:
            self.planner.refine_plan(state)
        finally:
            duration_ms = int(round((time.perf_counter() - started) * 1000))
            after_plan = state.get('research_plan') or {}
            after_subtasks = after_plan.get('sub_tasks') or []
            after_subtask_ids = [t.get('task_id') for t in after_subtasks]
            added = [tid for tid in after_subtask_ids if tid not in before_subtask_ids]
            removed = [tid for tid in before_subtask_ids if tid not in after_subtask_ids]
            record_trace_event(
                state.get('trace'),
                event_type='plan_refinement',
                name='planner_refine',
                node='researcher',
                latency_ms=duration_ms,
                input_snapshot={
                    'completed_subtasks': completed,
                    'remaining_subtasks': remaining,
                },
                output_snapshot={
                    'added_task_ids': added,
                    'removed_task_ids': removed,
                    'rationale': (after_plan.get('history') or [{}])[-1].get('rationale')
                    if after_plan.get('history')
                    else None,
                },
                metadata={'plan_refined': bool(state.get('plan_refined'))},
            )
            trace = state.get('trace')
            if trace is not None:
                metrics = trace.setdefault('metrics', {})
                metrics['plan_refinement_count'] = int(
                    metrics.get('plan_refinement_count', 0)
                ) + (1 if state.get('plan_refined') else 0)
                if added:
                    metrics['plan_refinement_added_tasks'] = len(added)
                if removed:
                    metrics['plan_refinement_removed_tasks'] = len(removed)

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
                "revision_count": state.get("revision_count", 0),
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
                metadata={
                    **(state.get("report_metrics") or {}),
                    "revision_count": state.get("revision_count", 0),
                },
            )

    def verifier_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Verifier node — grade the freshly written report.

        On success, populates ``state['verification_result']`` and appends to
        ``state['verification_history']``. The downstream conditional edge
        ``should_revise_or_end`` reads those fields to decide whether to loop
        back to the rapporteur.

        Skipped when ``state['skip_verification']`` is true or the workflow
        was constructed without a ``Verifier``; in both cases the node is a
        no-op so the graph still advances to END.

        IMPORTANT: ``revision_count`` is incremented *here* (not in the
        conditional edge below). LangGraph persists state mutations made
        inside node functions but discards mutations made inside conditional
        edge functions, so doing the bump here is what prevents an infinite
        critique-revise loop.
        """
        started = time.perf_counter()
        skip = bool(state.get("skip_verification")) or self.verifier is None
        record_trace_event(
            state.get('trace'),
            "node_start",
            "verifier",
            node="verifier",
            input_snapshot={
                "skip_verification": skip,
                "revision_count": state.get("revision_count", 0),
                "max_revisions": state.get("max_revisions", 0),
            },
        )
        try:
            if skip:
                state['current_step'] = 'verification_skipped'
                return state

            state['current_step'] = 'verifying'
            verification_result = self.verifier.verify(state)
            state['verification_result'] = verification_result
            append_verification_history(state, verification_result)

            # Mirror the critique into the trace so it is visible in inspect-run
            # and survives deterministic replay.
            record_trace_event(
                state.get('trace'),
                event_type="verification",
                name=f"critique#{state.get('revision_count', 0) + 1}",
                node="verifier",
                status="ok" if not verification_result.get("verifier_failed") else "error",
                output_snapshot={
                    "scores": verification_result.get("scores"),
                    "overall_quality": verification_result.get("overall_quality"),
                    "should_revise": verification_result.get("should_revise"),
                    "weakest_dimension": verification_result.get("weakest_dimension"),
                },
                metadata={
                    "revision_count": state.get("revision_count", 0),
                    "summary": verification_result.get("summary"),
                },
            )

            # Decide whether the next edge will route to a revise. We bump
            # revision_count here so the bump is captured in the persisted
            # state — see the docstring for why the conditional edge cannot
            # do it itself.
            should_revise = bool(verification_result.get("should_revise"))
            current_count = int(state.get("revision_count", 0) or 0)
            max_revisions = int(state.get("max_revisions", 0) or 0)
            will_route_to_revise = should_revise and current_count < max_revisions
            if will_route_to_revise:
                state["revision_count"] = current_count + 1
            state["_pending_route_to_revise"] = will_route_to_revise

            # Surface the latest critique into trace metrics so it lands in
            # downstream summaries (diff-runs, eval reports).
            trace = state.get("trace")
            if trace is not None:
                citation_audit = verification_result.get("citation_audit") or {}
                trace.setdefault("metrics", {}).update(
                    {
                        "verifier_overall_quality": verification_result.get("overall_quality"),
                        "verifier_should_revise": verification_result.get("should_revise"),
                        "verifier_weakest_dimension": verification_result.get("weakest_dimension"),
                        "verifier_revision_count": state["revision_count"],
                        "verifier_invalid_citation_count": citation_audit.get("invalid_citation_count"),
                        "verifier_unsupported_key_finding_count": citation_audit.get(
                            "unsupported_key_finding_count"
                        ),
                        "verifier_citation_validity": citation_audit.get("citation_validity"),
                    }
                )

            return state
        finally:
            record_node_event(
                state.get('trace'),
                "verifier",
                int(round((time.perf_counter() - started) * 1000)),
                metadata={
                    "skipped": skip,
                    "revision_count": state.get("revision_count", 0),
                    "should_revise": (state.get("verification_result") or {}).get(
                        "should_revise"
                    ),
                },
            )

    def should_revise_or_end(self, state: Dict[str, Any]) -> str:
        """Decide whether to loop back to the rapporteur for one more pass.

        This function only *reads* state — see ``verifier_node`` for why the
        revision counter is bumped there instead of here.

        Routing rules (in order):
        1. If verification was skipped or never ran -> end.
        2. If the verifier said "do not revise" -> end.
        3. If we have already used up ``max_revisions`` -> end (with a
           routing-level note in the trace so the operator can see the cap
           kicked in).
        4. Otherwise route back to the rapporteur. The verifier hints are
           already on state for the rapporteur to read.
        """
        verification = state.get("verification_result") or {}
        max_revisions = state.get("max_revisions") or 0

        if state.get("skip_verification") or not verification:
            record_decision(
                state.get('trace'),
                node="verifier",
                decision="route_end",
                reason="verification skipped or unavailable",
            )
            return "end"

        if not verification.get("should_revise"):
            record_decision(
                state.get('trace'),
                node="verifier",
                decision="route_end",
                reason="verifier accepted the report",
                metadata={"overall_quality": verification.get("overall_quality")},
            )
            return "end"

        if not state.get("_pending_route_to_revise"):
            # The verifier_node already concluded we hit the cap; honor it.
            record_decision(
                state.get('trace'),
                node="verifier",
                decision="route_end",
                reason="max_revisions reached, accepting current report",
                metadata={
                    "revision_count": state.get("revision_count", 0),
                    "max_revisions": max_revisions,
                    "overall_quality": verification.get("overall_quality"),
                },
            )
            return "end"

        record_decision(
            state.get('trace'),
            node="verifier",
            decision="route_rapporteur_revise",
            reason=f"revision {state.get('revision_count', 0)}/{max_revisions}",
            metadata={
                "weakest_dimension": verification.get("weakest_dimension"),
                "overall_quality": verification.get("overall_quality"),
            },
        )
        return "rapporteur"

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

        # Check if context is sufficient
        if self.planner.evaluate_context_sufficiency(state):
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
    rapporteur: Rapporteur,
    verifier: "Verifier | None" = None,
) -> WorkflowNodes:
    """
    Create workflow node functions.

    Args:
        coordinator: Coordinator agent
        planner: Planner agent
        researcher: Researcher agent
        rapporteur: Rapporteur agent
        verifier: Optional Verifier agent (v0.6+). When ``None`` the verifier
            node is a no-op and the workflow effectively skips verification.

    Returns:
        WorkflowNodes instance
    """
    return WorkflowNodes(coordinator, planner, researcher, rapporteur, verifier)
