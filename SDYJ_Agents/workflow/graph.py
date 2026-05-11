"""
Research Workflow Graph

This module creates and manages the LangGraph workflow for the research system.
"""

from typing import Optional
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from .nodes import WorkflowNodes
from ..agents.coordinator import Coordinator
from ..agents.planner import Planner
from ..agents.researcher import Researcher
from ..agents.rapporteur import Rapporteur
from ..agents.verifier import DEFAULT_MAX_REVISIONS, Verifier
from ..utils.tracing import record_decision


def create_research_graph(
    coordinator: Coordinator,
    planner: Planner,
    researcher: Researcher,
    rapporteur: Rapporteur,
    verifier: Optional[Verifier] = None,
):
    """
    Create the research workflow graph.

    Args:
        coordinator: Coordinator agent instance
        planner: Planner agent instance
        researcher: Researcher agent instance
        rapporteur: Rapporteur agent instance
        verifier: Optional Verifier agent (v0.6+). When provided the graph
            grows a critique-revise loop after the rapporteur. When ``None``
            the verifier node is wired in but short-circuits to END,
            preserving v0.5 behavior.

    Returns:
        Compiled LangGraph workflow
    """
    # Create workflow nodes
    nodes = WorkflowNodes(coordinator, planner, researcher, rapporteur, verifier)

    # Initialize state graph
    workflow = StateGraph(dict)  # Use dict instead of TypedDict for compatibility

    # Add nodes to the graph
    workflow.add_node("coordinator", nodes.coordinator_node)
    workflow.add_node("planner", nodes.planner_node)
    workflow.add_node("human_review", nodes.human_review_node)
    workflow.add_node("researcher", nodes.researcher_node)
    workflow.add_node("rapporteur", nodes.rapporteur_node)
    workflow.add_node("verifier", nodes.verifier_node)

    # Add edges from START instead of using set_entry_point
    workflow.add_edge(START, "coordinator")

    # Coordinator -> conditional edge (simple query ends, research continues)
    workflow.add_conditional_edges(
        "coordinator",
        nodes.should_continue_to_planner,
        {
            "planner": "planner",  # Research query
            "end": END             # Simple query (greeting/inappropriate)
        }
    )

    # Planner -> Human Review
    workflow.add_edge("planner", "human_review")

    # Human Review -> conditional edge
    workflow.add_conditional_edges(
        "human_review",
        nodes.should_continue_research,
        {
            "planner": "planner",      # User wants modifications
            "researcher": "researcher"  # User approved, start research
        }
    )

    # Researcher -> conditional edge
    workflow.add_conditional_edges(
        "researcher",
        nodes.should_generate_report,
        {
            "researcher": "researcher",  # Continue research
            "rapporteur": "rapporteur"   # Generate report
        }
    )

    # Rapporteur -> Verifier (always; verifier may be a no-op)
    workflow.add_edge("rapporteur", "verifier")

    # Verifier -> conditional edge: either accept and end, or loop back to
    # the rapporteur for one more pass with the critique hints applied.
    workflow.add_conditional_edges(
        "verifier",
        nodes.should_revise_or_end,
        {
            "end": END,
            "rapporteur": "rapporteur",
        }
    )

    # Compile the graph with checkpointer
    # Add interrupt before human_review for human-in-the-loop
    checkpointer = MemorySaver()
    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"]
    )


class ResearchWorkflow:
    """
    Research workflow manager.

    This class provides a high-level interface for running the research workflow.
    """

    def __init__(
        self,
        coordinator: Coordinator,
        planner: Planner,
        researcher: Researcher,
        rapporteur: Rapporteur,
        verifier: Optional[Verifier] = None,
    ):
        """
        Initialize the research workflow.

        Args:
            coordinator: Coordinator agent
            planner: Planner agent
            researcher: Researcher agent
            rapporteur: Rapporteur agent
            verifier: Optional Verifier agent for the v0.6 critique-revise
                loop. ``None`` keeps the v0.5 behavior (no verification).
        """
        self.coordinator = coordinator
        self.planner = planner
        self.researcher = researcher
        self.rapporteur = rapporteur
        self.verifier = verifier
        self.graph = create_research_graph(
            coordinator, planner, researcher, rapporteur, verifier
        )

    def _seed_verifier_state(
        self,
        initial_state: dict,
        skip_verification: Optional[bool],
        max_revisions: Optional[int],
    ) -> dict:
        """Populate v0.6 verifier fields on the initial state.

        Centralized so ``run`` / ``stream`` / ``stream_interactive`` all stamp
        the same defaults. ``skip_verification`` is True when no verifier was
        wired in, so the workflow node short-circuits cleanly.
        """
        initial_state.setdefault("revision_count", 0)
        initial_state.setdefault("verification_history", [])
        initial_state.setdefault("verification_result", None)
        initial_state.setdefault("plan_refined", False)
        initial_state["max_revisions"] = (
            max_revisions if max_revisions is not None else DEFAULT_MAX_REVISIONS
        )
        if skip_verification is None:
            skip_verification = self.verifier is None
        initial_state["skip_verification"] = bool(skip_verification)
        return initial_state

    def run(
        self,
        query: str,
        max_iterations: Optional[int] = None,
        auto_approve: bool = False,
        output_format: str = "markdown",
        trace: Optional[dict] = None,
        skip_verification: Optional[bool] = None,
        max_revisions: Optional[int] = None,
    ) -> dict:
        """
        Run the research workflow.

        Args:
            query: Research query
            max_iterations: Maximum number of research iterations
            auto_approve: Whether to auto-approve the research plan
            output_format: Output format for the final report ("markdown" or "html")
            skip_verification: Bypass the verifier loop. Defaults to True
                when this workflow has no Verifier wired in.
            max_revisions: Hard cap on rapporteur revisions. Defaults to
                ``DEFAULT_MAX_REVISIONS`` from ``agents/verifier.py``.

        Returns:
            Final research state
        """
        # Initialize state
        initial_state = self.coordinator.initialize_research(query, auto_approve=auto_approve, output_format=output_format)
        if trace:
            trace.setdefault("config", {}).update(
                {
                    "max_iterations": max_iterations,
                    "auto_approve": auto_approve,
                    "output_format": output_format,
                    "skip_verification": skip_verification if skip_verification is not None else (self.verifier is None),
                    "max_revisions": max_revisions if max_revisions is not None else DEFAULT_MAX_REVISIONS,
                    "enable_parallel_tool_execution": getattr(self.researcher, "enable_parallel_tool_execution", True),
                }
            )
            initial_state['trace'] = trace

        if max_iterations:
            initial_state['max_iterations'] = max_iterations

        initial_state = self._seed_verifier_state(initial_state, skip_verification, max_revisions)

        # Run the graph with thread configuration for checkpointer
        thread_id = trace.get("run_id", "1") if trace else "1"
        config = {"configurable": {"thread_id": thread_id}}
        final_state = self.graph.invoke(initial_state, config=config)

        return final_state

    def stream(
        self,
        query: str,
        max_iterations: Optional[int] = None,
        auto_approve: bool = False,
        output_format: str = "markdown",
        trace: Optional[dict] = None,
        skip_verification: Optional[bool] = None,
        max_revisions: Optional[int] = None,
    ):
        """
        Stream the research workflow execution.

        Args:
            query: Research query
            max_iterations: Maximum number of research iterations
            auto_approve: Whether to auto-approve the research plan
            output_format: Output format for the final report ("markdown" or "html")

        Yields:
            State updates during execution
        """
        # Initialize state
        initial_state = self.coordinator.initialize_research(query, auto_approve=auto_approve, output_format=output_format)
        if trace:
            trace.setdefault("config", {}).update(
                {
                    "max_iterations": max_iterations,
                    "auto_approve": auto_approve,
                    "output_format": output_format,
                    "skip_verification": skip_verification if skip_verification is not None else (self.verifier is None),
                    "max_revisions": max_revisions if max_revisions is not None else DEFAULT_MAX_REVISIONS,
                    "enable_parallel_tool_execution": getattr(self.researcher, "enable_parallel_tool_execution", True),
                }
            )
            initial_state['trace'] = trace

        if max_iterations:
            initial_state['max_iterations'] = max_iterations

        initial_state = self._seed_verifier_state(initial_state, skip_verification, max_revisions)

        # Stream the graph execution with thread configuration for checkpointer
        thread_id = trace.get("run_id", "1") if trace else "1"
        config = {"configurable": {"thread_id": thread_id}}
        for output in self.graph.stream(initial_state, config=config):
            yield output

    def stream_interactive(
        self,
        query: str,
        max_iterations: Optional[int] = None,
        auto_approve: bool = False,
        human_approval_callback = None,
        output_format: str = "markdown",
        trace: Optional[dict] = None,
        skip_verification: Optional[bool] = None,
        max_revisions: Optional[int] = None,
    ):
        """
        Stream the research workflow execution with interactive human approval.

        Args:
            query: Research query
            max_iterations: Maximum number of research iterations
            auto_approve: Whether to auto-approve the research plan
            human_approval_callback: Callback function for human approval
                                   Should return (approved: bool, feedback: str)
            output_format: Output format for the final report ("markdown" or "html")

        Yields:
            State updates during execution
        """
        # Initialize state
        initial_state = self.coordinator.initialize_research(query, auto_approve=auto_approve, output_format=output_format)
        if trace:
            trace.setdefault("config", {}).update(
                {
                    "max_iterations": max_iterations,
                    "auto_approve": auto_approve,
                    "output_format": output_format,
                    "skip_verification": skip_verification if skip_verification is not None else (self.verifier is None),
                    "max_revisions": max_revisions if max_revisions is not None else DEFAULT_MAX_REVISIONS,
                    "enable_parallel_tool_execution": getattr(self.researcher, "enable_parallel_tool_execution", True),
                }
            )
            initial_state['trace'] = trace

        if max_iterations:
            initial_state['max_iterations'] = max_iterations

        initial_state = self._seed_verifier_state(initial_state, skip_verification, max_revisions)

        thread_id = trace.get("run_id", "1") if trace else "1"
        config = {"configurable": {"thread_id": thread_id}}

        # Track if we've handled the approval
        approval_handled = False

        # Stream execution
        for output in self.graph.stream(initial_state, config=config):
            # Yield the output first
            yield output

            # Check if we hit an interrupt
            if "__interrupt__" in output and not approval_handled:
                # Get the current state from the graph
                current_snapshot = self.graph.get_state(config)
                current_state = current_snapshot.values

                # Check if this state needs approval (we interrupt after planning, before human_review)
                if isinstance(current_state, dict) and current_state.get('research_plan'):
                    # If auto-approve, set plan_approved to True
                    if auto_approve:
                        current_state['plan_approved'] = True
                        current_state['user_feedback'] = None
                        record_decision(
                            current_state.get("trace"),
                            node="human_review",
                            decision="plan_auto_approved_at_interrupt",
                            reason="auto_approve stream interrupt handling",
                        )
                        self.graph.update_state(config, current_state)
                    # Otherwise, ask user via callback
                    elif human_approval_callback and not current_state.get('plan_approved', False):
                        # Set the step for display
                        current_state['current_step'] = 'awaiting_approval'

                        # Call the approval callback
                        approved, feedback = human_approval_callback(current_state)

                        # Update the state
                        if approved:
                            current_state['plan_approved'] = True
                            current_state['user_feedback'] = None
                            record_decision(
                                current_state.get("trace"),
                                node="human_review",
                                decision="plan_approved",
                                reason="human callback approved the plan",
                            )
                        else:
                            current_state['plan_approved'] = False
                            current_state['user_feedback'] = feedback
                            record_decision(
                                current_state.get("trace"),
                                node="human_review",
                                decision="plan_rejected",
                                reason="human callback requested revision",
                                metadata={"feedback": feedback},
                            )

                        # Update graph state
                        self.graph.update_state(config, current_state)

                    approval_handled = True

                    # Continue from this point
                    for continue_output in self.graph.stream(None, config=config):
                        yield continue_output
                    return  # Exit after handling approval

    def get_workflow_schema(self) -> dict:
        """
        Get the workflow schema/structure.

        Returns:
            Workflow schema dictionary
        """
        return {
            "nodes": [
                "coordinator",
                "planner",
                "human_review",
                "researcher",
                "rapporteur"
            ],
            "edges": [
                ("coordinator", "planner"),
                ("planner", "human_review"),
                ("human_review", ["planner", "researcher"]),
                ("researcher", ["researcher", "rapporteur"]),
                ("rapporteur", "END")
            ],
            "entry_point": "coordinator",
            "conditional_edges": [
                {
                    "from": "human_review",
                    "function": "should_continue_research",
                    "destinations": ["planner", "researcher"]
                },
                {
                    "from": "researcher",
                    "function": "should_generate_report",
                    "destinations": ["researcher", "rapporteur"]
                }
            ]
        }

    def visualize(self, output_path: Optional[str] = None) -> str:
        """
        Visualize the workflow graph.

        Args:
            output_path: Optional path to save the visualization

        Returns:
            Path to the visualization file or visualization string
        """
        try:
            mermaid = self.graph.get_graph().draw_mermaid()

            if output_path:
                with open(output_path, 'w') as f:
                    f.write(mermaid)
                return output_path
            else:
                return mermaid
        except Exception as e:
            return f"Visualization not available: {str(e)}"
