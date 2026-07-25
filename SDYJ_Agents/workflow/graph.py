"""
Research Workflow Graph

This module creates and manages the LangGraph workflow for the research system.
"""

from pathlib import Path
from typing import Optional
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from .nodes import WorkflowNodes
from ..agents.coordinator import Coordinator
from ..agents.planner import Planner
from ..agents.researcher import Researcher
from ..agents.rapporteur import Rapporteur
from ..utils.tracing import record_decision


DEFAULT_MAX_ITERATIONS = 5


def build_invoke_config(thread_id: str, max_iterations: Optional[int] = None) -> dict:
    """Build the LangGraph invocation config for one run.

    The researcher self-loop consumes one super-step per iteration, so the
    recursion limit must scale with max_iterations or LangGraph's default (25)
    aborts long runs with GraphRecursionError.
    """
    iterations = max_iterations or DEFAULT_MAX_ITERATIONS
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": max(25, 2 * iterations + 10),
    }


def open_sqlite_checkpointer(db_path: str | Path):
    """Open a durable sqlite checkpointer; the caller owns the connection.

    Constructed directly (not via ``from_conn_string``, whose context-manager
    lifetime does not fit a workflow object). Imported lazily so MemorySaver
    paths never require langgraph-checkpoint-sqlite to be installed.
    """
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn)


def create_research_graph(
    coordinator: Coordinator,
    planner: Planner,
    researcher: Researcher,
    rapporteur: Rapporteur,
    checkpointer=None,
):
    """
    Create the research workflow graph.

    Args:
        coordinator: Coordinator agent instance
        planner: Planner agent instance
        researcher: Researcher agent instance
        rapporteur: Rapporteur agent instance
        checkpointer: Optional LangGraph checkpointer (defaults to in-memory)

    Returns:
        Compiled LangGraph workflow
    """
    # Create workflow nodes
    nodes = WorkflowNodes(coordinator, planner, researcher, rapporteur)

    # Initialize state graph
    workflow = StateGraph(dict)  # Use dict instead of TypedDict for compatibility

    # Add nodes to the graph
    workflow.add_node("coordinator", nodes.coordinator_node)
    workflow.add_node("planner", nodes.planner_node)
    workflow.add_node("human_review", nodes.human_review_node)
    workflow.add_node("researcher", nodes.researcher_node)
    workflow.add_node("rapporteur", nodes.rapporteur_node)

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

    # Rapporteur -> END
    workflow.add_edge("rapporteur", END)

    # Compile the graph with checkpointer
    # Add interrupt before human_review for human-in-the-loop
    if checkpointer is None:
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
        checkpointer=None,
    ):
        """
        Initialize the research workflow.

        Args:
            coordinator: Coordinator agent
            planner: Planner agent
            researcher: Researcher agent
            rapporteur: Rapporteur agent
            checkpointer: Optional durable checkpointer (defaults to in-memory)
        """
        self.coordinator = coordinator
        self.planner = planner
        self.researcher = researcher
        self.rapporteur = rapporteur
        self.checkpointer = checkpointer
        self.graph = create_research_graph(
            coordinator, planner, researcher, rapporteur, checkpointer=checkpointer
        )

    def close(self) -> None:
        """Release the checkpointer's sqlite connection.

        On Windows an open sqlite handle keeps the run directory locked, so
        callers must close durable workflows when done (no-op for MemorySaver).
        """
        conn = getattr(self.checkpointer, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def run(
        self,
        query: str,
        max_iterations: Optional[int] = None,
        auto_approve: bool = False,
        output_format: str = "markdown",
        trace: Optional[dict] = None
    ) -> dict:
        """
        Run the research workflow.

        Args:
            query: Research query
            max_iterations: Maximum number of research iterations
            auto_approve: Whether to auto-approve the research plan
            output_format: Output format for the final report ("markdown" or "html")

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
                }
            )
            initial_state['trace'] = trace

        if max_iterations:
            initial_state['max_iterations'] = max_iterations

        # Run the graph with thread configuration for checkpointer
        thread_id = trace.get("run_id", "1") if trace else "1"
        config = build_invoke_config(thread_id, max_iterations)
        final_state = self.graph.invoke(initial_state, config=config)

        return final_state

    def stream(
        self,
        query: str,
        max_iterations: Optional[int] = None,
        auto_approve: bool = False,
        output_format: str = "markdown",
        trace: Optional[dict] = None
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
                }
            )
            initial_state['trace'] = trace

        if max_iterations:
            initial_state['max_iterations'] = max_iterations

        # Stream the graph execution with thread configuration for checkpointer
        thread_id = trace.get("run_id", "1") if trace else "1"
        config = build_invoke_config(thread_id, max_iterations)
        for output in self.graph.stream(initial_state, config=config):
            yield output

    def stream_interactive(
        self,
        query: str,
        max_iterations: Optional[int] = None,
        auto_approve: bool = False,
        human_approval_callback = None,
        output_format: str = "markdown",
        trace: Optional[dict] = None
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
                }
            )
            initial_state['trace'] = trace

        if max_iterations:
            initial_state['max_iterations'] = max_iterations

        thread_id = trace.get("run_id", "1") if trace else "1"
        config = build_invoke_config(thread_id, max_iterations)

        yield from self._stream_with_approvals(
            initial_state, config, auto_approve, human_approval_callback
        )

    def resume_interactive(
        self,
        thread_id: str,
        max_iterations: Optional[int] = None,
        auto_approve: bool = False,
        human_approval_callback = None,
        trace: Optional[dict] = None,
    ):
        """Continue a checkpointed run from its last saved super-step.

        Works for both a pending human_review interrupt and a mid-research
        crash: streaming with a ``None`` input resumes from the checkpoint.

        Args:
            thread_id: The original run's thread id (its run_id)
            max_iterations: Optional new iteration budget
            auto_approve: Whether to auto-approve a pending plan review
            human_approval_callback: Callback for pending plan review
            trace: Trace object to attach for continued event recording

        Yields:
            State updates during execution
        """
        config = build_invoke_config(thread_id, max_iterations)
        snapshot = self.graph.get_state(config)
        values = getattr(snapshot, "values", None)
        if not isinstance(values, dict) or not values:
            raise ValueError(f"No checkpoint found for thread '{thread_id}'")

        # The graph uses StateGraph(dict): the whole state is ONE last-value
        # channel, so update_state must always receive the FULL state dict —
        # a partial dict would replace the state and drop every other key.
        if trace is not None:
            values["trace"] = trace
        if max_iterations:
            values["max_iterations"] = max_iterations
        self.graph.update_state(config, values)

        yield from self._stream_with_approvals(
            None, config, auto_approve, human_approval_callback
        )

    def _stream_with_approvals(
        self,
        stream_input,
        config: dict,
        auto_approve: bool,
        human_approval_callback,
    ):
        # LangGraph interrupts before every human_review node. A rejected plan
        # routes back to Planner and creates another interrupt, so the resume
        # loop must handle approval more than once.
        while True:
            interrupted = False

            for output in self.graph.stream(stream_input, config=config):
                yield output

                if "__interrupt__" not in output:
                    continue

                current_snapshot = self.graph.get_state(config)
                current_state = current_snapshot.values
                if not isinstance(current_state, dict) or not current_state.get('research_plan'):
                    return

                current_state['current_step'] = 'awaiting_approval'

                if not self._apply_approval(
                    config, current_state, auto_approve, human_approval_callback
                ):
                    return
                stream_input = None
                interrupted = True
                break

            if not interrupted:
                return

    def _apply_approval(
        self,
        config: dict,
        current_state: dict,
        auto_approve: bool,
        human_approval_callback,
    ) -> bool:
        """Resolve one human_review interrupt; False means stop streaming.

        The graph uses StateGraph(dict): the whole state is ONE last-value
        channel, so update_state must receive the FULL state dict — writing a
        partial dict would replace the state and drop every other key.
        """
        if auto_approve:
            current_state['plan_approved'] = True
            current_state['user_feedback'] = None
            record_decision(
                current_state.get("trace"),
                node="human_review",
                decision="plan_auto_approved_at_interrupt",
                reason="auto_approve stream interrupt handling",
            )
        elif human_approval_callback:
            approved, feedback = human_approval_callback(current_state)

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
        else:
            return False

        self.graph.update_state(config, current_state)
        return True

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
