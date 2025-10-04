"""
Research Workflow Graph

This module creates and manages the LangGraph workflow for the research system.
"""

from typing import Optional
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from .state import ResearchState
from .nodes import WorkflowNodes
from ..agents.coordinator import Coordinator
from ..agents.planner import Planner
from ..agents.researcher import Researcher
from ..agents.rapporteur import Rapporteur


def create_research_graph(
    coordinator: Coordinator,
    planner: Planner,
    researcher: Researcher,
    rapporteur: Rapporteur
):
    """
    Create the research workflow graph.

    Args:
        coordinator: Coordinator agent instance
        planner: Planner agent instance
        researcher: Researcher agent instance
        rapporteur: Rapporteur agent instance

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
        rapporteur: Rapporteur
    ):
        """
        Initialize the research workflow.

        Args:
            coordinator: Coordinator agent
            planner: Planner agent
            researcher: Researcher agent
            rapporteur: Rapporteur agent
        """
        self.coordinator = coordinator
        self.planner = planner
        self.researcher = researcher
        self.rapporteur = rapporteur
        self.graph = create_research_graph(
            coordinator, planner, researcher, rapporteur
        )

    def run(
        self,
        query: str,
        max_iterations: Optional[int] = None,
        auto_approve: bool = False
    ) -> dict:
        """
        Run the research workflow.

        Args:
            query: Research query
            max_iterations: Maximum number of research iterations
            auto_approve: Whether to auto-approve the research plan

        Returns:
            Final research state
        """
        # Initialize state
        initial_state = self.coordinator.initialize_research(query, auto_approve=auto_approve)

        if max_iterations:
            initial_state['max_iterations'] = max_iterations

        # Run the graph with thread configuration for checkpointer
        config = {"configurable": {"thread_id": "1"}}
        final_state = self.graph.invoke(initial_state, config=config)

        return final_state

    def stream(
        self,
        query: str,
        max_iterations: Optional[int] = None,
        auto_approve: bool = False
    ):
        """
        Stream the research workflow execution.

        Args:
            query: Research query
            max_iterations: Maximum number of research iterations
            auto_approve: Whether to auto-approve the research plan

        Yields:
            State updates during execution
        """
        # Initialize state
        initial_state = self.coordinator.initialize_research(query, auto_approve=auto_approve)

        if max_iterations:
            initial_state['max_iterations'] = max_iterations

        # Stream the graph execution with thread configuration for checkpointer
        config = {"configurable": {"thread_id": "1"}}
        for output in self.graph.stream(initial_state, config=config):
            yield output

    def stream_interactive(
        self,
        query: str,
        max_iterations: Optional[int] = None,
        auto_approve: bool = False,
        human_approval_callback = None
    ):
        """
        Stream the research workflow execution with interactive human approval.

        Args:
            query: Research query
            max_iterations: Maximum number of research iterations
            auto_approve: Whether to auto-approve the research plan
            human_approval_callback: Callback function for human approval
                                   Should return (approved: bool, feedback: str)

        Yields:
            State updates during execution
        """
        # Initialize state
        initial_state = self.coordinator.initialize_research(query, auto_approve=auto_approve)

        if max_iterations:
            initial_state['max_iterations'] = max_iterations

        config = {"configurable": {"thread_id": "1"}}

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
                        else:
                            current_state['plan_approved'] = False
                            current_state['user_feedback'] = feedback

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
            # Try to get graph visualization
            from langgraph.graph import Graph

            mermaid = self.graph.get_graph().draw_mermaid()

            if output_path:
                with open(output_path, 'w') as f:
                    f.write(mermaid)
                return output_path
            else:
                return mermaid
        except Exception as e:
            return f"Visualization not available: {str(e)}"
