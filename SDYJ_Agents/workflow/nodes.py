"""
Workflow Nodes

This module defines the node functions for the LangGraph workflow.
"""

from typing import Dict, Any
from ..agents.coordinator import Coordinator
from ..agents.planner import Planner
from ..agents.researcher import Researcher
from ..agents.rapporteur import Rapporteur


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
        # Check if this is a simple query that was already handled
        if state.get('query_type') in ['GREETING', 'INAPPROPRIATE']:
            # Simple query already handled in initialize_research
            state['current_step'] = 'completed'
            return state

        # For research queries, delegate to planner
        state['current_step'] = 'coordinating'
        state = self.coordinator.delegate_to_planner(state)
        return state

    def planner_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Planner node - creates or updates research plan.

        Args:
            state: Current research state

        Returns:
            Updated state with research plan
        """
        state['current_step'] = 'planning'

        # If there's user feedback and a plan exists, modify it
        if state.get('user_feedback') and state.get('research_plan'):
            state = self.planner.modify_plan(state, state['user_feedback'])
        # Otherwise create a new plan
        elif not state.get('research_plan'):
            state = self.planner.create_research_plan(state)

        return state

    def human_review_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Human review node - pauses for user approval.

        Args:
            state: Current research state

        Returns:
            Updated state
        """
        state['current_step'] = 'awaiting_approval'

        # Check if auto-approve is enabled
        if state.get('auto_approve_plan', False):
            state['plan_approved'] = True

        # In actual implementation, this will pause and wait for user input
        # For now, we just mark the state
        return state

    def researcher_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Researcher node - executes research tasks.

        Args:
            state: Current research state

        Returns:
            Updated state with research results
        """
        state['current_step'] = 'researching'

        # Get next task from plan
        next_task = self.planner.get_next_task(state)

        if next_task:
            # Execute the task
            state = self.researcher.execute_task(state, next_task)
            state['current_task'] = next_task
            state['iteration_count'] += 1
        else:
            # No more tasks
            state['needs_more_research'] = False

        return state

    def rapporteur_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rapporteur node - generates final report.

        Args:
            state: Current research state

        Returns:
            Updated state with final report
        """
        state['current_step'] = 'generating_report'
        state = self.rapporteur.generate_report(state)
        return state

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
            return "end"

        # Otherwise, continue to planner for research
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
            return "planner"

        # If plan approved, start research
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
            return "rapporteur"

        # Check if context is sufficient
        if self.planner.evaluate_context_sufficiency(state):
            return "rapporteur"

        # Check if there are more tasks
        next_task = self.planner.get_next_task(state)
        if next_task:
            return "researcher"
        else:
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
