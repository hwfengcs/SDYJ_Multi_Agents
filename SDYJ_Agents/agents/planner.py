"""
Planner Agent

This module implements the Planner agent, which is responsible for
creating and managing research plans.
"""

import json
from typing import Dict, List, Optional
from ..workflow.state import ResearchState, PlanStructure, SubTask
from ..llm.base import BaseLLM
from ..prompts.loader import PromptLoader


class Planner:
    """
    Planner agent - strategic planning component.

    Responsibilities:
    - Analyze research objectives
    - Create structured research plans
    - Break down complex tasks into subtasks
    - Accept and process user modifications
    - Evaluate context sufficiency
    - Decide when to continue research or generate report
    """

    def __init__(self, llm: BaseLLM):
        """
        Initialize the Planner.

        Args:
            llm: Language model instance for planning
        """
        self.llm = llm
        self.prompt_loader = PromptLoader()

    def create_research_plan(self, state: ResearchState) -> ResearchState:
        """
        Create a research plan based on the query.

        Args:
            state: Current research state

        Returns:
            Updated state with research plan
        """
        query = state['query']
        user_feedback = state.get('user_feedback', '')

        # Build prompt for plan generation
        prompt = self.prompt_loader.load(
            'planner_create_plan',
            query=query,
            user_feedback=user_feedback if user_feedback else None
        )

        # Generate plan
        response = self.llm.generate(prompt, temperature=0.7)

        # Parse JSON response
        try:
            # Extract JSON from response (in case LLM adds extra text)
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                plan = json.loads(json_str)
            else:
                # Fallback plan if parsing fails
                plan = self._create_fallback_plan(query)

            # Add status to subtasks
            for task in plan.get('sub_tasks', []):
                task['status'] = 'pending'

            # Update state
            state['research_plan'] = plan
            state['max_iterations'] = plan.get('estimated_iterations', 3)

        except json.JSONDecodeError:
            # Create fallback plan
            plan = self._create_fallback_plan(query)
            state['research_plan'] = plan

        return state

    def _create_fallback_plan(self, query: str) -> PlanStructure:
        """
        Create a simple fallback plan if JSON parsing fails.

        Args:
            query: Research query

        Returns:
            Basic research plan
        """
        return {
            'research_goal': query,
            'sub_tasks': [
                {
                    'task_id': 1,
                    'description': f'Research: {query}',
                    'search_queries': [query],
                    'sources': ['tavily'],
                    'status': 'pending',
                    'priority': 1
                }
            ],
            'completion_criteria': 'Gather sufficient information to answer the query',
            'estimated_iterations': 2
        }

    def modify_plan(self, state: ResearchState, modifications: str) -> ResearchState:
        """
        Modify the research plan based on user feedback.

        Args:
            state: Current research state
            modifications: User's modification requests

        Returns:
            Updated state with modified plan
        """
        current_plan = state['research_plan']

        prompt = self.prompt_loader.load(
            'planner_modify_plan',
            current_plan=json.dumps(current_plan, indent=2),
            modifications=modifications
        )

        response = self.llm.generate(prompt, temperature=0.7)

        # Parse modified plan
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                modified_plan = json.loads(json_str)
                state['research_plan'] = modified_plan
        except json.JSONDecodeError:
            # Keep current plan if parsing fails
            pass

        return state

    def evaluate_context_sufficiency(self, state: ResearchState) -> bool:
        """
        Evaluate whether gathered context is sufficient.

        Args:
            state: Current research state

        Returns:
            True if context is sufficient, False otherwise
        """
        query = state['query']
        plan = state['research_plan']
        results = state['research_results']
        iteration = state['iteration_count']
        max_iterations = state['max_iterations']

        # Check if max iterations reached
        if iteration >= max_iterations:
            return True

        # Check if we have results
        if not results:
            return False

        # Use LLM to evaluate sufficiency
        prompt = self.prompt_loader.load(
            'planner_evaluate_context',
            query=query,
            research_goal=plan.get('research_goal', query),
            completion_criteria=plan.get('completion_criteria', 'N/A'),
            results_count=len(results),
            current_iteration=iteration + 1,
            max_iterations=max_iterations
        )

        response = self.llm.generate(prompt, temperature=0.3).strip().upper()
        return response == "YES"

    def get_next_task(self, state: ResearchState) -> Optional[SubTask]:
        """
        Get the next pending task from the plan.

        Args:
            state: Current research state

        Returns:
            Next task to execute, or None if all tasks completed
        """
        plan = state.get('research_plan')
        if not plan:
            return None

        # Find first pending task by priority
        tasks = sorted(
            plan.get('sub_tasks', []),
            key=lambda t: (t.get('priority', 99), t.get('task_id', 0))
        )

        for task in tasks:
            if task.get('status') == 'pending':
                return task

        return None

    def format_plan_for_display(self, plan: PlanStructure) -> str:
        """
        Format plan for display to user.

        Args:
            plan: Research plan

        Returns:
            Formatted plan string
        """
        output = []
        output.append(f"=Ë Research Goal: {plan.get('research_goal', 'N/A')}")
        output.append(f"\n=Ê Estimated Iterations: {plan.get('estimated_iterations', 'N/A')}")
        output.append(f"\n Completion Criteria: {plan.get('completion_criteria', 'N/A')}")
        output.append("\n\n=Ý Subtasks:")

        for task in plan.get('sub_tasks', []):
            output.append(f"\n  {task['task_id']}. {task['description']}")
            output.append(f"     Queries: {', '.join(task.get('search_queries', []))}")
            output.append(f"     Sources: {', '.join(task.get('sources', []))}")
            output.append(f"     Priority: {task.get('priority', 'N/A')}")
            output.append(f"     Status: {task.get('status', 'pending')}")

        return ''.join(output)

    def __repr__(self) -> str:
        """String representation."""
        return f"Planner(llm={self.llm})"
