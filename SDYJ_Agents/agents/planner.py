"""
Planner Agent

This module implements the Planner agent, which is responsible for
creating and managing research plans.
"""

import json
from typing import Optional
from ..workflow.state import ResearchState, PlanStructure, SubTask
from ..llm.base import BaseLLM
from ..prompts.loader import PromptLoader
from ..utils.evidence import format_evidence_for_prompt
from ..utils.structured_output import generate_json_object


# After this many completed sub-tasks, give the Planner a chance to refine
# the *remaining* sub-tasks based on the evidence collected so far. Set to
# something larger than 1 so the refine call has actual signal to work
# with, but small enough that mid-flight pivots can still happen.
DEFAULT_REFINE_AFTER_N_TASKS = 2


PLAN_JSON_SCHEMA = {
    "type": "object",
    "required": ["research_goal", "sub_tasks", "completion_criteria", "estimated_iterations"],
    "properties": {
        "research_goal": {"type": "string"},
        "sub_tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["task_id", "description", "search_queries", "sources", "priority"],
                "properties": {
                    "task_id": {"type": "integer"},
                    "description": {"type": "string"},
                    "search_queries": {"type": "array", "items": {"type": "string"}},
                    "sources": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["tavily", "arxiv", "mcp"]},
                    },
                    "priority": {"type": "integer"},
                    "status": {"type": "string"},
                },
                "additionalProperties": True,
            },
        },
        "completion_criteria": {"type": "string"},
        "estimated_iterations": {"type": "integer"},
        "refinement_rationale": {"type": "string"},
    },
    "additionalProperties": True,
}


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

    def __init__(
        self,
        llm: BaseLLM,
        enable_plan_refinement: bool = True,
        refine_after_n_tasks: int = DEFAULT_REFINE_AFTER_N_TASKS,
    ):
        """
        Initialize the Planner.

        Args:
            llm: Language model instance for planning
            enable_plan_refinement: When True (the default in v0.6+), the
                workflow asks the planner to refine remaining sub-tasks
                once enough sub-tasks have completed and there is real
                evidence to react to. Set to False to restore the v0.5
                fixed-plan behaviour — useful for the v0.5-vs-v0.6
                ablation in evaluation runs.
            refine_after_n_tasks: How many sub-tasks must complete before
                refinement is triggered. The default is intentionally
                small enough that a 3–5 task plan can still pivot.
        """
        self.llm = llm
        self.prompt_loader = PromptLoader()
        self.enable_plan_refinement = enable_plan_refinement
        self.refine_after_n_tasks = max(1, int(refine_after_n_tasks))

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

        try:
            plan = generate_json_object(
                self.llm,
                prompt,
                schema=PLAN_JSON_SCHEMA,
                temperature=0.7,
            )

            # Add status to subtasks
            for task in plan.get('sub_tasks', []):
                task['status'] = 'pending'

            # Update state
            state['research_plan'] = plan
            state['estimated_iterations'] = plan.get('estimated_iterations', 3)

        except (ValueError, TypeError, json.JSONDecodeError):
            # Create fallback plan
            plan = self._create_fallback_plan(query)
            state['research_plan'] = plan
            state['estimated_iterations'] = plan.get('estimated_iterations', 2)

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

        try:
            modified_plan = generate_json_object(
                self.llm,
                prompt,
                schema=PLAN_JSON_SCHEMA,
                temperature=0.7,
            )
            state['research_plan'] = modified_plan
        except (ValueError, TypeError, json.JSONDecodeError):
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

    def refine_plan(self, state: ResearchState) -> ResearchState:
        """Adapt the *remaining* sub-tasks based on what we have learned.

        Called once mid-flight (after ``DEFAULT_REFINE_AFTER_N_TASKS``
        sub-tasks have completed) so the Planner can drop redundant work,
        tighten weak queries, or add a follow-up sub-task that the
        evidence itself surfaced. The set of completed sub-tasks is kept
        intact — only pending sub-tasks may be modified.

        Sets ``state['plan_refined'] = True`` so the workflow does not
        re-enter refinement on the same run.
        """
        plan = state.get('research_plan') or {}
        sub_tasks = list(plan.get('sub_tasks') or [])
        evidence_items = state.get('evidence_items') or []

        completed = [task for task in sub_tasks if task.get('status') == 'completed']
        remaining = [task for task in sub_tasks if task.get('status') != 'completed']
        if not remaining:
            # Nothing left to refine; mark refined to avoid retrying.
            state['plan_refined'] = True
            return state

        prompt = self.prompt_loader.load(
            'planner_refine_plan',
            query=state.get('query', ''),
            research_goal=plan.get('research_goal', state.get('query', '')),
            completed_count=len(completed),
            full_plan_subtasks=json.dumps(sub_tasks, ensure_ascii=False, indent=2),
            completed_subtasks_brief=self._brief_subtasks(completed),
            remaining_subtasks_brief=self._brief_subtasks(remaining),
            evidence_so_far=format_evidence_for_prompt(evidence_items, limit=15)
            if evidence_items
            else '(no evidence collected yet)',
        )

        try:
            refined_plan = generate_json_object(
                self.llm,
                prompt,
                schema=PLAN_JSON_SCHEMA,
                temperature=0.3,
                max_tokens=2000,
            )
        except Exception:
            # If the refine call fails we fall back to the original plan.
            state['plan_refined'] = True
            return state

        if refined_plan:
            self._merge_refined_plan_into_state(state, refined_plan, completed)

        # Always set plan_refined so we don't retry. If parsing failed we
        # still mark it refined; the original plan stays in place.
        state['plan_refined'] = True
        return state

    def _brief_subtasks(self, subtasks: list) -> str:
        if not subtasks:
            return '(none)'
        lines = []
        for task in subtasks:
            queries = task.get('search_queries') or []
            qstr = ', '.join(f'"{q}"' for q in queries[:3])
            lines.append(
                f"- [{task.get('task_id')}] ({task.get('status', 'pending')}) "
                f"{task.get('description', '')} | queries: {qstr}"
            )
        return '\n'.join(lines)

    @staticmethod
    def _parse_refined_plan(response: str) -> Optional[dict]:
        """Pull the JSON plan out of the LLM response, tolerating fences."""
        try:
            from ..llm.base import parse_json_object

            return parse_json_object(response)
        except (ValueError, json.JSONDecodeError):
            return None

    @staticmethod
    def _merge_refined_plan_into_state(
        state: ResearchState,
        refined_plan: dict,
        completed_subtasks: list,
    ) -> None:
        """Apply the refined plan, preserving completed task metadata.

        We always keep the *actual* completed-task records that the
        Researcher updated (with status=completed) — even if the LLM left
        them out of its output by mistake — and append the refined
        non-completed tasks behind them. This guarantees we never lose
        evidence-of-execution that the trace already has.
        """
        if not refined_plan or not isinstance(refined_plan, dict):
            return
        new_subtasks_raw = refined_plan.get('sub_tasks') or []
        if not isinstance(new_subtasks_raw, list):
            return

        completed_ids = {task.get('task_id') for task in completed_subtasks}
        new_pending = []
        for task in new_subtasks_raw:
            if not isinstance(task, dict):
                continue
            # The LLM is told to preserve completed tasks; in practice we
            # *re-insert* the originals from state to be safe, so we drop
            # any duplicate/edited completed entries here.
            if task.get('task_id') in completed_ids:
                continue
            task.setdefault('status', 'pending')
            # Reset _reflected so a refined query gets a fresh chance to
            # trigger reflection if it also fails.
            task.pop('_reflected', None)
            new_pending.append(task)

        merged = list(completed_subtasks) + new_pending
        # Re-key any task with a missing or duplicate task_id so downstream
        # code (Researcher, Verifier prompts) keeps working.
        existing_ids = {
            Planner._normalize_task_id(task.get('task_id'))
            for task in completed_subtasks
        }
        existing_ids.discard(None)
        next_id = max(existing_ids, default=0) + 1
        for task in new_pending:
            normalized_id = Planner._normalize_task_id(task.get('task_id'))
            if normalized_id is None or normalized_id in existing_ids:
                while next_id in existing_ids:
                    next_id += 1
                task['task_id'] = next_id
                existing_ids.add(next_id)
                next_id += 1
            else:
                task['task_id'] = normalized_id
                existing_ids.add(normalized_id)
                next_id = max(next_id, normalized_id + 1)

        plan = state.get('research_plan') or {}
        plan['sub_tasks'] = merged
        if 'research_goal' in refined_plan:
            plan['research_goal'] = refined_plan['research_goal']
        if 'completion_criteria' in refined_plan:
            plan['completion_criteria'] = refined_plan['completion_criteria']
        if 'estimated_iterations' in refined_plan and isinstance(
            refined_plan['estimated_iterations'], int
        ):
            plan['estimated_iterations'] = refined_plan['estimated_iterations']
        if 'refinement_rationale' in refined_plan:
            plan.setdefault('history', []).append(
                {
                    'event': 'plan_refined',
                    'rationale': refined_plan['refinement_rationale'],
                    'completed_count': len(completed_subtasks),
                }
            )
        state['research_plan'] = plan

    @staticmethod
    def _normalize_task_id(value) -> Optional[int]:
        """Return a positive integer task id, or None for unusable IDs."""
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value if value > 0 else None
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isdigit():
                parsed = int(stripped)
                return parsed if parsed > 0 else None
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
        output.append(f"Research Goal: {plan.get('research_goal', 'N/A')}")
        output.append(f"\nEstimated Iterations: {plan.get('estimated_iterations', 'N/A')}")
        output.append(f"\nCompletion Criteria: {plan.get('completion_criteria', 'N/A')}")
        output.append("\n\nSubtasks:")

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
