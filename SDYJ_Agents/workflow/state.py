"""
Research Workflow State Definition

This module defines the state structure for the research workflow.
"""

from typing import TypedDict, List, Annotated, Optional, Any
import operator


# Define state with Annotated fields for LangGraph
class ResearchState(TypedDict):
    """
    Research workflow state for LangGraph.

    This state is passed between all nodes in the workflow.
    """
    query: str
    research_plan: Optional[dict]
    plan_approved: bool
    research_results: Annotated[list, operator.add]  # Accumulates results
    current_task: Optional[dict]
    iteration_count: int
    max_iterations: int
    final_report: Optional[str]
    current_step: str
    needs_more_research: bool
    user_feedback: Optional[str]
    output_format: str  # "markdown" or "html"


class PlanStructure(TypedDict):
    """
    Structure for the research plan.
    """
    research_goal: str  # Overall research goal
    sub_tasks: List[dict]  # List of subtasks
    completion_criteria: str  # Criteria for completion
    estimated_iterations: int  # Estimated number of iterations needed


class SubTask(TypedDict):
    """
    Structure for a research subtask.
    """
    task_id: int  # Unique task ID
    description: str  # Task description
    search_queries: List[str]  # List of search queries
    sources: List[str]  # Data sources to use (tavily, arxiv, mcp)
    status: str  # Task status (pending, in_progress, completed)
    priority: Optional[int]  # Task priority


class SearchResult(TypedDict):
    """
    Structure for search results.
    """
    task_id: int  # Associated task ID
    query: str  # Search query
    source: str  # Data source (tavily, arxiv, mcp)
    results: List[dict]  # List of individual results
    timestamp: str  # Timestamp of search


class IndividualResult(TypedDict):
    """
    Structure for an individual search result.
    """
    title: str  # Result title
    url: Optional[str]  # Result URL (if applicable)
    snippet: str  # Result snippet/summary
    relevance_score: Optional[float]  # Relevance score
    metadata: Optional[dict]  # Additional metadata
