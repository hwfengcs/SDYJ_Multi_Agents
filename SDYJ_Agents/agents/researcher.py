"""
Researcher Agent

This module implements the Researcher agent, which is responsible for
executing information retrieval tasks.
"""

import time
from typing import Dict, List, Optional
from ..workflow.state import ResearchState, SubTask, SearchResult
from ..tools.tavily_search import TavilySearch
from ..tools.arxiv_search import ArxivSearch
from ..tools.mcp_client import MCPClient
from ..llm.base import BaseLLM
from ..utils.evidence import merge_evidence_items, normalize_search_batch
from ..utils.tracing import record_tool_call


class Researcher:
    """
    Researcher agent - information collection component.

    Responsibilities:
    - Execute information retrieval tasks
    - Search from multiple data sources
    - Filter and organize search results
    - Aggregate results from different sources
    """

    def __init__(
        self,
        llm: BaseLLM,
        tavily_api_key: Optional[str] = None,
        mcp_server_url: Optional[str] = None,
        mcp_api_key: Optional[str] = None
    ):
        """
        Initialize the Researcher.

        Args:
            llm: Language model instance for processing
            tavily_api_key: Tavily API key (optional)
            mcp_server_url: MCP server URL (optional)
            mcp_api_key: MCP API key (optional)
        """
        self.llm = llm
        self.tavily = TavilySearch(tavily_api_key) if tavily_api_key else None
        self.arxiv = ArxivSearch()
        self.mcp = MCPClient(mcp_server_url, mcp_api_key) if mcp_server_url else None

    def execute_task(self, state: ResearchState, task: SubTask) -> ResearchState:
        """
        Execute a research task.

        Args:
            state: Current research state
            task: Task to execute

        Returns:
            Updated state with research results
        """
        results = []

        # Execute searches for each query
        for query in task.get('search_queries', []):
            for source in task.get('sources', []):
                started = time.perf_counter()
                result = self._search(query, source)
                latency_ms = int(round((time.perf_counter() - started) * 1000))
                if result:
                    result['task_id'] = task['task_id']
                    result['latency_ms'] = latency_ms
                    results.append(result)
                    record_tool_call(
                        state.get('trace'),
                        source=result.get('source', source),
                        query=query,
                        task_id=task.get('task_id'),
                        latency_ms=latency_ms,
                        result_count=len(result.get('results', [])),
                        error=result.get('error'),
                        result=result,
                    )
                else:
                    record_tool_call(
                        state.get('trace'),
                        source=source,
                        query=query,
                        task_id=task.get('task_id'),
                        latency_ms=latency_ms,
                        result_count=0,
                        error="source unavailable or unsupported",
                    )

        # Add results to state
        if 'research_results' not in state:
            state['research_results'] = []

        state['research_results'].extend(results)
        evidence_items = state.get('evidence_items') or []
        for result in results:
            evidence_items = merge_evidence_items(
                evidence_items,
                normalize_search_batch(result),
            )
        state['evidence_items'] = evidence_items

        # Mark task as completed
        if state.get('research_plan'):
            for t in state['research_plan'].get('sub_tasks', []):
                if t.get('task_id') == task['task_id']:
                    t['status'] = 'completed'
                    break

        return state

    def _search(self, query: str, source: str) -> Optional[SearchResult]:
        """
        Perform search using specified source.

        Args:
            query: Search query
            source: Source name ('tavily', 'arxiv', 'mcp')

        Returns:
            Search results or None
        """
        source = source.lower().strip()
        try:
            if source == 'tavily' and self.tavily:
                return self.tavily.search(query)
            elif source == 'arxiv':
                return self.arxiv.search(query)
            elif source == 'mcp' and self.mcp:
                import asyncio
                result = self.mcp.search(query)
                if asyncio.iscoroutine(result):
                    return asyncio.run(result)
                return result
            else:
                return None
        except Exception as e:
            return {
                'query': query,
                'source': source,
                'results': [],
                'error': str(e)
            }

    def aggregate_results(self, results: List[SearchResult]) -> Dict:
        """
        Aggregate and organize search results.

        Args:
            results: List of search results

        Returns:
            Aggregated results summary
        """
        # Group results by source
        by_source = {}
        for result in results:
            source = result.get('source', 'unknown')
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(result)

        # Calculate statistics
        total_results = sum(len(r.get('results', [])) for r in results)

        return {
            'total_searches': len(results),
            'total_results': total_results,
            'by_source': {
                source: {
                    'count': len(source_results),
                    'total_items': sum(len(r.get('results', [])) for r in source_results)
                }
                for source, source_results in by_source.items()
            }
        }

    def __repr__(self) -> str:
        """String representation."""
        sources = []
        if self.tavily:
            sources.append('tavily')
        if self.arxiv:
            sources.append('arxiv')
        if self.mcp:
            sources.append('mcp')
        return f"Researcher(sources={sources})"
