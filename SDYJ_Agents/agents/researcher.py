"""
Researcher Agent

This module implements the Researcher agent, which is responsible for
executing information retrieval tasks.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional
from ..workflow.state import ResearchState, SubTask, SearchResult
from ..tools.tavily_search import TavilySearch
from ..tools.arxiv_search import ArxivSearch
from ..tools.mcp_client import MCPClient
from ..llm.base import BaseLLM
from ..prompts.loader import PromptLoader
from ..utils.evidence import merge_evidence_items, normalize_search_batch
from ..utils.structured_output import generate_json_object
from ..utils.tracing import record_tool_call, record_trace_event
from ..workflow.parallel_executor import (
    SearchJobResult,
    build_search_jobs,
    run_search_jobs_sync,
)


# Below this average relevance score we treat a batch as "weak" and consider
# triggering reflection. Tavily relevance is in [0, 1]; arXiv has no relevance
# field so we treat its results as relevance=0.5 for the purpose of this check
# (see ``_average_relevance`` for the implementation).
WEAK_RELEVANCE_THRESHOLD = 0.5
# At or above this fraction of failing tool calls in a single task we always
# trigger reflection, even if the few successful calls had decent results.
HIGH_ERROR_RATE_THRESHOLD = 0.5

REFLECTION_JSON_SCHEMA = {
    "type": "object",
    "required": ["diagnosis", "rewritten_queries"],
    "properties": {
        "diagnosis": {"type": "string"},
        "rewritten_queries": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 0,
            "maxItems": 2,
        },
    },
    "additionalProperties": True,
}


class Researcher:
    """
    Researcher agent - information collection component.

    Responsibilities:
    - Execute information retrieval tasks
    - Search from multiple data sources
    - Filter and organize search results
    - Aggregate results from different sources
    - Extract relevant information
    """

    def __init__(
        self,
        llm: BaseLLM,
        tavily_api_key: Optional[str] = None,
        mcp_server_url: Optional[str] = None,
        mcp_api_key: Optional[str] = None,
        mcp_transport: Optional[str] = None,
        mcp_tool_name: str = "web_search",
        mcp_config_path: Optional[str] = None,
        mcp_server_name: Optional[str] = None,
        mcp_command: Optional[str] = None,
        mcp_args: Optional[List[str]] = None,
        mcp_env: Optional[Dict[str, str]] = None,
        mcp_query_arg: str = "query",
        mcp_tool_args: Optional[Dict[str, Any]] = None,
        enable_reflection: bool = True,
        enable_parallel_tool_execution: bool = True,
        parallel_concurrency_limit: int = 4,
    ):
        """
        Initialize the Researcher.

        Args:
            llm: Language model instance for processing
            tavily_api_key: Tavily API key (optional)
            mcp_server_url: MCP server URL (optional)
            mcp_api_key: MCP API key (optional)
            mcp_transport: MCP transport (legacy_http, streamable_http, or stdio)
            mcp_tool_name: Default MCP search tool name
            mcp_config_path: Claude-style MCP config JSON path
            mcp_server_name: Server key inside the MCP config
            mcp_command: Direct stdio MCP server command
            mcp_args: Direct stdio MCP server args
            mcp_env: Direct stdio MCP server environment overrides
            enable_reflection: When True (the default in v0.6+), the
                researcher inspects the task's results after the first pass
                and asks the LLM to rewrite weak queries before giving up.
                Set to False to restore v0.5 single-pass behavior — useful
                for the v0.5-vs-v0.6 ablation in evaluation runs.
            enable_parallel_tool_execution: When True (the default in v0.6+),
                run the query/source lookups inside one task concurrently.
                Set to False to restore v0.5-style sequential retrieval.
            parallel_concurrency_limit: Maximum number of in-flight tool calls
                inside one task when parallel execution is enabled.
        """
        self.llm = llm
        self.tavily = TavilySearch(tavily_api_key) if tavily_api_key else None
        self.arxiv = ArxivSearch()
        has_mcp_config = bool(mcp_server_url or mcp_config_path or mcp_command)
        self.mcp = (
            MCPClient(
                server_url=mcp_server_url,
                api_key=mcp_api_key,
                transport=mcp_transport,
                default_tool_name=mcp_tool_name,
                config_path=mcp_config_path,
                server_name=mcp_server_name,
                command=mcp_command,
                args=mcp_args,
                env=mcp_env,
                query_argument=mcp_query_arg,
                tool_arguments=mcp_tool_args,
            )
            if has_mcp_config
            else None
        )
        self.prompt_loader = PromptLoader()
        self.enable_reflection = enable_reflection
        self.enable_parallel_tool_execution = enable_parallel_tool_execution
        self.parallel_concurrency_limit = max(1, int(parallel_concurrency_limit or 1))

    def execute_task(self, state: ResearchState, task: SubTask) -> ResearchState:
        """
        Execute a research task.

        v0.6 behavior: after the scheduled queries run, evaluate the result
        quality. If results are empty / low relevance / dominated by tool
        errors, ask the LLM to rewrite the queries and run them once more.
        Reflection is single-shot per task (controlled by ``_reflected``)
        so an adversarial source cannot induce an infinite reflection loop.
        Disable entirely with ``enable_reflection=False``.

        Args:
            state: Current research state
            task: Task to execute

        Returns:
            Updated state with research results
        """
        results: List[Dict[str, Any]] = []
        original_queries = list(task.get('search_queries') or [])
        sources = list(task.get('sources') or [])

        first_pass = self._run_queries(state, task, original_queries, sources)
        results.extend(first_pass)

        if (
            self.enable_reflection
            and not task.get('_reflected')
            and self._should_reflect(first_pass)
        ):
            rewritten = self._reflect_and_rewrite(
                state=state,
                task=task,
                first_pass=first_pass,
                sources=sources,
                original_queries=original_queries,
            )
            if rewritten:
                task['_reflected'] = True
                second_pass = self._run_queries(state, task, rewritten, sources)
                results.extend(second_pass)
                # Surface the rewritten queries on the task itself so the
                # report and the trace both show what was actually executed.
                task['search_queries'] = original_queries + rewritten
            else:
                task['_reflected'] = True

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

    def _run_queries(
        self,
        state: ResearchState,
        task: SubTask,
        queries: List[str],
        sources: List[str],
    ) -> List[Dict[str, Any]]:
        """Run a list of (query, source) pairs and return raw result batches.

        Pulled out as a helper so the first pass and a reflection-driven
        retry share the exact same trace-recording / latency-measuring path.
        """
        jobs = build_search_jobs(queries, sources)
        if (
            self.enable_parallel_tool_execution
            and self.parallel_concurrency_limit > 1
            and len(jobs) > 1
        ):
            return self._run_queries_parallel(state, task, queries, sources)
        return self._run_queries_sequential(state, task, queries, sources)

    def _run_queries_sequential(
        self,
        state: ResearchState,
        task: SubTask,
        queries: List[str],
        sources: List[str],
    ) -> List[Dict[str, Any]]:
        """Run search jobs one at a time, preserving the legacy v0.5 order."""
        results: List[Dict[str, Any]] = []
        for query in queries:
            for source in sources:
                started = time.perf_counter()
                result = self._search(query, source)
                latency_ms = int(round((time.perf_counter() - started) * 1000))
                self._record_search_outcome(
                    state=state,
                    task=task,
                    query=query,
                    source=source,
                    result=result,
                    latency_ms=latency_ms,
                    results=results,
                )
        return results

    def _run_queries_parallel(
        self,
        state: ResearchState,
        task: SubTask,
        queries: List[str],
        sources: List[str],
    ) -> List[Dict[str, Any]]:
        """Run search jobs concurrently and record each tool call separately."""
        job_results = run_search_jobs_sync(
            build_search_jobs(queries, sources),
            self._search_async,
            concurrency_limit=self.parallel_concurrency_limit,
        )
        results: List[Dict[str, Any]] = []
        for job_result in job_results:
            self._record_parallel_search_outcome(state, task, job_result, results)
        return results

    def _record_parallel_search_outcome(
        self,
        state: ResearchState,
        task: SubTask,
        job_result: SearchJobResult,
        results: List[Dict[str, Any]],
    ) -> None:
        self._record_search_outcome(
            state=state,
            task=task,
            query=job_result.job.query,
            source=job_result.job.source,
            result=job_result.result,
            latency_ms=job_result.latency_ms,
            results=results,
        )

    def _record_search_outcome(
        self,
        state: ResearchState,
        task: SubTask,
        query: str,
        source: str,
        result: Optional[SearchResult],
        latency_ms: int,
        results: List[Dict[str, Any]],
    ) -> None:
        """Normalize state and trace updates for one tool call outcome."""
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

    async def _search_async(self, query: str, source: str) -> Optional[SearchResult]:
        """Async search adapter used by the parallel executor."""
        source = source.lower().strip()
        if source == 'mcp' and self.mcp:
            result = self.mcp.search(query)
            if asyncio.iscoroutine(result):
                return await result
            return result
        # Preserve the legacy _search extension point for sync tools and tests,
        # while keeping Tavily/arXiv calls off the event loop via to_thread.
        return await asyncio.to_thread(self._search, query, source)

    @staticmethod
    def _should_reflect(batches: List[Dict[str, Any]]) -> bool:
        """Decide whether the first pass was poor enough to warrant reflection.

        Returns True when *any* of:
        - no batches were even produced (sources misconfigured),
        - every batch returned zero results,
        - the average relevance score across all hits was below the threshold,
        - half or more of the batches reported a tool error.
        """
        if not batches:
            return True

        total_hits = sum(len(b.get('results') or []) for b in batches)
        if total_hits == 0:
            return True

        error_count = sum(1 for b in batches if b.get('error'))
        if error_count / len(batches) >= HIGH_ERROR_RATE_THRESHOLD:
            return True

        avg_relevance = Researcher._average_relevance(batches)
        if avg_relevance is not None and avg_relevance < WEAK_RELEVANCE_THRESHOLD:
            return True
        return False

    @staticmethod
    def _average_relevance(batches: List[Dict[str, Any]]) -> Optional[float]:
        """Average relevance across all individual results.

        Sources without a relevance score (e.g. arXiv) contribute a neutral
        0.5 so they neither force a reflection by themselves nor block one
        when paired with a clearly-weak Tavily batch.
        """
        scores: List[float] = []
        for batch in batches:
            for item in batch.get('results') or []:
                raw = item.get('relevance_score')
                if isinstance(raw, (int, float)):
                    scores.append(max(0.0, min(1.0, float(raw))))
                else:
                    scores.append(0.5)
        if not scores:
            return None
        return sum(scores) / len(scores)

    def _reflect_and_rewrite(
        self,
        state: ResearchState,
        task: SubTask,
        first_pass: List[Dict[str, Any]],
        sources: List[str],
        original_queries: List[str],
    ) -> List[str]:
        """Ask the LLM to diagnose the failure and propose new queries.

        Returns the list of rewritten queries (possibly empty if the LLM
        gave us nothing usable). The reflection itself is recorded in the
        trace as an event so it is visible in inspect-run --timeline.
        """
        diagnosis_inputs = self._summarize_first_pass(first_pass)
        evidence_terms = self._collect_known_terms(state)

        try:
            prompt = self.prompt_loader.load(
                'researcher_reflect',
                task_id=task.get('task_id'),
                task_description=task.get('description', ''),
                sources=', '.join(sources),
                failed_queries=original_queries,
                result_count=diagnosis_inputs['result_count'],
                avg_relevance=(
                    f"{diagnosis_inputs['avg_relevance']:.2f}"
                    if diagnosis_inputs['avg_relevance'] is not None
                    else 'n/a'
                ),
                error_rate=f"{diagnosis_inputs['error_rate']:.2f}",
                errors=diagnosis_inputs['error_messages'],
                evidence_terms=evidence_terms,
            )
        except FileNotFoundError:
            # If the reflect prompt is missing for any reason, fail open —
            # we just skip reflection rather than crashing the workflow.
            return []

        try:
            parsed = generate_json_object(
                self.llm,
                prompt,
                schema=REFLECTION_JSON_SCHEMA,
                temperature=0.4,
                max_tokens=500,
            )
        except Exception as exc:
            record_trace_event(
                state.get('trace'),
                event_type='reflection',
                name='researcher_reflect_failed',
                node='researcher',
                status='error',
                metadata={
                    'task_id': task.get('task_id'),
                    'error': str(exc),
                },
            )
            return []

        rewritten = self._clean_rewritten_queries(parsed.get('rewritten_queries') or [])

        record_trace_event(
            state.get('trace'),
            event_type='reflection',
            name='researcher_reflect',
            node='researcher',
            status='ok' if rewritten else 'noop',
            input_snapshot={
                'task_id': task.get('task_id'),
                'original_queries': original_queries,
                'result_count': diagnosis_inputs['result_count'],
                'avg_relevance': diagnosis_inputs['avg_relevance'],
                'error_rate': diagnosis_inputs['error_rate'],
            },
            output_snapshot={
                'rewritten_queries': rewritten,
            },
            metadata={'task_id': task.get('task_id')},
        )

        # Trace metrics counter so v0.5-vs-v0.6 dashboards can show how
        # often reflection actually fired.
        trace = state.get('trace')
        if trace is not None and rewritten:
            trace.setdefault('metrics', {})
            trace['metrics']['reflection_count'] = (
                int(trace['metrics'].get('reflection_count', 0)) + 1
            )

        return rewritten

    @staticmethod
    def _summarize_first_pass(batches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compact failure summary for the reflection prompt."""
        result_count = sum(len(b.get('results') or []) for b in batches)
        error_messages = [b.get('error') for b in batches if b.get('error')]
        error_rate = len(error_messages) / len(batches) if batches else 1.0
        return {
            'result_count': result_count,
            'avg_relevance': Researcher._average_relevance(batches),
            'error_rate': error_rate,
            'error_messages': error_messages,
        }

    @staticmethod
    def _collect_known_terms(state: ResearchState) -> str:
        """Pull a few representative terms from already-collected evidence.

        Helps the LLM reflection step bias toward terminology the available
        sources actually index. Keeps the result short so we don't blow the
        prompt budget.
        """
        evidence_items = state.get('evidence_items') or []
        seen = []
        for item in evidence_items[:8]:
            title = (item.get('title') or '').strip()
            domain = (item.get('domain') or '').strip()
            if title:
                seen.append(f"{title} ({domain})" if domain else title)
        if not seen:
            return '(no prior evidence yet)'
        return '; '.join(seen)

    @staticmethod
    def _parse_reflection_response(response: str) -> List[str]:
        """Extract ``rewritten_queries`` from the reflection LLM output.

        Tolerant of fenced code blocks and prose preambles. Returns an empty
        list when the response is unusable (defensive: skipping reflection
        is always safer than running junk queries).
        """
        if not response:
            return []
        try:
            from ..llm.base import parse_json_object

            parsed = parse_json_object(response)
        except (ValueError, json.JSONDecodeError):
            return []
        rewritten = parsed.get('rewritten_queries') or []
        return Researcher._clean_rewritten_queries(rewritten)

    @staticmethod
    def _clean_rewritten_queries(rewritten: Any) -> List[str]:
        """Validate, strip, de-duplicate, and cap rewritten queries."""
        if not isinstance(rewritten, list):
            return []
        # Strip blanks and de-dup while preserving order.
        clean: List[str] = []
        seen_lower = set()
        for item in rewritten:
            query = str(item).strip()
            if not query:
                continue
            key = query.lower()
            if key in seen_lower:
                continue
            seen_lower.add(key)
            clean.append(query)
        # Two queries is plenty; more means the LLM ignored the prompt.
        return clean[:2]

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

    def extract_relevant_info(self, state: ResearchState) -> str:
        """
        Extract relevant information from all research results.

        Args:
            state: Current research state

        Returns:
            Extracted and summarized information
        """
        results = state.get('research_results', [])

        if not results:
            return "No research results available."

        # Compile all search results
        all_items = []
        for result in results:
            for item in result.get('results', []):
                all_items.append({
                    'source': result.get('source'),
                    'query': result.get('query'),
                    'title': item.get('title'),
                    'snippet': item.get('snippet'),
                    'url': item.get('url')
                })

        # Use LLM to extract and summarize
        prompt = self.prompt_loader.load(
            'researcher_extract_info',
            query=state['query'],
            search_results=self._format_results_for_prompt(all_items[:20])  # Limit to top 20 results
        )

        summary = self.llm.generate(prompt, temperature=0.5)
        return summary

    def _format_results_for_prompt(self, items: List[Dict]) -> str:
        """
        Format search results for LLM prompt.

        Args:
            items: List of search result items

        Returns:
            Formatted string
        """
        formatted = []
        for i, item in enumerate(items, 1):
            formatted.append(f"\n{i}. [{item.get('source')}] {item.get('title', 'No title')}")
            formatted.append(f"   URL: {item.get('url', 'N/A')}")
            formatted.append(f"   {item.get('snippet', 'No snippet')[:200]}...")

        return '\n'.join(formatted)

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
