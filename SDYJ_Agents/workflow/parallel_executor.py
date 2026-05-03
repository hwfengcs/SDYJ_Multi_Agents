"""Small async executor for bounded parallel tool calls."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from threading import Thread
from typing import Any, Awaitable, Callable, Dict, List, Optional


@dataclass(frozen=True)
class SearchJob:
    """One query/source lookup scheduled inside a research task."""

    query: str
    source: str


@dataclass
class SearchJobResult:
    """Result plus timing for one scheduled lookup."""

    job: SearchJob
    result: Optional[Dict[str, Any]]
    latency_ms: int


AsyncSearchFn = Callable[[str, str], Awaitable[Optional[Dict[str, Any]]]]


def build_search_jobs(queries: List[str], sources: List[str]) -> List[SearchJob]:
    """Preserve the existing query-outer/source-inner execution order."""
    return [SearchJob(query=query, source=source) for query in queries for source in sources]


async def run_search_jobs(
    jobs: List[SearchJob],
    search_one: AsyncSearchFn,
    concurrency_limit: int = 4,
) -> List[SearchJobResult]:
    """Run search jobs concurrently while returning results in input order."""
    if not jobs:
        return []

    limit = max(1, int(concurrency_limit or 1))
    semaphore = asyncio.Semaphore(limit)

    async def _run_one(job: SearchJob) -> SearchJobResult:
        async with semaphore:
            started = time.perf_counter()
            try:
                result = await search_one(job.query, job.source)
            except Exception as exc:
                result = {
                    "query": job.query,
                    "source": job.source,
                    "results": [],
                    "error": str(exc),
                }
            latency_ms = int(round((time.perf_counter() - started) * 1000))
            return SearchJobResult(job=job, result=result, latency_ms=latency_ms)

    return await asyncio.gather(*(_run_one(job) for job in jobs))


def run_search_jobs_sync(
    jobs: List[SearchJob],
    search_one: AsyncSearchFn,
    concurrency_limit: int = 4,
) -> List[SearchJobResult]:
    """Synchronous wrapper used by the current Researcher API.

    If a caller is already inside an event loop, run the bounded async executor
    in a short-lived helper thread so the public Researcher API stays sync.
    """
    coroutine = run_search_jobs(jobs, search_one, concurrency_limit=concurrency_limit)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    holder: Dict[str, Any] = {}

    def _runner() -> None:
        try:
            holder["result"] = asyncio.run(coroutine)
        except BaseException as exc:
            holder["error"] = exc

    thread = Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in holder:
        raise holder["error"]
    return holder["result"]
