"""Hard evaluation scenarios for research-agent behavior."""

from __future__ import annotations

from typing import Any, Dict, List


Scenario = Dict[str, Any]


HARD_SCENARIOS: List[Scenario] = [
    {
        "id": "agent_reliability_hard",
        "title": "Agent reliability benchmark design",
        "query": (
            "为一个具备 RAG、网页检索、MCP 工具调用和人工审批节点的企业研究型 Agent "
            "设计一套可复现实验评测方案。要求覆盖任务成功率、证据可靠性、工具失败恢复、"
            "成本/延迟、人工干预收益，并给出 ablation study 和上线门槛。"
        ),
        "required_terms": [
            "evidence",
            "citation",
            "latency",
            "cost",
            "ablation",
            "tool",
            "human",
            "trace",
        ],
        "expected_sections": ["执行摘要", "核心发现", "深度分析", "来源概览", "参考资料", "结论"],
        "canned_results": {
            "tavily": [
                {
                    "title": "LangGraph production agents emphasize durable execution and human review",
                    "url": "https://docs.langchain.com/oss/python/langgraph/overview",
                    "snippet": (
                        "LangGraph positions durable execution, human-in-the-loop control, "
                        "memory, and debugging as core production-agent capabilities."
                    ),
                    "relevance_score": 0.94,
                    "metadata": {"published_date": "2026-04-01"},
                },
                {
                    "title": "OpenAI Agents SDK tracing records model and tool events",
                    "url": "https://openai.github.io/openai-agents-python/tracing/",
                    "snippet": (
                        "Tracing captures agent runs, LLM generations, tool calls, handoffs, "
                        "guardrails, and custom spans for debugging and evaluation."
                    ),
                    "relevance_score": 0.91,
                    "metadata": {"published_date": "2026-03-15"},
                },
                {
                    "title": "Duplicate OpenAI tracing mirror",
                    "url": "https://openai.github.io/openai-agents-python/tracing/",
                    "snippet": "Duplicate URL used to test evidence deduplication.",
                    "relevance_score": 0.4,
                    "metadata": {"published_date": "2026-03-15"},
                },
                {
                    "title": "Agent evaluation needs more than final-answer grading",
                    "url": "https://example.org/agent-eval-final-answer-is-not-enough",
                    "snippet": (
                        "A robust benchmark should inspect intermediate plans, retrieval "
                        "coverage, citation grounding, tool errors, and recovery behavior."
                    ),
                    "relevance_score": 0.88,
                    "metadata": {"published_date": "2026-01-20"},
                },
            ],
            "arxiv": [
                {
                    "title": "Tool-Augmented Agent Evaluation with Trace-Level Metrics",
                    "url": "https://arxiv.org/abs/2601.00001",
                    "snippet": (
                        "The paper proposes trace-level evaluation: tool success rate, "
                        "redundant call ratio, retry quality, and grounded claim coverage."
                    ),
                    "relevance_score": None,
                    "metadata": {
                        "published": "2026-01-03T00:00:00",
                        "authors": ["A. Researcher", "B. Engineer"],
                    },
                },
                {
                    "title": "Human Approval Gates Reduce Costly Agent Failures",
                    "url": "https://arxiv.org/abs/2602.00002",
                    "snippet": (
                        "Human-in-the-loop approval is most useful before broad retrieval "
                        "or irreversible tool actions; experiments report lower wasted cost."
                    ),
                    "relevance_score": None,
                    "metadata": {"published": "2026-02-12T00:00:00"},
                },
            ],
        },
    },
    {
        "id": "tool_failure_recovery_hard",
        "title": "Tool failure and evidence recovery",
        "query": (
            "评估一个多工具研究 Agent 在 Tavily 超时、arXiv 无结果、重复 URL、低质量来源混入时，"
            "如何保持报告可靠性。请设计故障注入实验、自动化指标和人工复核策略。"
        ),
        "required_terms": [
            "timeout",
            "retry",
            "duplicate",
            "dedup",
            "fallback",
            "citation",
            "quality",
            "human",
        ],
        "expected_sections": ["执行摘要", "核心发现", "深度分析", "来源概览", "参考资料", "结论"],
        "canned_results": {
            "tavily": [
                {
                    "title": "Search timeout incident report",
                    "url": "https://example.org/tool-timeout-incident",
                    "snippet": (
                        "Timeouts should be represented in traces with latency, empty result "
                        "count, retry budget, and final fallback decision."
                    ),
                    "relevance_score": 0.86,
                    "metadata": {"published_date": "2026-02-08"},
                },
                {
                    "title": "Source quality rubric for AI research agents",
                    "url": "https://example.org/source-quality-rubric",
                    "snippet": (
                        "Source quality scoring combines authority, freshness, specificity, "
                        "citation availability, and cross-source corroboration."
                    ),
                    "relevance_score": 0.9,
                    "metadata": {"published_date": "2026-01-29"},
                },
            ],
            "arxiv": [
                {
                    "title": "Robust Retrieval Agents under Tool Failures",
                    "url": "https://arxiv.org/abs/2603.00003",
                    "snippet": (
                        "Failure-aware retrieval agents improve answer quality when they expose "
                        "empty results, recover with fallback tools, and avoid hallucinated citations."
                    ),
                    "relevance_score": None,
                    "metadata": {"published": "2026-03-02T00:00:00"},
                }
            ],
        },
        "forced_errors": {
            "tavily": [
                {
                    "query_contains": "timeout",
                    "error": "simulated Tavily timeout for failure-injection coverage",
                }
            ]
        },
    },
    {
        "id": "mcp_rag_ops_hard",
        "title": "MCP + RAG operations evaluation",
        "query": (
            "为一个面向客服知识库的 MCP+RAG Agent 设计上线前评测：它需要调用 CRM、工单、知识库和网页搜索，"
            "同时满足隐私、拒答、安全边界、延迟 SLO 与成本预算。请给出场景集、指标、trace schema 和验收阈值。"
        ),
        "required_terms": [
            "mcp",
            "rag",
            "privacy",
            "refusal",
            "slo",
            "cost",
            "trace",
            "acceptance",
        ],
        "expected_sections": ["执行摘要", "核心发现", "深度分析", "来源概览", "参考资料", "结论"],
        "canned_results": {
            "tavily": [
                {
                    "title": "Model Context Protocol architecture",
                    "url": "https://modelcontextprotocol.io/docs/learn/architecture",
                    "snippet": (
                        "MCP describes clients, servers, tools, resources, prompts, JSON-RPC "
                        "communication, structured outputs, and error handling boundaries."
                    ),
                    "relevance_score": 0.95,
                    "metadata": {"published_date": "2026-04-10"},
                },
                {
                    "title": "Customer support AI safety acceptance gates",
                    "url": "https://example.org/customer-agent-safety-gates",
                    "snippet": (
                        "Pre-launch gates include privacy tests, refusal correctness, escalation "
                        "coverage, freshness checks, latency SLO, and cost budget adherence."
                    ),
                    "relevance_score": 0.87,
                    "metadata": {"published_date": "2026-03-19"},
                },
            ],
            "arxiv": [
                {
                    "title": "Evaluating Retrieval-Augmented Assistants in Regulated Domains",
                    "url": "https://arxiv.org/abs/2604.00004",
                    "snippet": (
                        "Regulated-domain RAG evaluation should combine retrieval freshness, "
                        "answer faithfulness, refusal behavior, privacy leakage tests, and latency."
                    ),
                    "relevance_score": None,
                    "metadata": {"published": "2026-04-05T00:00:00"},
                }
            ],
        },
    },
]


def list_scenarios() -> List[Scenario]:
    """Return all built-in hard scenarios."""
    return HARD_SCENARIOS


def get_scenario(scenario_id: str) -> Scenario:
    """Return one scenario by ID."""
    for scenario in HARD_SCENARIOS:
        if scenario["id"] == scenario_id:
            return scenario
    available = ", ".join(s["id"] for s in HARD_SCENARIOS)
    raise ValueError(f"Unknown scenario: {scenario_id}. Available: {available}")
