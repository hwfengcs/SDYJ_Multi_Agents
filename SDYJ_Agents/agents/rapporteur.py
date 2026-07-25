"""
Rapporteur Agent

This module implements the Rapporteur agent, which is responsible for
generating the final research report.
"""

import json
import re
from typing import Dict, List
from datetime import datetime
from ..workflow.state import ResearchState
from ..llm.base import BaseLLM
from ..prompts.loader import PromptLoader
from ..utils.evidence import (
    append_citations,
    build_evidence_from_results,
    calculate_evidence_metrics,
    format_evidence_for_prompt,
    validate_citations,
)
from ..utils.tracing import record_degraded_event, record_report_summary


SECTION_FAILURE_PLACEHOLDER = "（本节生成失败：LLM 错误，已降级输出）"


class Rapporteur:
    """
    Rapporteur agent - report generation component.

    Responsibilities:
    - Summarize research findings
    - Organize collected information
    - Generate structured reports (Markdown or HTML)
    - Format citations and references
    - Ensure report coherence and readability
    """

    def __init__(self, llm: BaseLLM):
        """
        Initialize the Rapporteur.

        Args:
            llm: Language model instance for report generation
        """
        self.llm = llm
        self.prompt_loader = PromptLoader()

    def _safe_section(self, state: ResearchState, section: str, generator, fallback=None):
        """Run one report-section generator, degrading instead of aborting.

        A failed LLM call yields the fallback (or a visible placeholder) and a
        degraded event, so one dead section can no longer cost the whole report.
        """
        try:
            return generator()
        except Exception as exc:
            record_degraded_event(
                state.get('trace'),
                state,
                node="rapporteur",
                where=f"rapporteur_{section}",
                error=str(exc),
            )
            if callable(fallback):
                return fallback()
            return SECTION_FAILURE_PLACEHOLDER

    def generate_report(self, state: ResearchState) -> ResearchState:
        """
        Generate a comprehensive research report.

        Args:
            state: Current research state with all research results

        Returns:
            Updated state with final report
        """
        query = state['query']
        plan = state.get('research_plan', {})
        results = state.get('research_results', [])
        evidence_items = state.get('evidence_items') or build_evidence_from_results(results)
        state['evidence_items'] = evidence_items
        output_format = state.get('output_format', 'markdown')

        # Summarize findings
        summary = self._safe_section(
            state,
            "summary",
            lambda: self._summarize_findings(query, results, evidence_items),
        )

        # Organize information
        organized_info = self._safe_section(
            state,
            "organize",
            lambda: self._organize_information(summary, results),
            fallback=lambda: {
                'themes': [
                    {
                        'name': '核心发现',
                        'key_points': [summary[:500]],
                    }
                ]
            },
        )

        # The LLM is instructed to cite inline; the keyword-overlap heuristic
        # only backfills key points the model left uncited.
        citation_stats = self._apply_bullet_citations(organized_info, evidence_items)

        # Markdown and HTML both embed analysis + conclusion; precompute them
        # under section guards so a single failed call degrades to a
        # placeholder instead of killing the report.
        analysis = None
        conclusion = None
        if output_format != 'json':
            analysis = self._safe_section(
                state,
                "analysis",
                lambda: self._generate_synthesized_analysis(
                    query,
                    summary,
                    organized_info,
                    results,
                    evidence_items=evidence_items,
                ),
            )
            conclusion = self._safe_section(
                state,
                "conclusion",
                lambda: self._generate_conclusion(query, summary),
            )

        # Generate report based on format
        if output_format == 'html':
            report = self._safe_section(
                state,
                "html_render",
                lambda: self._generate_html_report(
                    query=query,
                    plan=plan,
                    summary=summary,
                    organized_info=organized_info,
                    results=results,
                    evidence_items=evidence_items,
                    analysis=analysis,
                    conclusion=conclusion,
                ),
                # If HTML rendering fails, still deliver the content as Markdown.
                fallback=lambda: self._generate_markdown_report(
                    query=query,
                    plan=plan,
                    summary=summary,
                    organized_info=organized_info,
                    results=results,
                    evidence_items=evidence_items,
                    analysis=analysis,
                    conclusion=conclusion,
                ),
            )
        elif output_format == 'json':
            report = self._generate_json_report(
                query=query,
                plan=plan,
                summary=summary,
                organized_info=organized_info,
                results=results,
                evidence_items=evidence_items,
            )
        else:
            # Default to markdown
            report = self._generate_markdown_report(
                query=query,
                plan=plan,
                summary=summary,
                organized_info=organized_info,
                results=results,
                evidence_items=evidence_items,
                analysis=analysis,
                conclusion=conclusion,
            )

        # Fabricated evidence ids must never reach the delivered report.
        report, validity_stats = validate_citations(report, evidence_items)

        degraded_events = state.get('degraded_events') or []
        metrics = calculate_evidence_metrics(results, evidence_items, report)
        metrics.update(
            {
                "generation_citation_validity_rate": validity_stats["citation_validity_rate"],
                "generation_invalid_citation_count": (
                    validity_stats["total_citation_mentions"]
                    - validity_stats["valid_citation_mentions"]
                ),
                "generation_invalid_citation_ids": validity_stats["invalid_citation_ids"],
                "degraded_event_count": len(degraded_events),
                "degraded": bool(degraded_events),
                **citation_stats,
            }
        )

        # Update state
        state['final_report'] = report
        state['report_metrics'] = metrics
        state['current_step'] = 'completed'
        record_report_summary(
            state.get('trace'),
            report_format=output_format,
            source_count=len(results),
            evidence_count=len(evidence_items),
            citation_count=metrics.get('citation_count', 0),
        )
        if state.get('trace'):
            state['trace'].setdefault('metrics', {}).update(metrics)

        return state

    def _summarize_findings(
        self,
        query: str,
        results: List[Dict],
        evidence_items: List[Dict] | None = None
    ) -> str:
        """
        Summarize all research findings.

        Args:
            query: Research query
            results: List of research results

        Returns:
            Summary of findings
        """
        if evidence_items is None:
            evidence_items = build_evidence_from_results(results)
        content_text = format_evidence_for_prompt(evidence_items, limit=30)

        prompt = self.prompt_loader.load(
            'rapporteur_summarize',
            query=query,
            research_findings=content_text
        )

        summary = self.llm.generate(prompt, temperature=0.5, max_tokens=2000)
        return summary

    def _organize_information(self, summary: str, results: List[Dict]) -> Dict:
        """
        Organize information into structured sections.

        Args:
            summary: Research summary
            results: List of research results

        Returns:
            Organized information structure
        """
        # Extract key themes using LLM
        prompt = self.prompt_loader.load(
            'rapporteur_organize_info',
            summary=summary
        )

        response = self.llm.generate(prompt, temperature=0.5)

        # Try to parse JSON
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                organized = json.loads(json_str)
                return organized
        except json.JSONDecodeError:
            pass

        # Fallback structure
        return {
            'themes': [
                {
                    'name': '核心发现',
                    'key_points': [summary[:500]]
                }
            ]
        }

    def _apply_bullet_citations(
        self,
        organized_info: Dict,
        evidence_items: List[Dict],
    ) -> Dict[str, int]:
        """Ensure every key point carries a citation, tracking how it got one."""
        llm_cited = 0
        fallback = 0
        for theme in organized_info.get('themes', []):
            points = theme.get('key_points', [])
            for index, point in enumerate(points):
                point = str(point)
                if re.search(r"\[E\d+\]", point):
                    llm_cited += 1
                    continue
                cited = append_citations(point, evidence_items)
                if cited != point:
                    fallback += 1
                points[index] = cited
        return {
            "llm_cited_bullet_count": llm_cited,
            "heuristic_citation_fallback_count": fallback,
        }

    def _generate_markdown_report(
        self,
        query: str,
        plan: Dict,
        summary: str,
        organized_info: Dict,
        results: List[Dict],
        evidence_items: List[Dict] | None = None,
        analysis: str | None = None,
        conclusion: str | None = None,
    ) -> str:
        """
        Generate a structured Markdown report.

        Args:
            query: Research query
            plan: Research plan
            summary: Research summary
            organized_info: Organized information
            results: Research results
            analysis: Precomputed synthesized analysis (generated when None)
            conclusion: Precomputed conclusion (generated when None)

        Returns:
            Markdown formatted report
        """
        # Build report sections
        sections = []
        evidence_items = evidence_items or build_evidence_from_results(results)
        if analysis is None:
            analysis = self._generate_synthesized_analysis(
                query,
                summary,
                organized_info,
                results,
                evidence_items=evidence_items,
            )
        if conclusion is None:
            conclusion = self._generate_conclusion(query, summary)

        # Title
        sections.append(f"# 研究报告：{query}\n")

        # Metadata
        sections.append(f"**生成时间：** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        sections.append(f"**研究目标：** {plan.get('research_goal', query)}\n")
        sections.append(f"**检索批次数量：** {len(results)}\n")
        sections.append(f"**去重证据数量：** {len(evidence_items)}\n")

        # Executive Summary
        sections.append("\n## 执行摘要\n")
        sections.append(summary)

        # Key Findings (organized by themes)
        sections.append("\n## 核心发现\n")
        for theme in organized_info.get('themes', []):
            sections.append(f"\n### {theme['name']}\n")
            for point in theme.get('key_points', []):
                sections.append(f"- {point}\n")

        # Synthesized Analysis (NEW: generate integrated analysis instead of simple listing)
        sections.append("\n## 深度分析\n")
        sections.append(analysis)

        # Source overview
        sections.append("\n## 来源概览\n")
        sections.append(self._format_source_overview(results, evidence_items))

        # References
        sections.append("\n## 参考资料\n")
        sections.append(self._format_citations(results, evidence_items))

        # Conclusion
        sections.append("\n## 结论\n")
        sections.append(conclusion)

        return '\n'.join(sections)

    def _generate_json_report(
        self,
        query: str,
        plan: Dict,
        summary: str,
        organized_info: Dict,
        results: List[Dict],
        evidence_items: List[Dict] | None = None,
    ) -> str:
        """Generate a machine-readable research report."""
        evidence_items = evidence_items or build_evidence_from_results(results)
        key_findings = []
        for theme in organized_info.get("themes", []):
            for point in theme.get("key_points", []):
                key_findings.append(
                    {
                        "theme": theme.get("name"),
                        "claim": point,
                    }
                )

        payload = {
            "query": query,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "research_goal": plan.get("research_goal", query),
            "sections": ["执行摘要", "核心发现", "深度分析", "来源概览", "参考资料", "结论"],
            "summary": summary,
            "key_findings": key_findings,
            "sources": [
                {
                    "evidence_id": item.get("evidence_id"),
                    "source": item.get("source"),
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "domain": item.get("domain"),
                    "query": item.get("query"),
                    "published_date": item.get("published_date"),
                    "relevance_score": item.get("relevance_score"),
                }
                for item in evidence_items
            ],
            "references": [
                {
                    "citation": f"[{item.get('evidence_id')}]",
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "source": item.get("source"),
                }
                for item in evidence_items
            ],
            "raw_batch_count": len(results),
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)

    def _format_detailed_results(self, results: List[Dict]) -> str:
        """
        Format detailed results section.

        Args:
            results: Research results

        Returns:
            Formatted results string
        """
        formatted = []
        result_num = 1

        for result in results:
            source = result.get('source', 'Unknown')
            query = result.get('query', 'N/A')

            formatted.append(f"\n### Source: {source.capitalize()}")
            formatted.append(f"**Query:** {query}\n")

            for item in result.get('results', [])[:5]:  # Top 5 per source
                title = item.get('title', 'No title')
                snippet = item.get('snippet', 'No description')
                url = item.get('url', '')

                formatted.append(f"{result_num}. **{title}**")
                if url:
                    formatted.append(f"   - URL: {url}")
                formatted.append(f"   - {snippet[:300]}...\n")
                result_num += 1

        return '\n'.join(formatted)

    def _format_source_overview(
        self,
        results: List[Dict],
        evidence_items: List[Dict] | None = None
    ) -> str:
        """
        Format a compact source overview for report traceability.

        Args:
            results: Research results

        Returns:
            Markdown table with source-level counts
        """
        if not results:
            return "未收集到外部来源。"
        evidence_items = evidence_items or build_evidence_from_results(results)

        stats = {}
        for result in results:
            source = result.get('source', 'unknown')
            if source not in stats:
                stats[source] = {
                    'searches': 0,
                    'items': 0,
                    'errors': 0,
                }
            stats[source]['searches'] += 1
            stats[source]['items'] += len(result.get('results', []))
            if result.get('error'):
                stats[source]['errors'] += 1

        lines = [
            "| Source | Searches | Results | Errors |",
            "| --- | ---: | ---: | ---: |",
        ]
        for source, item in sorted(stats.items()):
            lines.append(
                f"| {source} | {item['searches']} | {item['items']} | {item['errors']} |"
            )
        lines.append("")
        lines.append(f"去重后证据项：{len(evidence_items)}")

        return '\n'.join(lines)

    def _format_citations(
        self,
        results: List[Dict],
        evidence_items: List[Dict] | None = None
    ) -> str:
        """
        Format citations and references.

        Args:
            results: Research results

        Returns:
            Formatted citations
        """
        evidence_items = evidence_items or build_evidence_from_results(results)
        citations = []

        for item in evidence_items[:50]:
            evidence_id = item.get('evidence_id', 'E?')
            title = item.get('title', 'Untitled')
            source = str(item.get('source', 'Unknown')).capitalize()
            query = item.get('query') or 'N/A'
            domain = item.get('domain') or 'N/A'
            published = item.get('published_date') or 'N/A'
            url = item.get('url', '')
            if url:
                citations.append(
                    f"- [{evidence_id}] {title} - {source} - {domain} - {published} - "
                    f"query: `{query}` - [{url}]({url})"
                )
            else:
                citations.append(
                    f"- [{evidence_id}] {title} - {source} - {domain} - {published} - "
                    f"query: `{query}`"
                )

        return '\n'.join(citations)

    def _generate_synthesized_analysis(
        self,
        query: str,
        summary: str,
        organized_info: Dict,
        results: List[Dict],
        evidence_items: List[Dict] | None = None
    ) -> str:
        """
        Generate synthesized analysis that integrates all findings.

        Args:
            query: Research query
            summary: Research summary
            organized_info: Organized themes
            results: Research results

        Returns:
            Integrated analysis text
        """
        evidence_items = evidence_items or build_evidence_from_results(results)
        content_text = format_evidence_for_prompt(evidence_items, limit=25)

        prompt = self.prompt_loader.load(
            'rapporteur_synthesized_analysis',
            query=query,
            summary=summary[:1500],
            key_content=content_text
        )

        analysis = self.llm.generate(prompt, temperature=0.6, max_tokens=2000)
        return analysis

    def _generate_conclusion(self, query: str, summary: str) -> str:
        """
        Generate a conclusion section.

        Args:
            query: Research query
            summary: Research summary

        Returns:
            Conclusion text
        """
        prompt = self.prompt_loader.load(
            'rapporteur_conclusion',
            query=query,
            summary=summary[:1000]
        )

        conclusion = self.llm.generate(prompt, temperature=0.5, max_tokens=800)
        return conclusion

    def _generate_html_report(
        self,
        query: str,
        plan: Dict,
        summary: str,
        organized_info: Dict,
        results: List[Dict],
        evidence_items: List[Dict] | None = None,
        analysis: str | None = None,
        conclusion: str | None = None,
    ) -> str:
        """
        Generate a structured HTML report.

        Args:
            query: Research query
            plan: Research plan
            summary: Research summary
            organized_info: Organized information
            results: Research results
            analysis: Precomputed synthesized analysis (generated when None)
            conclusion: Precomputed conclusion (generated when None)

        Returns:
            HTML formatted report
        """
        evidence_items = evidence_items or build_evidence_from_results(results)
        if analysis is None:
            analysis = self._generate_synthesized_analysis(
                query,
                summary,
                organized_info,
                results,
                evidence_items=evidence_items,
            )
        if conclusion is None:
            conclusion = self._generate_conclusion(query, summary)

        # Format themes as HTML-friendly text
        themes_text = ""
        for theme in organized_info.get('themes', []):
            themes_text += f"<h3>{theme['name']}</h3>\n<ul>\n"
            for point in theme.get('key_points', []):
                themes_text += f"<li>{point}</li>\n"
            themes_text += "</ul>\n"

        # Format citations
        citations = self._format_citations(results, evidence_items)

        # Generate HTML using LLM
        prompt = self.prompt_loader.load(
            'rapporteur_generate_html',
            query=query,
            research_goal=plan.get('research_goal', query),
            summary=summary,
            themes=themes_text,
            analysis=analysis,
            citations=citations,
            conclusion=conclusion
        )

        html_report = self.llm.generate(prompt, temperature=0.3, max_tokens=4000)

        # Clean up the HTML (remove markdown code blocks if LLM added them)
        if '```html' in html_report:
            html_report = html_report.split('```html')[1].split('```')[0].strip()
        elif '```' in html_report:
            html_report = html_report.split('```')[1].split('```')[0].strip()

        return html_report

    def save_report(self, report: str, filepath: str) -> bool:
        """
        Save report to file.

        Args:
            report: Report content
            filepath: Path to save the report

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report)
            return True
        except Exception as e:
            print(f"Error saving report: {e}")
            return False

    def __repr__(self) -> str:
        """String representation."""
        return f"Rapporteur(llm={self.llm})"
