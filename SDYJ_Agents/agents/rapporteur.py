"""
Rapporteur Agent

This module implements the Rapporteur agent, which is responsible for
generating the final research report.
"""

import json
from typing import Any, Dict, List
from datetime import datetime
from ..workflow.state import ResearchState
from ..llm.base import BaseLLM
from ..prompts.loader import PromptLoader
from ..utils.evidence import (
    append_citations,
    build_evidence_from_results,
    calculate_evidence_metrics,
    format_evidence_for_prompt,
)
from ..utils.structured_output import generate_json_object
from ..utils.tracing import record_report_summary


ORGANIZED_INFO_SCHEMA = {
    "type": "object",
    "required": ["themes"],
    "properties": {
        "themes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "key_points"],
                "properties": {
                    "name": {"type": "string"},
                    "key_points": {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": True,
            },
        }
    },
    "additionalProperties": True,
}


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

    def generate_report(self, state: ResearchState) -> ResearchState:
        """
        Generate a comprehensive research report.

        When ``state['revision_count']`` is greater than zero this method
        switches to *revise* mode: it reads the previous report and the
        latest verifier critique from state, sends them to the rapporteur
        revise prompt, and replaces ``final_report`` with the revised text.
        That keeps the structure stable, addresses the verifier's hints, and
        avoids burning extra tokens on summarize/organize/synthesis steps
        that already ran on the first pass.

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

        revision_count = state.get('revision_count', 0)
        if revision_count > 0 and state.get('final_report'):
            report = self._revise_report(
                query=query,
                plan=plan,
                previous_report=state['final_report'],
                evidence_items=evidence_items,
                verification_result=state.get('verification_result') or {},
            )
        else:
            # Summarize findings
            summary = self._summarize_findings(query, results, evidence_items)

            # Organize information
            organized_info = self._organize_information(summary, results)

            # Generate report based on format
            if output_format == 'html':
                report = self._generate_html_report(
                    query=query,
                    plan=plan,
                    summary=summary,
                    organized_info=organized_info,
                    results=results,
                    evidence_items=evidence_items
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
                    evidence_items=evidence_items
                )

        metrics = calculate_evidence_metrics(results, evidence_items, report)

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
            state['trace']['metrics']['revision_count'] = revision_count

        return state

    def _revise_report(
        self,
        query: str,
        plan: Dict,
        previous_report: str,
        evidence_items: List[Dict],
        verification_result: Dict[str, Any],
    ) -> str:
        """Apply verifier hints to the previous report and return the revision.

        We deliberately do not regenerate the summary/organize/synthesis
        chain — those would discard the parts of the prior report that
        already passed the critic. Re-using the previous text means
        revisions are predictable, cheaper, and easier to diff in trace.
        """
        prompt = self.prompt_loader.load(
            'rapporteur_revise',
            query=query,
            research_goal=plan.get('research_goal', query),
            previous_report=previous_report,
            weakest_dimension=verification_result.get('weakest_dimension', 'unknown'),
            critic_summary=verification_result.get('summary', ''),
            revision_hints=verification_result.get('revision_hints') or [],
            evidence=format_evidence_for_prompt(evidence_items, limit=30),
        )
        # Slightly higher max_tokens so the model can rewrite long sections
        # without being forced to truncate.
        revised = self.llm.generate(prompt, temperature=0.3, max_tokens=4000)

        # If the model wraps the report in fenced code blocks, strip them.
        if revised.startswith('```'):
            revised = revised.strip('`')
            for marker in ("html", "json", "markdown", "md"):
                if revised.lower().startswith(marker):
                    revised = revised[len(marker):].lstrip("\n")
                    break
        return revised.strip()

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

        try:
            return generate_json_object(
                self.llm,
                prompt,
                schema=ORGANIZED_INFO_SCHEMA,
                temperature=0.5,
            )
        except (ValueError, TypeError, json.JSONDecodeError):
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

    def _generate_markdown_report(
        self,
        query: str,
        plan: Dict,
        summary: str,
        organized_info: Dict,
        results: List[Dict],
        evidence_items: List[Dict] | None = None
    ) -> str:
        """
        Generate a structured Markdown report.

        Args:
            query: Research query
            plan: Research plan
            summary: Research summary
            organized_info: Organized information
            results: Research results

        Returns:
            Markdown formatted report
        """
        # Build report sections
        sections = []
        evidence_items = evidence_items or build_evidence_from_results(results)

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
                sections.append(f"- {append_citations(point, evidence_items)}\n")

        # Synthesized Analysis (NEW: generate integrated analysis instead of simple listing)
        sections.append("\n## 深度分析\n")
        sections.append(self._generate_synthesized_analysis(
            query,
            summary,
            organized_info,
            results,
            evidence_items=evidence_items,
        ))

        # Source overview
        sections.append("\n## 来源概览\n")
        sections.append(self._format_source_overview(results, evidence_items))

        # References
        sections.append("\n## 参考资料\n")
        sections.append(self._format_citations(results, evidence_items))

        # Conclusion
        sections.append("\n## 结论\n")
        sections.append(self._generate_conclusion(query, summary))

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
                        "claim": append_citations(point, evidence_items),
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
        return append_citations(analysis, evidence_items, max_ids=3)

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
        evidence_items: List[Dict] | None = None
    ) -> str:
        """
        Generate a structured HTML report.

        Args:
            query: Research query
            plan: Research plan
            summary: Research summary
            organized_info: Organized information
            results: Research results

        Returns:
            HTML formatted report
        """
        # Generate analysis and conclusion
        evidence_items = evidence_items or build_evidence_from_results(results)
        analysis = self._generate_synthesized_analysis(
            query,
            summary,
            organized_info,
            results,
            evidence_items=evidence_items,
        )
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
