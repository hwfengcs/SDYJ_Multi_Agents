"""
Rapporteur Agent

This module implements the Rapporteur agent, which is responsible for
generating the final research report.
"""

from typing import Dict, List
from datetime import datetime
from ..workflow.state import ResearchState
from ..llm.base import BaseLLM
from ..prompts.loader import PromptLoader


class Rapporteur:
    """
    Rapporteur agent - report generation component.

    Responsibilities:
    - Summarize research findings
    - Organize collected information
    - Generate structured Markdown reports
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

        Args:
            state: Current research state with all research results

        Returns:
            Updated state with final report
        """
        query = state['query']
        plan = state.get('research_plan', {})
        results = state.get('research_results', [])

        # Summarize findings
        summary = self._summarize_findings(query, results)

        # Organize information
        organized_info = self._organize_information(summary, results)

        # Generate markdown report
        report = self._generate_markdown_report(
            query=query,
            plan=plan,
            summary=summary,
            organized_info=organized_info,
            results=results
        )

        # Update state
        state['final_report'] = report
        state['current_step'] = 'completed'

        return state

    def _summarize_findings(self, query: str, results: List[Dict]) -> str:
        """
        Summarize all research findings.

        Args:
            query: Research query
            results: List of research results

        Returns:
            Summary of findings
        """
        # Compile all result snippets
        all_content = []
        for result in results:
            for item in result.get('results', []):
                all_content.append(f"- {item.get('title', 'No title')}: {item.get('snippet', '')[:200]}")

        content_text = '\n'.join(all_content[:30])  # Limit to 30 items

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
        import json
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

    def _generate_markdown_report(
        self,
        query: str,
        plan: Dict,
        summary: str,
        organized_info: Dict,
        results: List[Dict]
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

        # Title
        sections.append(f"# 研究报告：{query}\n")

        # Metadata
        sections.append(f"**生成时间：** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        sections.append(f"**研究目标：** {plan.get('research_goal', query)}\n")
        sections.append(f"**信息来源数量：** {len(results)}\n")

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
        sections.append(self._generate_synthesized_analysis(query, summary, organized_info, results))

        # References
        sections.append("\n## 参考资料\n")
        sections.append(self._format_citations(results))

        # Conclusion
        sections.append("\n## 结论\n")
        sections.append(self._generate_conclusion(query, summary))

        return '\n'.join(sections)

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

    def _format_citations(self, results: List[Dict]) -> str:
        """
        Format citations and references.

        Args:
            results: Research results

        Returns:
            Formatted citations
        """
        citations = []
        citation_num = 1

        for result in results:
            for item in result.get('results', []):
                title = item.get('title', 'Untitled')
                url = item.get('url', '')
                source = result.get('source', 'Unknown')

                if url:
                    citations.append(f"{citation_num}. {title} - {source.capitalize()} - [{url}]({url})")
                else:
                    citations.append(f"{citation_num}. {title} - {source.capitalize()}")

                citation_num += 1

        return '\n'.join(citations[:50])  # Limit to 50 citations

    def _generate_synthesized_analysis(
        self,
        query: str,
        summary: str,
        organized_info: Dict,
        results: List[Dict]
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
        # Extract key content from results
        key_content = []
        for result in results[:10]:  # Limit to first 10 results
            for item in result.get('results', [])[:3]:  # Top 3 per result
                key_content.append(f"- {item.get('snippet', '')[:300]}")

        content_text = '\n'.join(key_content)

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
