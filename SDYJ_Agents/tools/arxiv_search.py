"""
arXiv Search Tool

This module provides academic paper search functionality using arXiv API.
"""

from typing import List, Dict, Optional
import arxiv
from datetime import datetime


class ArxivSearch:
    """
    arXiv search tool for academic paper retrieval.
    """

    def __init__(self):
        """
        Initialize arXiv search.
        """
        self.client = arxiv.Client()

    def search(
        self,
        query: str,
        max_results: int = 5,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
        sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending
    ) -> Dict:
        """
        Search for academic papers on arXiv.

        Args:
            query: Search query
            max_results: Maximum number of results to return
            sort_by: Sort criterion (Relevance, LastUpdatedDate, SubmittedDate)
            sort_order: Sort order (Ascending or Descending)

        Returns:
            Dictionary containing search results
        """
        try:
            # Create search
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=sort_by,
                sort_order=sort_order
            )

            # Perform search
            results = []
            for paper in self.client.results(search):
                results.append({
                    'title': paper.title,
                    'url': paper.entry_id,
                    'snippet': paper.summary,
                    'relevance_score': None,  # arXiv doesn't provide relevance scores
                    'metadata': {
                        'authors': [author.name for author in paper.authors],
                        'published': paper.published.isoformat() if paper.published else None,
                        'updated': paper.updated.isoformat() if paper.updated else None,
                        'categories': paper.categories,
                        'primary_category': paper.primary_category,
                        'pdf_url': paper.pdf_url,
                        'doi': paper.doi,
                        'journal_ref': paper.journal_ref,
                        'comment': paper.comment
                    }
                })

            return {
                'query': query,
                'source': 'arxiv',
                'results': results,
                'timestamp': datetime.now().isoformat(),
                'total_results': len(results)
            }

        except Exception as e:
            return {
                'query': query,
                'source': 'arxiv',
                'results': [],
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }

    def get_paper_by_id(self, paper_id: str) -> Optional[Dict]:
        """
        Get a specific paper by its arXiv ID.

        Args:
            paper_id: arXiv paper ID (e.g., "2301.07041")

        Returns:
            Paper details or None if not found
        """
        try:
            search = arxiv.Search(id_list=[paper_id])
            paper = next(self.client.results(search))

            return {
                'title': paper.title,
                'url': paper.entry_id,
                'summary': paper.summary,
                'authors': [author.name for author in paper.authors],
                'published': paper.published.isoformat() if paper.published else None,
                'pdf_url': paper.pdf_url,
                'categories': paper.categories
            }

        except Exception as e:
            return None

    def download_pdf(self, paper_id: str, dirpath: str = "./") -> Optional[str]:
        """
        Download PDF of a paper.

        Args:
            paper_id: arXiv paper ID
            dirpath: Directory to save the PDF

        Returns:
            Path to downloaded PDF or None if failed
        """
        try:
            search = arxiv.Search(id_list=[paper_id])
            paper = next(self.client.results(search))
            filepath = paper.download_pdf(dirpath=dirpath)
            return filepath
        except Exception as e:
            return None
