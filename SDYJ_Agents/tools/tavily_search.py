"""
Tavily Search Tool

This module provides web search functionality using Tavily API.
"""

from typing import List, Dict, Optional
from tavily import TavilyClient
from datetime import datetime


class TavilySearch:
    """
    Tavily search tool for web information retrieval.
    """

    def __init__(self, api_key: str):
        """
        Initialize Tavily search.

        Args:
            api_key: Tavily API key
        """
        self.client = TavilyClient(api_key=api_key)

    def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "advanced",
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None
    ) -> Dict:
        """
        Perform a web search using Tavily.

        Args:
            query: Search query
            max_results: Maximum number of results to return
            search_depth: Search depth ("basic" or "advanced")
            include_domains: List of domains to include
            exclude_domains: List of domains to exclude

        Returns:
            Dictionary containing search results
        """
        try:
            # Perform search
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_domains=include_domains,
                exclude_domains=exclude_domains
            )

            # Format results
            results = []
            for item in response.get('results', []):
                results.append({
                    'title': item.get('title', ''),
                    'url': item.get('url', ''),
                    'snippet': item.get('content', ''),
                    'relevance_score': item.get('score', 0.0),
                    'metadata': {
                        'published_date': item.get('published_date'),
                        'raw_content': item.get('raw_content')
                    }
                })

            return {
                'query': query,
                'source': 'tavily',
                'results': results,
                'timestamp': datetime.now().isoformat(),
                'total_results': len(results)
            }

        except Exception as e:
            return {
                'query': query,
                'source': 'tavily',
                'results': [],
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }

    def get_search_context(
        self,
        query: str,
        max_results: int = 5
    ) -> str:
        """
        Get search context as formatted text.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            Formatted search context string
        """
        try:
            context = self.client.get_search_context(
                query=query,
                max_results=max_results
            )
            return context
        except Exception as e:
            return f"Error retrieving search context: {str(e)}"
