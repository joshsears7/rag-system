"""
Web Search Fallback — extend CRAG beyond local documents.

When local document retrieval returns low-confidence results (CRAG score < threshold),
fall back to live web search via Tavily. Merge local + web results before generation.

This transforms the system from "only knows what you ingested" to "knows everything,
prefers your docs." Production pattern used by Perplexity, You.com, and enterprise
RAG systems where documents may not cover every question.

Supported providers:
  - Tavily (recommended): domain-filtered, RAG-optimized, free tier available
  - DuckDuckGo: no API key required, basic quality

Setup:
  pip install tavily-python
  TAVILY_API_KEY=tvly-... in .env
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class WebSearchResult:
    """A single result from web search, compatible with RetrievalResult interface."""
    title: str
    url: str
    content: str
    score: float = 1.0
    published_date: str = ""

    @property
    def chunk_text(self) -> str:
        """Format as a retrievable chunk for the RAG pipeline."""
        date_str = f" ({self.published_date})" if self.published_date else ""
        return f"[Web: {self.title}{date_str}]\nSource: {self.url}\n\n{self.content}"

    @property
    def source(self) -> str:
        return self.url


def tavily_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
) -> list[WebSearchResult]:
    """
    Search the web via Tavily and return formatted results.

    Tavily is optimized for RAG: it strips HTML, ranks by relevance, and
    returns clean text passages rather than raw HTML. The 'basic' depth
    is fast and cheap; 'advanced' does deeper extraction.

    Args:
        query: search query
        max_results: number of results to return (1-10)
        search_depth: "basic" (fast, 1 credit) or "advanced" (thorough, 2 credits)
        include_domains: restrict to these domains
        exclude_domains: exclude these domains

    Returns:
        List of WebSearchResult objects
    """
    try:
        from tavily import TavilyClient
        from config import settings
    except ImportError:
        logger.warning("tavily-python not installed. pip install tavily-python")
        return []

    api_key = getattr(settings, "tavily_api_key", "")
    if not api_key:
        logger.warning("TAVILY_API_KEY not set. Web search unavailable.")
        return []

    try:
        client = TavilyClient(api_key=api_key)
        kwargs: dict = {
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
        }
        if include_domains:
            kwargs["include_domains"] = include_domains
        if exclude_domains:
            kwargs["exclude_domains"] = exclude_domains

        response = client.search(**kwargs)
        results = []
        for r in response.get("results", []):
            results.append(WebSearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                content=r.get("content", ""),
                score=float(r.get("score", 1.0)),
                published_date=r.get("published_date", ""),
            ))

        logger.info("Tavily web search: %d results for '%s'", len(results), query[:60])
        return results

    except Exception as e:
        logger.warning("Tavily search failed: %s", e)
        return []


def duckduckgo_search(
    query: str,
    max_results: int = 5,
) -> list[WebSearchResult]:
    """
    Fallback web search using DuckDuckGo (no API key required).

    Lower quality than Tavily but works without credentials.
    Rate-limited; not for production high-throughput use.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.warning("duckduckgo-search not installed. pip install duckduckgo-search")
        return []

    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
        results = []
        for r in raw:
            results.append(WebSearchResult(
                title=r.get("title", ""),
                url=r.get("href", ""),
                content=r.get("body", ""),
                score=1.0,
            ))
        logger.info("DuckDuckGo: %d results for '%s'", len(results), query[:60])
        return results
    except Exception as e:
        logger.warning("DuckDuckGo search failed: %s", e)
        return []


def web_search(
    query: str,
    max_results: int = 5,
    provider: str = "tavily",
) -> list[WebSearchResult]:
    """
    Unified web search interface. Tries Tavily first, falls back to DuckDuckGo.

    Args:
        query: search query
        max_results: number of results
        provider: "tavily" | "duckduckgo" | "auto"
    """
    if provider == "tavily" or provider == "auto":
        results = tavily_search(query, max_results)
        if results:
            return results
        if provider == "tavily":
            return []

    # Fallback to DuckDuckGo
    return duckduckgo_search(query, max_results)


def web_results_to_retrieval(
    web_results: list[WebSearchResult],
) -> list:
    """
    Convert WebSearchResult objects to RetrievalResult-compatible objects.

    Returns lightweight dicts that carry the same fields as RetrievalResult
    but also include the web URL for citation purposes.
    """
    from models import RetrievalResult
    results = []
    for i, r in enumerate(web_results):
        results.append(RetrievalResult(
            chunk_text=r.chunk_text,
            source=r.url,
            similarity_score=min(1.0, r.score),
            chunk_index=i,
            page_number=None,
            section_title=r.title,
            metadata={"web_result": True, "url": r.url, "title": r.title},
        ))
    return results
