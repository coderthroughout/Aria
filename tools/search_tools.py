from __future__ import annotations
import json
import asyncio
from langchain_core.tools import tool
from config import settings


def _client():
    from tavily import TavilyClient
    return TavilyClient(api_key=settings.tavily_api_key)


@tool
async def search_web(query: str, max_results: int = 5) -> str:
    """
    Search the web for current information. Use for market research, news,
    competitive intelligence, financial analysis, and any real-world facts.
    Returns a direct answer and supporting sources with snippets.
    """
    def _run() -> dict:
        results = _client().search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            include_answer=True,
        )
        return {
            "query": query,
            "answer": results.get("answer", ""),
            "sources": [
                {
                    "title": r.get("title"),
                    "url": r.get("url"),
                    "snippet": (r.get("content") or "")[:500],
                    "relevance_score": r.get("score"),
                }
                for r in results.get("results", [])
            ],
        }

    try:
        data = await asyncio.to_thread(_run)
    except Exception as e:
        data = {"error": str(e), "query": query}

    return json.dumps(data, indent=2)


@tool
async def search_company_news(company_name: str) -> str:
    """
    Search for recent news, press releases, and developments for a specific company.
    Returns the most recent articles with publication dates.
    Useful for: recent events, acquisitions, leadership changes, regulatory news,
    earnings surprises, product launches, and market sentiment signals.
    """
    def _run() -> dict:
        results = _client().search(
            query=f"{company_name} recent news developments announcements",
            max_results=8,
            search_depth="advanced",
            topic="news",
        )
        return {
            "company": company_name,
            "articles": [
                {
                    "title": r.get("title"),
                    "url": r.get("url"),
                    "snippet": (r.get("content") or "")[:400],
                    "published_date": r.get("published_date"),
                }
                for r in results.get("results", [])
            ],
        }

    try:
        data = await asyncio.to_thread(_run)
    except Exception as e:
        data = {"error": str(e), "company": company_name}

    return json.dumps(data, indent=2)


@tool
async def verify_claim(claim: str) -> str:
    """
    Verify a specific factual claim by searching for corroborating or contradicting sources.
    Use this to fact-check citations, statistics, and assertions before including them
    in the final analysis. Returns evidence and source URLs.
    """
    def _run() -> dict:
        results = _client().search(
            query=f"fact check: {claim}",
            max_results=5,
            search_depth="advanced",
        )
        return {
            "claim": claim,
            "verdict_summary": results.get("answer", "Could not determine — insufficient sources found"),
            "evidence_sources": [
                {
                    "title": r.get("title"),
                    "url": r.get("url"),
                    "snippet": (r.get("content") or "")[:400],
                }
                for r in results.get("results", [])
            ],
        }

    try:
        data = await asyncio.to_thread(_run)
    except Exception as e:
        data = {"error": str(e), "claim": claim}

    return json.dumps(data, indent=2)
