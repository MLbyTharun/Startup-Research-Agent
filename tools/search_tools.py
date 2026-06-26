# tools/search_tools.py
# Custom tools that agents can use
# Tavily is the best search tool for agents — designed specifically for LLMs
from crewai_tools import TavilySearchTool
 
from crewai.tools import BaseTool
from pydantic import Field
import os


def get_search_tool(max_results: int = 5):
    """
    Tavily search tool — designed for LLM agents.
    Much better than DuckDuckGo for agentic use:
    - Returns clean, LLM-friendly results
    - Filters out ads and noise
    - Can search recent news specifically
    """
    return TavilySearchTool(
        max_results=max_results,
        search_depth="advanced",   # deeper search, costs 2 credits vs 1
        include_answer=True,       # Tavily summarizes the answer too
        include_raw_content=False, # keep it clean
        include_images=False,
    )


def get_news_search_tool():
    """Separate tool specifically for recent news"""
    return TavilySearchTool(
        max_results=3,
        search_depth="basic",
        topic="news",              # news-specific search
        days=180,                  # last 6 months
        include_answer=True,
    )