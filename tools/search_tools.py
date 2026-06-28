# tools/search_tools.py
# Custom tools that agents can use
# Tavily is the best search tool for agents — designed specifically for LLMs
# tools/search_tools.py
import os
import asyncio
from crewai.tools import BaseTool
from crewai_tools import TavilySearchTool
from tavily_agent_toolkit import (
    crawl_and_summarize,
    extract_and_summarize,
    search_dedup,
    ModelConfig,
    ModelObject
)
from dotenv import load_dotenv

load_dotenv

# Model config using Groq gpt-oss-120b
model_config = ModelConfig(
    model=ModelObject(model="groq:openai/gpt-oss-120b")
)

# ── Company Intelligence Tool (the new addition) ──────────────────────────────

class CompanyIntelligenceTool(BaseTool):
    name: str = "Company Intelligence"
    description: str = (
        "Deep company research tool. Crawls the company website, "
        "extracts key pages, and searches the web for funding, news, "
        "competitors and business model. Use this first for any company research."
    )

    def _run(self, company: str) -> str:
        return asyncio.run(self._async_run(company))

    async def _async_run(self, company: str) -> str:
        website_url = f"https://www.{company.lower().replace(' ', '')}.com"

        # Step 1 — crawl company website
        crawl_result = await crawl_and_summarize(
            url=website_url,
            api_key=os.getenv("TAVILY_API_KEY"),
            model_config=model_config,
            instructions="Extract product info, team, mission, and business model",
            max_depth=2,
            max_breadth=10,
            limit=20,
        )

        # Step 2 — search web for funding, news, competitors
        search_result = await search_dedup(
            api_key=os.getenv("TAVILY_API_KEY"),
            queries=[
                f"{company} startup funding investors",
                f"{company} company news 2025 2026",
                f"{company} competitors market analysis",
                f"{company} business model revenue",
            ],
            search_depth="advanced",
            max_results=5,
            topic="general",
        )

        # Combine both results
        return f"""
## Website Research:
{crawl_result.get('summary', 'No website data found')}

## Web Research:
{chr(10).join([r.get('content', '') for r in search_result.get('results', [])])}
"""


# ── Regular search tools (kept as fallback) ───────────────────────────────────


def get_search_tool(max_results: int = 5):
    return TavilySearchTool(
        max_results=max_results,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
        include_images=False,
    )

def get_news_search_tool():
    return TavilySearchTool(
        max_results=3,
        search_depth="basic",
        topic="news",
        days=180,
        include_answer=True,
    )