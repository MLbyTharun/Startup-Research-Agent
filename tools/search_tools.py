# tools/search_tools.py
# Custom tools that agents can use
# Tavily is the best search tool for agents — designed specifically for LLMs
import os
import asyncio
from urllib.parse import urlparse
from crewai.tools import BaseTool
from crewai_tools import TavilySearchTool
from tavily_agent_toolkit import (
    crawl_and_summarize,
    search_dedup,
    ModelConfig,
    ModelObject
)
from dotenv import load_dotenv

load_dotenv()

# Model config using Groq gpt-oss-120b
model_config = ModelConfig(
    model=ModelObject(model="groq:openai/gpt-oss-120b")
)

# Company Intelligence Tool (the new addition)

class CompanyIntelligenceTool(BaseTool):
    name: str = "Company Intelligence"
    description: str = (
        "Deep company research tool. Crawls the company website, "
        "extracts key pages, and searches the web for funding, news, "
        "competitors and business model. Use this first for any company research."
    )

    def _run(self, company: str) -> str:
        return asyncio.run(self._async_run(company))

    async def _resolve_website(self, company: str) -> str:
        """Find the company's real website via search instead of guessing the domain."""
        api_key = os.getenv("TAVILY_API_KEY")
        try:
            result = await search_dedup(
                api_key=api_key,
                queries=[f"{company} official website"],
                search_depth="basic",
                max_results=5,
                topic="general",
            )
        except Exception:
            result = {}

        skip_domains = (
            "linkedin.com", "twitter.com", "x.com", "facebook.com",
            "instagram.com", "crunchbase.com", "wikipedia.org", "youtube.com",
        )
        for r in result.get("results", []):
            url = r.get("url") or r.get("content_url") or ""
            if not url.startswith("http"):
                continue
            domain = urlparse(url).netloc.lower().removeprefix("www.")
            if not any(domain.endswith(s) for s in skip_domains):
                return url

        # Fallback — guess the domain (works for many startups, not all)
        return f"https://www.{company.lower().replace(' ', '')}.com"

    async def _async_run(self, company: str) -> str:
        website_url = await self._resolve_website(company)
        api_key = os.getenv("TAVILY_API_KEY")

        # Step 1 — crawl company website (non-fatal if it fails)
        try:
            crawl_result = await crawl_and_summarize(
                url=website_url,
                api_key=api_key,
                model_config=model_config,
                instructions="Extract product info, team, mission, and business model",
                max_depth=4,
                max_breadth=10,
                limit=20,
            )
            website_summary = crawl_result.get("summary", "No website data found")
        except Exception as e:
            website_summary = f"Website crawl failed for {website_url}: {e}"

        # Step 2 — search web for funding, news, competitors
        try:
            search_result = await search_dedup(
                api_key=api_key,
                queries=[
                    f"{company} startup funding investors",
                    f"{company} company news 2025 2026",
                    f"{company} competitors market analysis",
                    f"{company} business model revenue",
                ],
                search_depth="advanced",
                max_results=7,
                topic="general",
            )
            web_research = chr(10).join(
                [r.get("content", "") for r in search_result.get("results", [])]
            )
        except Exception as e:
            web_research = f"Web search failed: {e}"

        # Combining both results
        return f"""
## Website Research:
{website_summary}

## Web Research:
{web_research}
"""


# Regular search tools (keeping it for fallback)


def get_search_tool(max_results: int = 2):
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
