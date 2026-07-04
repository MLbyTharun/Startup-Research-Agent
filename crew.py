# crew.py
# Core of the project — defines the 3-agent research crew

import os
import litellm
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
from tools.search_tools import CompanyIntelligenceTool, get_news_search_tool, get_search_tool

load_dotenv()

# Disable LiteLLM caching — required for non-Anthropic models
os.environ["LITELLM_CACHE"] = "False"
os.environ["LITELLM_DROP_PARAMS"] = "True"
litellm.drop_params = True

# LLM
# Swap model/api_key here to change the LLM for all agents
# To use different LLMs for different agents just define the LLM as like below and change the LLM parameter to new llm
# Each agent can also have its own LLM if needed

llm = LLM(
    model="nvidia_nim/z-ai/glm-5.2",
    api_key=os.getenv("NVIDIA_NIM_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1",
    temperature=0.3,
    max_tokens=4096,
)


@CrewBase
class StartupResearchCrew:
    """
    3-agent crew that researches a startup and produces a structured brief.

    Flow:
      researcher → analyst → writer
      Each agent receives previous agents' outputs via context=[]
    """

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # AGENTS 
    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],
            llm=llm,
            tools=[
                CompanyIntelligenceTool(),  # deep company research (crawl + search)
                get_news_search_tool(),     # recent news specifically
            ],
            verbose=True,
            max_iter=5,
            memory=False,
        )

    @agent
    def analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["analyst"],
            llm=llm,
            tools=[CompanyIntelligenceTool(),
                   get_search_tool(max_results=3)],  # targeted competitor searches
            verbose=True,
            max_iter=3,
            memory=False,
        )

    @agent
    def writer(self) -> Agent:
        return Agent(
            config=self.agents_config["writer"],
            llm=llm,
            tools=[],       # writer works only from context — no search needed
            verbose=True,
        )

    # TASKS 
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],
            agent=self.researcher(),
        )

    @task
    def analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["analysis_task"],
            agent=self.analyst(),
            context=[self.research_task()],
        )

    @task
    def writing_task(self) -> Task:
        return Task(
            config=self.tasks_config["writing_task"],
            agent=self.writer(),
            context=[self.research_task(), self.analysis_task()],
            output_file="output/brief.md",
        )

    # CREW

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            memory=False,
            max_rpm=5,
        )


def run_research(company: str) -> dict:
    """
    Main entry point — takes a company name, returns the brief and token usage.
    Called by both CLI and Streamlit UI.
    """
    os.makedirs("output", exist_ok=True)

    result = StartupResearchCrew().crew().kickoff(inputs={"company": company})

    return {
        "company": company,
        "brief": result.tasks_output[1].raw,   # writing_task output
        "token_usage": result.token_usage,
    }

