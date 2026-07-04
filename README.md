# Startup Research Agent

A multi-agent AI system that researches any startup and generates a structured company brief — automatically. Give it a company name, and within a few minutes it returns a clean, readable report covering the product, team, funding history, competitors, and recent news.

Built as a portfolio project to explore production patterns in agentic AI — specifically how to combine CrewAI and LangGraph together, and how to make agents reliable enough to actually trust their output.

---
**Watch Demo -->[Here](https://github.com/user-attachments/assets/5b79d6b3-fbb2-47ba-8e1a-d2968888e88c)**

![App Screenshot-1](assests/screenshot1.png)   ||
![App Screenshot-2](assests/screenshot2.png)   ||
![App Screenshot-3](assests/screenshot3.png)   ||
![App Screenshot-4](assests/screenshot4.png)   ||
## What It Does

Most startup research is tedious. You open five tabs, skim Crunchbase, search for recent news, try to piece together who the founders are and whether the company is actually growing. This project automates that workflow using three specialized AI agents that each own a distinct part of the research process.

The output is a markdown brief you can read in under three minutes — structured, sourced where possible, and honest about what it couldn't verify.

---

## Architecture

The system is built in two layers.

**Outer layer — LangGraph**
LangGraph acts as the stateful orchestrator. It validates the input, runs the crew, handles retries if something fails, and formats the final output. Think of it as the production wrapper that makes the system resilient.

```
Input → Validate → Run Crew → Format Output → Done
                      ↑              |
                   Retry (x2)  ←  Error
```

**Inner layer — CrewAI**
Inside the LangGraph graph, CrewAI runs three agents sequentially. Each agent gets the previous agent's output as context, so information flows naturally from research to analysis to writing.

```
Researcher → Analyst → Writer
```

| Agent | Role | Tools |
|---|---|---|
| Researcher | Finds company info, funding, team, news | Tavily Company Intelligence + News Search |
| Analyst | Maps competitors, evaluates business model and market | Tavily Web Search |
| Writer | Formats everything into a structured brief | None — works from context |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | CrewAI |
| Orchestration | LangGraph |
| Web Research | Tavily Company Intelligence API |
| LLM | Kimi K2.6 via NVIDIA NIM |
| LLM Routing | LiteLLM (model-agnostic) |
| UI | Streamlit |
| Config | YAML (agents and tasks defined separately from code) |

---

## Project Structure

```
Startup-Research-Agent/
├── config/
│   ├── agents.yaml        # Agent roles, goals, and backstories
│   └── tasks.yaml         # Task descriptions and expected outputs
├── tools/
│   └── search_tools.py    # Tavily tool definitions
├── crew.py                # CrewAI crew — agents, tasks, and execution
├── graph.py               # LangGraph orchestrator — state, flow, retries
├── app.py                 # Streamlit UI
├── .env.example           # Environment variable template
└── requirements.txt
```

The YAML config pattern is intentional — agent behavior (roles, goals, backstories) lives separately from the Python code. This means you can tune how an agent reasons without touching the execution logic.

---

## Setup

**1. Clone and install**
```bash
git clone https://github.com/MLbyTharun/Startup-Research-Agent
cd Startup-Research-Agent
pip install -r requirements.txt
```

**2. Set up environment variables**
```bash
cp .env.example .env
```

Open `.env` and fill in:
```
NVIDIA_NIM_API_KEY=your_key_here     # build.nvidia.com — free tier available

TAVILY_API_KEY=your_key_here         # tavily.com — free tier available

GROQ_API_KEY=your_key_here         # console.groq.com — free tier available
```

**3. Run**
```bash
streamlit run app.py
```


---

## How the Research Works

The Researcher agent uses Tavily's Company Intelligence tool, which does more than a standard web search. It crawls the company's actual website, extracts key pages, runs multiple targeted searches in parallel, deduplicates results, and returns a synthesized summary. This gives significantly better funding and product data compared to a single search query.

The Analyst then takes that research and does a separate pass focused specifically on competitors — searching for alternatives, comparisons, and market positioning. This separation is intentional: the Researcher goes deep on the company itself, the Analyst goes wide on the competitive landscape.

The Writer receives both outputs and formats them into a structured brief following a consistent template: TL;DR, Product, Team, Funding, Market, Competitors, Recent News, Risks.

---

## Design Decisions Worth Knowing

**Why LangGraph AND CrewAI together?**
They solve different problems. CrewAI is excellent at defining agent roles and passing context between them — it handles the "who does what" layer. LangGraph handles the "what happens when things go wrong" layer: retries, state management, input validation, and conditional flow. Using both means the system is readable at the agent level and robust at the orchestration level.

**Why YAML for agent config?**
Separating agent personalities from execution code makes it much easier to tune agent behavior. Changing how the Researcher reasons only requires editing a YAML file, not refactoring Python. It also makes the system easier to hand off or collaborate on.

**Token efficiency**
Early versions of this project consumed over 1 million input tokens per run. After tuning `max_iter` values and with adjusting the Tavily Search tool's parameters ,**token usage dropped by roughly 50% with minimal impact on output quality.**

**Why no evaluator agent?**
An evaluator that uses the same search tools as the researcher doesn't add real value — it would just re-confirm what the researcher already found from the same sources. Instead, the Writer task instructions explicitly tell the agent to mark anything uncertain as `[UNVERIFIED]` rather than filling in gaps. Honest output is more useful than a rubber-stamped confidence score.
 
---

## Limitations

- Financial data (exact funding amounts, valuations) can be inaccurate for companies that don't disclose publicly. The system marks these as `[UNVERIFIED]` when it can't find a reliable source, but it's not foolproof.
- NVIDIA NIM's free tier has rate limits. If you hit a 429 error, wait a few minutes or check your credit balance at `build.nvidia.com`.
- Very new or obscure companies may produce thin research if there isn't much public information available.

---

## Author

**Tharun K** — BS in Data Science, IIT Madras

[LinkedIn](https://www.linkedin.com/in/tharun-k-71a5673a9/) · [GitHub](https://github.com/MLbyTharun)