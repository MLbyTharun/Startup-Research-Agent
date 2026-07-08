
import streamlit as st
import time
from graph import run_research

# PAGE CONFIG

st.set_page_config(
    page_title="Startup Research Agent",
    page_icon="🔍",
    layout="wide",
)

# STYLES

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem; }
    .subtitle { color: #666; font-size: 1.1rem; margin-bottom: 2rem; }
    .agent-step {
        background: #f8f9fa;
        border-left: 3px solid #0066cc;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        border-radius: 0 4px 4px 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown('<div class="main-header">🔍 Startup Research Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by CrewAI + LangGraph + Tavily — Multi-agent research in minutes</div>', unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.header("⚙️ How It Works")
    st.markdown("""
    This app runs **3 AI agents** in sequence:

    1. 🔍 **Researcher** — crawls the company website + searches the web for funding, news, team
    2. 📊 **Analyst** — finds competitors, evaluates business model and market position
    3. ✍️ **Writer** — formats everything into a clean structured brief

    Built with:
    - **CrewAI** — multi-agent orchestration
    - **LangGraph** — stateful graph with retry logic
    - **Tavily** — LLM-optimized web search + company intelligence
    - **NVIDIA NIM** — GLM 5.2 model powering all agents
    """)

    st.divider()
    st.header("💡 Try These")
    examples = ["Zepto", "Razorpay", "CRED", "Sarvam AI", "Perplexity AI", "Linear"]
    for company in examples:
        if st.button(company, key=f"btn_{company}"):
            st.session_state["company_input"] = company

# INPUT

col1, col2 = st.columns([4, 1])
with col1:
    company = st.text_input(
        "Company name",
        value=st.session_state.get("company_input", ""),
        placeholder="e.g. Zepto, Razorpay, Notion...",
        label_visibility="collapsed",
    )
with col2:
    run_button = st.button("🚀 Research", type="primary", use_container_width=True)

# RUN 

if run_button and company:

    with st.spinner(f"Agents are Researching {company}... (this takes 3-5 minutes)"):
        try:
            result = run_research(company)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()


    # RESULTS 

    if result.get("error"):
        st.error(f"❌ {result['error']}")
    else:
        tab1, tab2 = st.tabs(["📋 Brief", "📊 Usage"])

        with tab1:
            st.markdown(result.get("brief", "No brief generated"))
            st.download_button(
                label="⬇️ Download Brief",
                data=result.get("brief", ""),
                file_name=f"{company.lower().replace(' ', '_')}_brief.md",
                mime="text/markdown",
            )

        with tab2:
            usage = result.get("token_usage")
            if usage:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Input Tokens", f"{getattr(usage, 'prompt_tokens', 0):,}")
                with col2:
                    st.metric("Output Tokens", f"{getattr(usage, 'completion_tokens', 0):,}")
                with col3:
                    cost = (getattr(usage, 'prompt_tokens', 0) * 0.00000016) + \
                           (getattr(usage, 'completion_tokens', 0) * 0.000004)
                    st.metric("Est. Cost", f"${cost:.4f}")
            else:
                st.info("Token usage not available")
                
                st.caption("💡 Cost estimate based on Moonshot's official Kimi K2.6 API pricing ($0.66/M input, $3.40/M output). Actual cost on NVIDIA NIM free tier may differ.")
elif run_button and not company:
    st.warning("Please enter a company name first")

# FOOTER 
st.divider()
st.caption("Built with CrewAI + LangGraph + Tavily + NVIDIA NIM ")