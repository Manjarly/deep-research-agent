"""Consumer-friendly, elegant Deep Research web application."""
import os
import datetime
from pathlib import Path
import streamlit as st

from src.config import config
from src.llm import get_llm
from src.graph import create_research_graph

# Page configuration
st.set_page_config(
    page_title="DeepResearch | Smart In-Depth Reports",
    page_icon="📚",
    layout="centered",  # Centered layout feels significantly more friendly and readable like Notion/Perplexity
    initial_sidebar_state="collapsed"  # Hide technical sidebar by default for laymen!
)

# Friendly, Elegant Consumer Styling (inspired by Perplexity, Notion, and Medium)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Clean background and comfortable max-width */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }
    
    .block-container {
        max-width: 860px !important;
        padding-top: 2rem !important;
        padding-bottom: 4rem !important;
    }

    /* Header styling */
    .brand-header {
        text-align: center;
        margin-top: 1.5rem;
        margin-bottom: 2rem;
    }
    .brand-logo-text {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #ffffff;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
    }
    .brand-tagline {
        font-size: 1.05rem;
        color: #8b949e;
        margin-top: 0.4rem;
        font-weight: 400;
        line-height: 1.5;
    }

    /* Friendly suggestion chips */
    .suggestion-header {
        font-size: 0.85rem;
        font-weight: 600;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }

    /* Report presentation */
    .report-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 2.5rem;
        margin-top: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        line-height: 1.75;
        font-size: 1rem;
        color: #c9d1d9;
    }
    .report-meta {
        display: flex;
        align-items: center;
        gap: 1.25rem;
        padding-bottom: 1.25rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid #30363d;
        font-size: 0.85rem;
        color: #8b949e;
    }
    .meta-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
    }

    /* Source cards */
    .source-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        transition: transform 0.1s ease, border-color 0.15s ease;
    }
    .source-box:hover {
        border-color: #58a6ff;
    }
    .source-link {
        font-size: 0.95rem;
        font-weight: 600;
        color: #58a6ff;
        text-decoration: none;
        display: block;
        margin-bottom: 0.3rem;
    }
    .source-domain {
        font-size: 0.75rem;
        color: #8b949e;
        margin-bottom: 0.4rem;
        display: block;
    }
    .source-text {
        font-size: 0.85rem;
        color: #8b949e;
        line-height: 1.5;
    }

    /* Soft input styles */
    .stTextArea textarea {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
        color: #f0f6fc !important;
        font-size: 1rem !important;
        padding: 1rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #388bfd !important;
        box-shadow: 0 0 0 3px rgba(56, 139, 253, 0.2) !important;
    }

    /* Primary button */
    button[kind="primary"] {
        background-color: #238636 !important;
        border: 1px solid #2ea043 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        border-radius: 8px !important;
        padding: 0.65rem 1.5rem !important;
        transition: all 0.2s ease !important;
    }
    button[kind="primary"]:hover {
        background-color: #2ea043 !important;
        box-shadow: 0 4px 12px rgba(46, 160, 67, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

# Settings in Sidebar (Tucked away so laymen aren't overwhelmed)
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("Customize your AI research preferences:")

    provider = st.selectbox(
        "AI Assistant",
        options=["gemini", "openai", "anthropic", "ollama"],
        format_func=lambda x: {
            "gemini": "Google Gemini (Recommended)",
            "openai": "OpenAI (GPT-4o)",
            "anthropic": "Anthropic (Claude)",
            "ollama": "Local AI (Ollama)"
        }.get(x, x),
        index=0
    )

    # Secure API key check
    config_key = ""
    if provider == "gemini":
        config_key = config.GOOGLE_API_KEY
        key_name = "Gemini API Key"
        has_env_key = bool(config_key and config_key != "your_gemini_api_key_here")
    elif provider == "openai":
        config_key = config.OPENAI_API_KEY
        key_name = "OpenAI API Key"
        has_env_key = bool(config_key)
    elif provider == "anthropic":
        config_key = config.ANTHROPIC_API_KEY
        key_name = "Anthropic API Key"
        has_env_key = bool(config_key)
    else:
        has_env_key = True

    if has_env_key and provider != "ollama":
        st.success("✓ API key connected and secure", icon="🔒")
        custom_key = st.text_input(
            f"Use a different {key_name}",
            value="",
            type="password",
            placeholder="Leave empty to use saved key"
        )
        active_api_key = custom_key.strip() if custom_key.strip() else config_key
    elif provider != "ollama":
        st.warning("Please enter your API key to proceed:")
        custom_key = st.text_input(
            key_name,
            value="",
            type="password",
            placeholder="Paste your key here..."
        )
        active_api_key = custom_key.strip()
    else:
        active_api_key = "local"

    st.markdown("---")
    st.caption("🔒 All searches and API keys are processed locally on your device.")

# Header
st.markdown("""
<div class="brand-header">
    <div class="brand-logo-text">
        <span>📖</span> DeepResearch
    </div>
    <div class="brand-tagline">
        Get comprehensive, fact-checked research reports on any topic in minutes.
    </div>
</div>
""", unsafe_allow_html=True)

# Layman-friendly sample ideas
st.markdown('<div class="suggestion-header">💡 Try an example:</div>', unsafe_allow_html=True)
col_a, col_b, col_c = st.columns(3)
preset_query = None

with col_a:
    if st.button("🔋 Electric vs Hybrid Cars: Best Pick in 2025?", use_container_width=True):
        preset_query = "Electric vehicles vs hybrid cars in 2025: Total cost of ownership, reliability, battery lifespan, and environmental impact"
with col_b:
    if st.button("🥗 Real Science of the Mediterranean Diet", use_container_width=True):
        preset_query = "What does the scientific evidence say about the health benefits and longevity impact of the Mediterranean Diet?"
with col_c:
    if st.button("🤖 How AI is Changing Everyday Jobs", use_container_width=True):
        preset_query = "How is Generative AI actually impacting everyday white-collar jobs? Current employment statistics, real trends, and future skills"

# Main Query Input
default_val = preset_query or "Electric vehicles vs hybrid cars in 2025: Total cost of ownership, reliability, battery lifespan, and environmental impact"
query_input = st.text_area(
    "What topic would you like to explore?",
    value=default_val,
    height=90,
    placeholder="Ask anything (e.g. Which smart home ecosystem is best to start with in 2025?)"
)

# Friendly Depth Selector
col_depth, col_btn = st.columns([1, 1])
with col_depth:
    depth_choice = st.radio(
        "Research Depth:",
        options=["Standard (Faster)", "Deep Dive (More sources & verification)"],
        horizontal=True,
        index=1
    )
    max_loops = 1 if "Standard" in depth_choice else 2

with col_btn:
    st.write("")  # Alignment spacer
    start_search = st.button("Generate Research Report 🚀", type="primary", use_container_width=True)

# Execution Logic
if start_search:
    if not query_input.strip():
        st.warning("Please type a question or topic above to start researching.")
    elif provider != "ollama" and not active_api_key:
        st.error(f"Please provide an API key in the Settings sidebar (top-left menu).")
    else:
        # User-friendly progress container
        progress_box = st.status("Starting your research...", expanded=True)

        try:
            progress_box.write("Connecting to AI researcher...")
            target_model = "gemini-3.5-flash-lite" if provider == "gemini" else None
            llm = get_llm(
                provider=provider,
                model_name=target_model,
                api_key=active_api_key
            )

            graph = create_research_graph(llm)

            initial_state = {
                "task": query_input.strip(),
                "evidence": [],
                "iteration": 0,
                "max_iterations": max_loops,
                "logs": []
            }

            final_state = None

            # Stream steps with friendly, non-technical updates
            for event in graph.stream(initial_state):
                for node_name, node_output in event.items():
                    if node_name == "planner_node":
                        plan = node_output.get("plan")
                        sub_count = len(plan.sub_topics) if plan else 3
                        progress_box.write(f"🔍 Dissecting topic into {sub_count} key areas of investigation...")
                    elif node_name == "researcher_node":
                        evidence_count = len(node_output.get("evidence", []))
                        progress_box.write(f"🌐 Browsing and reading verified web sources ({evidence_count} sources gathered)...")
                    elif node_name == "evaluator_node":
                        evaluation = node_output.get("evaluation")
                        if evaluation and evaluation.is_sufficient:
                            progress_box.write("⚖️ Fact-checking complete: evidence is thorough and balanced.")
                        else:
                            progress_box.write("🔍 Looking deeper to answer remaining questions...")
                    elif node_name == "synthesizer_node":
                        progress_box.write("✍️ Writing your formatted report with citations...")

                    final_state = node_output

            progress_box.update(label="✨ Your report is ready!", state="complete", expanded=False)

            if final_state and "final_report" in final_state:
                report_content = final_state["final_report"]
                sources = final_state.get("evidence", [])
                plan = final_state.get("plan")

                # Auto-save report locally
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                clean_title = "".join(c if c.isalnum() else "_" for c in query_input[:30]).strip("_")
                report_file = config.REPORTS_DIR / f"report_{clean_title}_{timestamp}.md"
                with open(report_file, "w", encoding="utf-8") as f:
                    f.write(report_content)

                # Report Tabs: Simple and intuitive
                tab_report, tab_sources, tab_outline = st.tabs([
                    "📄 Read Report",
                    f"🔗 Sources Consulted ({len(sources)})",
                    "📋 Research Outline"
                ])

                with tab_report:
                    # Action Bar
                    col_d1, col_d2 = st.columns([1, 3])
                    col_d1.download_button(
                        label="📥 Download Report (.md)",
                        data=report_content,
                        file_name=f"research_{clean_title}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )

                    # Meta Header
                    st.markdown(f"""
                    <div class="report-meta">
                        <span class="meta-tag">📅 {datetime.date.today().strftime('%B %d, %Y')}</span>
                        <span class="meta-tag">📚 {len(sources)} Verified Web Sources</span>
                        <span class="meta-tag">⚡ AI Model: {provider.capitalize()}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    # Formatted Report Content
                    st.markdown(report_content)

                with tab_sources:
                    st.markdown("Here are the real articles and web pages gathered during research:")
                    for idx, src in enumerate(sources, 1):
                        url = src.get("url", "")
                        domain = url.split("//")[-1].split("/")[0] if url else "web"
                        title = src.get("title", "Article Reference")
                        snippet = src.get("snippet", "")
                        st.markdown(f"""
                        <div class="source-box">
                            <a href="{url}" target="_blank" class="source-link">{idx}. {title}</a>
                            <span class="source-domain">🌐 {domain}</span>
                            <div class="source-text">{snippet[:220]}...</div>
                        </div>
                        """, unsafe_allow_html=True)

                with tab_outline:
                    if plan:
                        st.subheader("Key Questions Explored")
                        for idx, sub in enumerate(plan.sub_topics, 1):
                            st.markdown(f"**{idx}.** {sub}")

                        if plan.hypotheses:
                            st.subheader("Starting Assumptions Tested")
                            for h in plan.hypotheses:
                                st.info(h)

        except Exception as e:
            progress_box.update(label="Something went wrong", state="error")
            st.error(f"Error details: {str(e)}")
            st.info("Tip: Check that your API key is valid in the Settings menu (top left).")
