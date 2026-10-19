"""Synthesizer Agent: Generates a comprehensive, cited, executive-ready research report."""
import logging
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

from ..models import ResearchPlan

logger = logging.getLogger(__name__)

SYNTHESIZER_SYSTEM_PROMPT = """You are a Principal Research Analyst and Technical Writer.
Your task is to synthesize the provided research evidence into an exhaustive, highly structured, authoritative, and engaging Markdown report.

Formatting Guidelines:
1. Title: Give the report an engaging, executive title (# Title).
2. Table of Contents: Quick navigational anchors.
3. Executive Summary: High-level overview, key takeaways, and bottom-line verdict.
4. Deep Dive Analysis: Grouped into logical thematic sections (## and ###) covering background, current paradigms, architecture, performance, and empirical findings.
5. Critical Evaluation & Trade-offs: Contrast alternative approaches, highlighting benefits vs limitations/risks.
6. Future Outlook & Strategic Recommendations: Forward-looking perspective on where the field or technology is heading.
7. Cited Sources & References: Include a dedicated references section listing all consulted sources formatted with clean Markdown links: `[Source Title](URL)`. Ensure you reference sources directly inline using numeric bracket tags like `[1]`, `[2]`.

Write in a clear, professional, analytical tone. Synthesize deeply—do NOT just paste snippets."""


def synthesize_report(
    task: str,
    plan: ResearchPlan,
    evidence: List[Dict[str, Any]],
    llm: BaseChatModel
) -> str:
    """Synthesize all research findings into a publication-ready Markdown report."""
    formatted_sources = []
    for idx, item in enumerate(evidence, start=1):
        title = item.get("title", "Untitled")
        url = item.get("url", "")
        content = item.get("content") or item.get("snippet", "")
        # Include substantial excerpt of content for the LLM
        truncated_content = content[:1500].replace("\n", " ").strip()
        formatted_sources.append(
            f"Source [{idx}]: {title}\nURL: {url}\nContent excerpt: {truncated_content}\n"
        )

    sources_block = "\n---\n".join(formatted_sources) if formatted_sources else "No external web sources available."

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYNTHESIZER_SYSTEM_PROMPT),
        (
            "human",
            "Research Query: {task}\n\n"
            "Key Research Objectives: {sub_topics}\n\n"
            "Primary Hypotheses: {hypotheses}\n\n"
            "Retrieved Research Material ({num_sources} sources):\n\n"
            "{sources_block}\n\n"
            "Please generate the complete, authoritative, and exhaustive research report in Markdown:"
        )
    ])

    chain = prompt | llm
    response = chain.invoke({
        "task": task,
        "sub_topics": ", ".join(plan.sub_topics) if plan else "Comprehensive investigation",
        "hypotheses": ", ".join(plan.hypotheses) if plan and plan.hypotheses else "None provided",
        "num_sources": len(evidence),
        "sources_block": sources_block
    })

    from ..llm import extract_text_content
    raw_content = response.content if hasattr(response, "content") else response
    return extract_text_content(raw_content)
