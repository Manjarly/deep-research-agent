"""Evaluator Agent: Evaluates completeness of retrieved evidence and identifies gaps."""
import logging
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.language_models.chat_models import BaseChatModel

from ..models import ResearchPlan, EvaluationResult

logger = logging.getLogger(__name__)

EVALUATOR_SYSTEM_PROMPT = """You are a Lead Research Auditor and Fact-Checking Evaluator.
Your job is to critically evaluate whether the collected evidence is sufficiently comprehensive, factual, and rigorous to generate a definitive, high-quality research report.

Review the research task, original plan, and current compiled evidence.
Determine:
1. Is there sufficient factual depth to answer all sub-questions authoritatively?
2. Are there noticeable blind spots, missing statistics, or unresolved contradictions?
3. If information is lacking, provide 2 to 3 targeted, specific search queries to fill the gaps.

If the evidence is solid and covers the main points well, mark `is_sufficient` as True.
If critical pieces are missing, mark `is_sufficient` as False and output follow-up queries.
Adhere strictly to the requested schema."""


def evaluate_evidence(
    task: str,
    plan: ResearchPlan,
    evidence: List[Dict[str, Any]],
    llm: BaseChatModel
) -> EvaluationResult:
    """Audit collected evidence against the research plan."""
    # Summarize evidence for the prompt to fit context
    evidence_summaries = []
    for i, item in enumerate(evidence[:15]):  # Up to top 15 sources
        title = item.get("title", "Unknown")
        snippet = item.get("snippet", "")
        content = item.get("content", "")
        preview = (content[:300] if content else snippet[:300]).replace("\n", " ")
        evidence_summaries.append(f"[{i+1}] {title}: {preview}...")

    evidence_text = "\n".join(evidence_summaries) if evidence_summaries else "No evidence collected yet."

    parser = PydanticOutputParser(pydantic_object=EvaluationResult)

    prompt = ChatPromptTemplate.from_messages([
        ("system", EVALUATOR_SYSTEM_PROMPT),
        (
            "human",
            "Research Goal: {task}\n\n"
            "Sub-topics Planned: {sub_topics}\n\n"
            "Collected Evidence Summaries ({count} sources):\n{evidence_text}\n\n"
            "Output your assessment as valid JSON conforming to:\n{format_instructions}"
        )
    ])

    try:
        structured_llm = llm.with_structured_output(EvaluationResult)
        result = structured_llm.invoke(
            f"Audit the evidence for research topic: {task}\n"
            f"Planned topics: {plan.sub_topics}\n"
            f"Evidence collected: {evidence_text}"
        )
        if isinstance(result, EvaluationResult):
            return result
    except Exception as e:
        logger.debug(f"Structured output failed for evaluator: {e}. Falling back to prompt parser.")

    chain = prompt | llm | parser
    return chain.invoke({
        "task": task,
        "sub_topics": ", ".join(plan.sub_topics) if plan else "General investigation",
        "count": len(evidence),
        "evidence_text": evidence_text,
        "format_instructions": parser.get_format_instructions()
    })
