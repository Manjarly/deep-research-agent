"""Planner Agent: Decomposes a research question into targeted search queries."""
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.language_models.chat_models import BaseChatModel

from ..models import ResearchPlan

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """You are an elite Research Director and Strategist.
Your goal is to take a research request, dissect it into comprehensive sub-dimensions, and formulate highly effective, diverse search queries to uncover deep, authoritative information.

Avoid generic or surface-level queries. Instead formulate queries that target:
1. Core fundamentals, background, and state-of-the-art developments.
2. Concrete statistics, benchmarks, case studies, or empirical evidence.
3. Contrasting viewpoints, known challenges, architectural tradeoffs, and future outlook.

Output a structured research plan adhering strictly to the schema."""


def plan_research(task: str, llm: BaseChatModel) -> ResearchPlan:
    """Analyze the task and generate a structured research plan."""
    parser = PydanticOutputParser(pydantic_object=ResearchPlan)

    prompt = ChatPromptTemplate.from_messages([
        ("system", PLANNER_SYSTEM_PROMPT),
        ("human", "Research Topic/Question: {task}\n\nFormat your response as valid JSON matching this schema:\n{format_instructions}")
    ])

    # Try structured output first (native function calling/JSON schema)
    try:
        structured_llm = llm.with_structured_output(ResearchPlan)
        plan = structured_llm.invoke(f"Deconstruct this research topic into an in-depth research plan: {task}")
        if isinstance(plan, ResearchPlan):
            return plan
    except Exception as e:
        logger.debug(f"Structured output method failed: {e}. Falling back to prompt parsing.")

    # Fallback to standard prompt + PydanticOutputParser
    chain = prompt | llm | parser
    return chain.invoke({
        "task": task,
        "format_instructions": parser.get_format_instructions()
    })
