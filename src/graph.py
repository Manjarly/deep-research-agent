"""LangGraph orchestration graph for the Autonomous Deep Research Agent."""
import logging
from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.language_models.chat_models import BaseChatModel

from .models import ResearchState
from .tools.search import WebSearchTool
from .tools.scraper import WebScraperTool
from .agents.planner import plan_research
from .agents.researcher import conduct_research
from .agents.evaluator import evaluate_evidence
from .agents.synthesizer import synthesize_report
from .config import config

logger = logging.getLogger(__name__)


def create_research_graph(llm: BaseChatModel, search_tool: WebSearchTool = None, scraper_tool: WebScraperTool = None):
    """Build and compile the LangGraph research state machine."""
    _search_tool = search_tool or WebSearchTool()
    _scraper_tool = scraper_tool or WebScraperTool()

    def planner_node(state: ResearchState) -> Dict[str, Any]:
        task = state["task"]
        logger.info(f"--- [Node: Planner] Deconstructing topic: {task} ---")
        logs = list(state.get("logs", []))
        logs.append(f"🧠 [Planner] Analyzing research topic: '{task}'")

        plan = plan_research(task, llm)
        logs.append(
            f"📋 [Planner] Created plan with {len(plan.sub_topics)} dimensions and "
            f"{len(plan.search_queries)} search angles."
        )

        return {
            "plan": plan,
            "evidence": state.get("evidence", []),
            "iteration": 0,
            "max_iterations": state.get("max_iterations", config.MAX_RESEARCH_LOOPS),
            "logs": logs
        }

    def researcher_node(state: ResearchState) -> Dict[str, Any]:
        plan = state["plan"]
        iteration = state.get("iteration", 0)
        logs = list(state.get("logs", []))

        # Determine which queries to search
        if iteration == 0 or not state.get("evaluation"):
            queries = plan.search_queries
            logs.append(f"🌐 [Researcher] Executing initial search batch ({len(queries)} queries)...")
        else:
            queries = state["evaluation"].followup_queries
            logs.append(f"🔍 [Researcher] Loop {iteration}: Executing {len(queries)} targeted gap queries...")

        existing_evidence = state.get("evidence", [])
        updated_evidence = conduct_research(
            queries=queries,
            existing_evidence=existing_evidence,
            search_tool=_search_tool,
            scraper_tool=_scraper_tool
        )

        new_count = len(updated_evidence) - len(existing_evidence)
        logs.append(f"📥 [Researcher] Collected {new_count} new sources (Total: {len(updated_evidence)}).")

        return {
            "evidence": updated_evidence,
            "logs": logs
        }

    def evaluator_node(state: ResearchState) -> Dict[str, Any]:
        task = state["task"]
        plan = state["plan"]
        evidence = state.get("evidence", [])
        current_iteration = state.get("iteration", 0) + 1
        logs = list(state.get("logs", []))

        logs.append(f"⚖️ [Evaluator] Auditing evidence sufficiency (Iteration {current_iteration})...")
        evaluation = evaluate_evidence(task, plan, evidence, llm)

        if evaluation.is_sufficient:
            logs.append(f"✅ [Evaluator] Research is complete and sufficient: {evaluation.critique[:120]}...")
        else:
            logs.append(
                f"⚠️ [Evaluator] Gaps identified: {evaluation.critique[:120]}... "
                f"Generated {len(evaluation.followup_queries)} follow-up queries."
            )

        return {
            "evaluation": evaluation,
            "iteration": current_iteration,
            "logs": logs
        }

    def synthesizer_node(state: ResearchState) -> Dict[str, Any]:
        task = state["task"]
        plan = state["plan"]
        evidence = state.get("evidence", [])
        logs = list(state.get("logs", []))

        logs.append(f"✍️ [Synthesizer] Synthesizing final report from {len(evidence)} verified sources...")
        final_report = synthesize_report(task, plan, evidence, llm)
        logs.append("🎉 [Synthesizer] Deep research report generated successfully!")

        return {
            "final_report": final_report,
            "logs": logs
        }

    def should_continue(state: ResearchState) -> Literal["researcher_node", "synthesizer_node"]:
        """Conditional routing: loop back if incomplete and budget remains, else synthesize."""
        evaluation = state.get("evaluation")
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", config.MAX_RESEARCH_LOOPS)

        if evaluation and not evaluation.is_sufficient and iteration < max_iterations:
            if evaluation.followup_queries:
                logger.info(f"Looping back to researcher for iteration {iteration + 1}...")
                return "researcher_node"

        return "synthesizer_node"

    # Assemble state graph
    workflow = StateGraph(ResearchState)

    workflow.add_node("planner_node", planner_node)
    workflow.add_node("researcher_node", researcher_node)
    workflow.add_node("evaluator_node", evaluator_node)
    workflow.add_node("synthesizer_node", synthesizer_node)

    workflow.add_edge(START, "planner_node")
    workflow.add_edge("planner_node", "researcher_node")
    workflow.add_edge("researcher_node", "evaluator_node")
    workflow.add_conditional_edges(
        "evaluator_node",
        should_continue,
        {
            "researcher_node": "researcher_node",
            "synthesizer_node": "synthesizer_node"
        }
    )
    workflow.add_edge("synthesizer_node", END)

    return workflow.compile()
