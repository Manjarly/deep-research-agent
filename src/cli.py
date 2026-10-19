"""Command-line interface for the Autonomous Deep Research Agent."""
import sys
import argparse
import datetime
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .config import config
from .llm import get_llm
from .graph import create_research_graph

console = Console()


def run_cli():
    parser = argparse.ArgumentParser(
        description="Autonomous Deep Research Agent powered by LangChain & LangGraph"
    )
    parser.add_argument(
        "topic",
        nargs="?",
        default=None,
        help="The research topic or question to investigate."
    )
    parser.add_argument(
        "--provider",
        default="gemini",
        choices=["gemini", "openai", "anthropic", "ollama"],
        help="LLM provider to use (default: gemini)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Specific model name (e.g., gemini-1.5-flash, gemini-1.5-pro, gpt-4o)"
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=config.MAX_RESEARCH_LOOPS,
        help="Max research reflection loops (default: 2)"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="LLM API Key (if not set in .env)"
    )

    args = parser.parse_args()

    console.print(
        Panel.fit(
            "[bold cyan]🔍 Autonomous Deep Research Agent[/bold cyan]\n"
            "[dim]LangChain • LangGraph • Multi-Agent Reflection • Web Grounding[/dim]",
            border_style="cyan"
        )
    )

    topic = args.topic
    if not topic:
        topic = console.input("[bold green]Enter your research topic or question: [/bold green]").strip()

    if not topic:
        console.print("[red]Error: Research topic cannot be empty.[/red]")
        sys.exit(1)

    # Initialize LLM
    try:
        llm = get_llm(
            provider=args.provider,
            model_name=args.model,
            api_key=args.api_key
        )
    except Exception as e:
        console.print(f"[bold red]Configuration Error:[/bold red] {e}")
        sys.exit(1)

    graph = create_research_graph(llm)

    initial_state = {
        "task": topic,
        "evidence": [],
        "iteration": 0,
        "max_iterations": args.loops,
        "logs": []
    }

    console.print(f"\n[bold yellow]🚀 Launching research workflow for:[/bold yellow] [italic]{topic}[/italic]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task_id = progress.add_task("[cyan]Executing multi-agent research graph...", total=None)

        final_state = None
        for event in graph.stream(initial_state):
            for node_name, node_output in event.items():
                if "logs" in node_output and node_output["logs"]:
                    latest_log = node_output["logs"][-1]
                    progress.update(task_id, description=f"[bold green]{latest_log}[/bold green]")
                final_state = node_output

        progress.remove_task(task_id)

    if not final_state or "final_report" not in final_state:
        # Fallback if streaming format wrapped state
        console.print("[red]Research completed but report synthesis was not returned.[/red]")
        return

    report = final_state.get("final_report", "")
    evidence = final_state.get("evidence", [])
    plan = final_state.get("plan")

    # Display Plan Summary
    if plan:
        table = Table(title="📋 Research Plan Deconstructed", border_style="blue")
        table.add_column("Sub-Dimensions Explored", style="cyan")
        for sub in plan.sub_topics:
            table.add_row(sub)
        console.print(table)
        console.print()

    # Display Sources Summary
    if evidence:
        table = Table(title=f"🌐 Sources & Evidence Gathered ({len(evidence)} total)", border_style="magenta")
        table.add_column("#", style="dim", width=4)
        table.add_column("Source Title", style="bold")
        table.add_column("URL", style="underline blue")
        for i, item in enumerate(evidence[:10], 1):
            table.add_row(str(i), item.get("title", "Untitled")[:50], item.get("url", ""))
        console.print(table)
        console.print()

    # Save report to file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_title = "".join(c if c.isalnum() else "_" for c in topic[:30]).strip("_")
    report_filename = f"report_{clean_title}_{timestamp}.md"
    report_path = config.REPORTS_DIR / report_filename

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    console.print(Panel(Markdown(report), title="📑 Final Synthesized Research Report", border_style="green"))
    console.print(f"\n[bold green]💾 Full report saved to:[/bold green] [underline]{report_path}[/underline]\n")


if __name__ == "__main__":
    run_cli()
