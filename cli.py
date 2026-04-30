"""
cli.py – Command-line interface for the College Knowledge Assistant.

Usage:
    python cli.py ingest                        # Load all sample docs
    python cli.py ingest path/to/doc.pdf        # Load specific file
    python cli.py ask "What is attendance rule?" # One-shot question
    python cli.py chat                           # Interactive multi-turn chat
    python cli.py stats                          # Show index info
    python cli.py clear                          # Clear the index
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel   import Panel
from rich.table   import Table
from rich.prompt  import Prompt
from rich.markup  import escape

from config import DOCS_DIR
from retrieval.vector_store import VectorStore
from generation.rag import ask, ask_with_history
from ingest.pipeline import ingest_document, ingest_folder, clear_collection


console = Console()
store   = VectorStore()


# ── Helpers ───────────────────────────────────────────────────────────────────

def print_answer(result: dict):
    console.print(Panel(
        result["answer"],
        title="[bold green]🤖 Assistant Answer[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))

    if result["sources"]:
        table = Table(title="📄 Source Citations",
                      show_header=True, header_style="bold cyan")
        table.add_column("Source",    style="cyan",  no_wrap=True)
        table.add_column("Relevance", style="green", justify="right")
        table.add_column("Snippet",   style="dim")

        for src in result["sources"]:
            table.add_row(
                src["source"],
                f"{src['score']:.2f}",
                escape(src["snippet"][:100]) + "…",
            )
        console.print(table)


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_ingest(args):
    if args.path:
        path = Path(args.path)
        if path.is_dir():
            ingest_folder(path, store=store)
        else:
            extra = {}
            if args.doc_type: extra["doc_type"] = args.doc_type
            if args.year:     extra["year"]     = args.year
            ingest_document(path, extra_meta=extra or None, store=store)
    else:
        console.print(f"[yellow]Loading all documents from {DOCS_DIR} …[/yellow]")
        ingest_folder(DOCS_DIR, store=store)


def cmd_ask(args):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]❌ ANTHROPIC_API_KEY not set. Export it first.[/red]")
        return
    filters = {}
    if args.doc_type: filters["doc_type"] = args.doc_type
    if args.year:     filters["year"]     = args.year

    console.print(f"\n[bold]Query:[/bold] {args.query}\n")
    with console.status("Searching and generating answer…"):
        result = ask(
            query   = args.query,
            mode    = args.mode,
            top_k   = args.top_k,
            filters = filters or None,
            store   = store,
        )
    print_answer(result)


def cmd_chat(args):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]❌ ANTHROPIC_API_KEY not set.[/red]")
        return
    if store.count() == 0:
        console.print("[red]❌ No documents indexed. Run: python cli.py ingest[/red]")
        return

    console.print(Panel(
        "[bold]College Knowledge Assistant – Multi-turn Chat[/bold]\n"
        "Type [cyan]'quit'[/cyan] or [cyan]'exit'[/cyan] to end the session.\n"
        "Type [cyan]'sources'[/cyan] to see what documents are indexed.",
        title="🎓 Chat Mode", border_style="blue"
    ))

    history = []
    while True:
        try:
            query = Prompt.ask("\n[bold cyan]You[/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if query.strip().lower() in {"quit", "exit", "q"}:
            console.print("[dim]Session ended.[/dim]")
            break
        if query.strip().lower() == "sources":
            for s in store.list_sources():
                console.print(f"  • {s}")
            continue
        if not query.strip():
            continue

        with console.status("Thinking…"):
            result = ask_with_history(
                query   = query,
                history = history,
                mode    = args.mode,
                top_k   = args.top_k,
                store   = store,
            )

        history = result["history"]
        print_answer(result)


def cmd_stats(args):
    count   = store.count()
    sources = store.list_sources()
    console.print(f"\n[bold]📊 Index Statistics[/bold]")
    console.print(f"  Total chunks : [green]{count}[/green]")
    console.print(f"  Documents    : [green]{len(sources)}[/green]")
    for s in sources:
        console.print(f"    • {s}")


def cmd_clear(args):
    confirm = Prompt.ask("Are you sure you want to clear the index? [y/N]", default="N")
    if confirm.lower() == "y":
        clear_collection(store)
    else:
        console.print("[dim]Cancelled.[/dim]")


# ── Argument parser ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="College Student Knowledge Assistant – CLI"
    )
    sub = parser.add_subparsers(dest="command")

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest documents into the index")
    p_ingest.add_argument("path",       nargs="?", help="File or folder to ingest")
    p_ingest.add_argument("--doc-type", dest="doc_type", help="Metadata: document type")
    p_ingest.add_argument("--year",     help="Metadata: document year")

    # ask
    p_ask = sub.add_parser("ask", help="Ask a one-shot question")
    p_ask.add_argument("query",         help="The question to ask")
    p_ask.add_argument("--mode",        default="hybrid",
                       choices=["hybrid","semantic","keyword"])
    p_ask.add_argument("--top-k",  dest="top_k", type=int, default=5)
    p_ask.add_argument("--doc-type", dest="doc_type", help="Filter by doc_type")
    p_ask.add_argument("--year",     help="Filter by year")

    # chat
    p_chat = sub.add_parser("chat", help="Multi-turn interactive chat")
    p_chat.add_argument("--mode",   default="hybrid",
                        choices=["hybrid","semantic","keyword"])
    p_chat.add_argument("--top-k",  dest="top_k", type=int, default=5)

    # stats
    sub.add_parser("stats", help="Show index statistics")

    # clear
    sub.add_parser("clear", help="Clear the vector index")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "ask":
        cmd_ask(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "clear":
        cmd_clear(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
