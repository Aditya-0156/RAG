"""
RAG Knowledge Base - Main Application.
Supports both CLI and web interface modes.
"""

import sys
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.rag_chain import RAGChain


def print_header():
    """Print application header."""
    print("\n" + "=" * 60)
    print("   RAG KNOWLEDGE BASE")
    print("   Document Question-Answering System")
    print("=" * 60)


def print_help():
    """Print help message."""
    print("""
COMMANDS:
  add <file_path>    Add a document to the knowledge base
  ask <question>     Ask a question about your documents
  stats              Show knowledge base statistics
  clear              Clear all documents from knowledge base
  help               Show this help message
  quit / exit        Exit the application

EXAMPLES:
  add /path/to/document.pdf
  add /path/to/folder/
  ask What is machine learning?
  stats
""")


def run_cli():
    """Run the interactive CLI."""
    print_header()

    print("\nInitializing RAG system...")
    try:
        rag = RAGChain()
        print("Ready.\n")
    except Exception as e:
        print(f"Failed to initialize: {e}")
        print("Make sure GOOGLE_API_KEY is set in your .env file")
        return

    stats = rag.get_stats()
    print(f"Documents in knowledge base: {stats['document_count']}")
    print("\nType 'help' for available commands.\n")

    while True:
        try:
            user_input = input("> ").strip()

            if not user_input:
                continue

            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            if command in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                break

            elif command == "help":
                print_help()

            elif command == "stats":
                stats = rag.get_stats()
                print("\nKnowledge Base Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                print()

            elif command == "clear":
                confirm = input("Are you sure? (yes/no): ")
                if confirm.lower() == "yes":
                    rag.retriever.vector_store.clear_collection()
                    print("Knowledge base cleared.\n")
                else:
                    print("Cancelled.\n")

            elif command == "add":
                if not args:
                    print("Please provide a file path. Example: add /path/to/file.pdf\n")
                    continue

                file_path = Path(args)
                if not file_path.exists():
                    print(f"File not found: {args}\n")
                    continue

                print(f"Adding documents from: {args}")
                try:
                    count = rag.add_documents(args)
                    print(f"Added {count} document chunks to knowledge base.\n")
                except Exception as e:
                    print(f"Error adding documents: {e}\n")

            elif command == "ask":
                if not args:
                    print("Please provide a question. Example: ask What is Python?\n")
                    continue

                print("\nSearching knowledge base...")
                result = rag.query(args)

                print(f"\nAnswer:\n{result['answer']}")

                if result["sources"]:
                    print(f"\nSources: {', '.join(result['sources'])}")
                print()

            else:
                # Treat unrecognized input as a question
                print("\nSearching knowledge base...")
                result = rag.query(user_input)

                print(f"\nAnswer:\n{result['answer']}")

                if result["sources"]:
                    print(f"\nSources: {', '.join(result['sources'])}")
                print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def run_web(host: str = "0.0.0.0", port: int = 8000):
    """Run the web interface."""
    import uvicorn
    from src.web.app import app

    print(f"\nStarting web interface at http://localhost:{port}")
    uvicorn.run(app, host=host, port=port)


def main():
    parser = argparse.ArgumentParser(description="RAG Knowledge Base")
    parser.add_argument(
        "--mode",
        choices=["cli", "web"],
        default="cli",
        help="Run mode: 'cli' for terminal, 'web' for browser UI (default: cli)",
    )
    parser.add_argument("--port", type=int, default=8000, help="Web server port")
    parser.add_argument("--host", default="0.0.0.0", help="Web server host")

    args = parser.parse_args()

    if args.mode == "web":
        run_web(host=args.host, port=args.port)
    else:
        run_cli()


if __name__ == "__main__":
    main()
