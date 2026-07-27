import argparse
import sys
from langgraph_rag.graph import build_arag_crag_graph, save_graph_diagram


def run_query(app, question: str, max_retries: int = 2):
    initial_state = {
        "question": question,
        "documents": [],
        "generation": "",
        "retries": 0,
        "max_retries": max_retries,
        "web_search": False,
        "sub_questions": [],
        "sub_answers": [],
        "is_multihop": False,
        "sources": [],
        "is_grounded": True,
        "answers_question": True
    }

    final_state = app.invoke(initial_state)

    print("\n" + "=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)

    if final_state.get("is_multihop"):
        print(f"📌 Multi-Hop Detected: Decomposed into -> {final_state.get('sub_questions')}")

    print("\n💡 GROUNDED ANSWER:")
    print(final_state.get("generation", "No generation produced."))

    print("\n📚 SOURCES PANEL:")
    sources = final_state.get("sources", [])
    if not sources:
        print("  No internal corpus sources utilized.")
    else:
        for src in sources:
            print(f"  - [{src['citation_id']}] {src['title']} ({src['source']})")
            print(f"    Snippet: {src['content_snippet']}\n")

    return final_state


def run_demo():
    print("🚀 Initializing LangGraph Agentic + Corrective RAG (ARAG/CRAG)...")
    app = build_arag_crag_graph()

    diagram_path = save_graph_diagram(app)
    if diagram_path:
        print(f"📊 Graph Mermaid diagram exported to: {diagram_path}")

    print("\n" + "#" * 80)
    print("DEMO CASE 1: Normal In-Corpus Question")
    print("#" * 80)
    run_query(app, "Where is Uruguay located?")

    print("\n" + "#" * 80)
    print("DEMO CASE 2: Multi-Hop Complex Question")
    print("#" * 80)
    run_query(app, "What is the capital of Uruguay and how many people live there?")

    print("\n" + "#" * 80)
    print("DEMO CASE 3: Out-of-Corpus Abstention Demonstration")
    print("#" * 80)
    run_query(app, "Who won the 2026 IPL final between CSK and MI?")


def main():
    parser = argparse.ArgumentParser(description="LangGraph Agentic & Corrective RAG CLI")
    parser.add_argument("--demo", action="store_true", help="Run automated test demo for 3 required scenarios")
    parser.add_argument("--question", type=str, help="Ask a custom question")
    args = parser.parse_args()

    if args.demo or not args.question:
        run_demo()
    else:
        app = build_arag_crag_graph()
        save_graph_diagram(app)
        run_query(app, args.question)


if __name__ == "__main__":
    main()
