import os
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph_rag.state import GraphState
from langgraph_rag.nodes import (
    agent_router,
    retrieve,
    grade_documents,
    transform_query,
    web_search_node,
    generate,
    grade_generation
)
from langgraph_rag.edges import (
    decide_to_generate,
    grade_generation_v_documents_and_question
)


def abstain_node(state: GraphState) -> Dict[str, Any]:
    return {
        "generation": "I don't have enough information in the provided context to answer this question.",
        "sources": state.get("sources", [])
    }


def build_arag_crag_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("agent_router", agent_router)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate)
    workflow.add_node("grade_generation", grade_generation)
    workflow.add_node("abstain", abstain_node)

    workflow.add_edge(START, "agent_router")
    workflow.add_edge("agent_router", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "web_search": "web_search",
            "generate": "generate"
        }
    )

    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("web_search", "generate")

    workflow.add_edge("generate", "grade_generation")

    workflow.add_conditional_edges(
        "grade_generation",
        grade_generation_v_documents_and_question,
        {
            "useful": END,
            "not_grounded": "generate",
            "not_useful": "transform_query",
            "abstain": "abstain"
        }
    )

    workflow.add_edge("abstain", END)

    app = workflow.compile()
    return app


def save_graph_diagram(app, output_path: str = "langgraph_rag/graph_diagram.png"):
    try:
        png_bytes = app.get_graph().draw_mermaid_png()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(png_bytes)
        return output_path
    except Exception as e:
        print(f"Graph PNG rendering warning: {str(e)}")
        return None
