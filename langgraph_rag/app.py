import os
import streamlit as st
from langgraph_rag.graph import build_arag_crag_graph, save_graph_diagram


st.set_page_config(
    page_title="Agentic + Corrective RAG (LangGraph)",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Stateful Agentic & Corrective RAG (LangGraph)")
st.caption("Part G Capstone: ARAG + CRAG + Self-RAG with Mini-Wikipedia Corpus")


@st.cache_resource
def load_rag_app():
    app = build_arag_crag_graph()
    save_graph_diagram(app, "langgraph_rag/graph_diagram.png")
    return app


app = load_rag_app()

st.sidebar.header("📊 Graph Architecture")
if os.path.exists("langgraph_rag/graph_diagram.png"):
    st.sidebar.image("langgraph_rag/graph_diagram.png", caption="StateGraph Visualizer")

st.sidebar.header("⚡ Quick Demo Presets")
preset = st.sidebar.radio(
    "Choose a preset query:",
    [
        "Custom Input",
        "Normal: Where is Uruguay located?",
        "Multi-Hop: What is the capital of Uruguay and how many people live there?",
        "Abstain: Who won the 2026 IPL final between CSK and MI?"
    ]
)

if preset == "Normal: Where is Uruguay located?":
    default_q = "Where is Uruguay located?"
elif preset == "Multi-Hop: What is the capital of Uruguay and how many people live there?":
    default_q = "What is the capital of Uruguay and how many people live there?"
elif preset == "Abstain: Who won the 2026 IPL final between CSK and MI?":
    default_q = "Who won the 2026 IPL final between CSK and MI?"
else:
    default_q = ""

user_question = st.text_input("Ask a question:", value=default_q)

if st.button("Submit Question", type="primary") and user_question.strip():
    with st.spinner("Executing LangGraph RAG Agent..."):
        initial_state = {
            "question": user_question,
            "documents": [],
            "generation": "",
            "retries": 0,
            "max_retries": 2,
            "web_search": False,
            "sub_questions": [],
            "sub_answers": [],
            "is_multihop": False,
            "sources": [],
            "is_grounded": True,
            "answers_question": True
        }

        final_state = app.invoke(initial_state)

    col1, col2 = st.column(2)

    with col1:
        st.subheader("💡 Grounded Answer")
        st.write(final_state.get("generation", "No output."))

        if final_state.get("is_multihop"):
            st.info(f"📌 Multi-Hop Decomposed Queries: {final_state.get('sub_questions')}")

    with col2:
        st.subheader("📚 Sources Panel")
        sources = final_state.get("sources", [])
        if not sources:
            st.warning("No internal corpus sources utilized.")
        else:
            for src in sources:
                with st.expander(f"[{src['citation_id']}] {src['title']}"):
                    st.write(f"**Source:** {src['source']}")
                    st.write(f"**Content:** {src['content_snippet']}")
