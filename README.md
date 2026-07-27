# Stateful Agentic RAG (ARAG) + Corrective RAG (CRAG) with LangGraph

A production-ready **Stateful Agentic & Corrective RAG (ARAG/CRAG)** system built using **LangGraph**, **LangChain**, and **FAISS + BM25 Hybrid Retrieval** over the `rag-datasets/rag-mini-wikipedia` dataset.

---

## 🎯 Architecture & Workflow

```
                        [START]
                           │
                           ▼
                    [agent_router] ──► (Detects multi-hop & decomposes into sub-questions)
                           │
                           ▼
                       [retrieve] ──► (FAISS Dense + BM25 Sparse Hybrid Ensemble)
                           │
                           ▼
                   [grade_documents] (Scores relevance of context passages)
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
  [Enough Relevant Docs]    [Weak / Insufficient Context]
             │                           │
             │           ┌───────────────┴───────────────┐
             │           ▼                               ▼
             │   [transform_query]                 [web_search]
             │   (Rewrite search query)        (DuckDuckGo fallback)
             │           │                               │
             │           └───────────────┬───────────────┘
             │                           │
             └─────────────┬─────────────┘
                           │
                           ▼
                      [generate] (Synthesizes grounded answer with [Doc X] citations)
                           │
                           ▼
                  [grade_generation] (Self-RAG Hallucination & Completeness Check)
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
    [Grounded & Answers]        [Ungrounded / Insufficient]
             │                           │
             ▼                           ▼
           [END]                     [abstain] ──► [END]
```

---

## 🧠 Node & Conditional Edge Breakdown

### Nodes (`langgraph_rag/nodes.py`)
- **`agent_router`**: Analyzes question complexity. If multi-hop, decomposes it into sub-questions.
- **`retrieve`**: Invokes the hybrid retriever (FAISS vector store + BM25 keyword search) cached from `rag-mini-wikipedia`.
- **`grade_documents`**: Evaluates document relevance in batches and drops irrelevant passages. Sets `web_search=True` if context is empty.
- **`transform_query`**: Rewrites the user query to optimize document retrieval on subsequent retries.
- **`web_search_node`**: Fetches external web results via DuckDuckGo as a fallback when corpus context is missing.
- **`generate`**: Generates grounded answers strictly using context documents with inline citations (`[Doc X]`).
- **`grade_generation`**: Self-RAG node evaluating answer groundedness (no hallucinations) and question completion.
- **`abstain`**: Produces a controlled abstention response (*"I don't have enough information..."*) when context is insufficient.

### Conditional Edges (`langgraph_rag/edges.py`)
- **`decide_to_generate` (CRAG Edge)**: After `grade_documents`, routes to `generate` if relevant context exists, or to `transform_query` / `web_search` if retrieval was weak.
- **`grade_generation_v_documents_and_question` (Self-RAG Edge)**: After `generate`, routes to `END` if useful, retries generation/query rewrite if failing within retry limits, or routes to `abstain`.

---

## 📁 Directory Structure

```
.
├── langgraph_rag/
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Centralized production configuration (RAGConfig)
│   ├── state.py              # Typed GraphState schema
│   ├── llm_factory.py        # OpenRouter / LLM client provider
│   ├── retriever_store.py    # FAISS + BM25 hybrid retriever with disk caching
│   ├── web_search.py         # DuckDuckGo fallback web search tool
│   ├── nodes.py              # Graph execution nodes
│   ├── edges.py              # Conditional routing logic (CRAG & Self-RAG)
│   ├── graph.py              # StateGraph compilation & Mermaid diagram exporter
│   ├── cli.py                # Command-line interface with --demo mode
│   ├── app.py                # Streamlit visual interface
│   └── graph_diagram.png     # Rendered workflow graph architecture
├── .env                      # API keys configuration
└── README.md                 # Single project documentation
```

---

## 🚀 Installation & Running

### 1. Environment Setup
Ensure your `.env` file contains your OpenRouter API key:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 2. Command Line Interface (CLI)
```bash
# Run automated demo suite (Normal, Multi-Hop, and Abstention scenarios)
python -m langgraph_rag.cli --demo

# Run a custom question
python -m langgraph_rag.cli --question "Where is Uruguay located?"
```

### 3. Streamlit Interface (Optional UI)
```bash
streamlit run langgraph_rag/app.py
```

---

## 🧪 Demonstration Cases

1. **Normal In-Corpus Question**:
   - **Query**: *"Where is Uruguay located?"*
   - **Behavior**: Retrieves Wikipedia passages, grades relevance, and produces a grounded answer with inline citations `[Doc X]` and a Sources panel.
2. **Multi-Hop Query**:
   - **Query**: *"What is the capital of Uruguay and how many people live there?"*
   - **Behavior**: `agent_router` decomposes the query into sub-questions, retrieves per sub-question, and composes a unified cited response.
3. **Out-of-Corpus Abstention**:
   - **Query**: *"Who won the 2026 IPL final between CSK and MI?"*
   - **Behavior**: Identifies absence of relevant Wikipedia context, avoids hallucination, and returns:
     > *"I don't have enough information in the provided context to answer this question."*
