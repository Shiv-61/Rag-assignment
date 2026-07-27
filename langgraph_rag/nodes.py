import json
import re
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langgraph_rag.state import GraphState
from langgraph_rag.retriever_store import get_wikipedia_retriever
from langgraph_rag.web_search import perform_web_search
from langgraph_rag.llm_factory import get_graph_llm


def extract_json(text: str) -> dict:
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception:
        return {}


def agent_router(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    llm = get_graph_llm()

    prompt = ChatPromptTemplate.from_template("""
Analyze the user question and determine:
1. Is it multi-hop or complex requiring sub-questions?
2. Return ONLY JSON:
{{"is_multihop": true/false, "sub_questions": ["sub_q1", "sub_q2"]}}

Question: {question}
""")
    chain = prompt | llm
    try:
        raw_res = chain.invoke({"question": question}).content
        parsed = extract_json(raw_res)
        is_multihop = parsed.get("is_multihop", False)
        sub_questions = parsed.get("sub_questions", [])
    except Exception:
        is_multihop = False
        sub_questions = []

    return {
        "is_multihop": is_multihop,
        "sub_questions": sub_questions
    }


def retrieve(state: GraphState) -> Dict[str, Any]:
    retriever = get_wikipedia_retriever()
    question = state["question"]
    sub_questions = state.get("sub_questions", [])

    all_docs = []
    seen_ids = set()
    queries = sub_questions if sub_questions else [question]

    for q in queries:
        docs = retriever.invoke(q)
        for doc in docs:
            doc_id = doc.metadata.get("id", doc.page_content[:30])
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_docs.append(doc)

    return {"documents": all_docs}


def grade_documents(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    documents = state.get("documents", [])
    if not documents:
        return {"documents": [], "web_search": True}

    llm = get_graph_llm()
    formatted_docs = []
    for i, doc in enumerate(documents, start=1):
        formatted_docs.append(f"Document {i}:\n{doc.page_content[:300]}")

    docs_str = "\n\n".join(formatted_docs)

    prompt = ChatPromptTemplate.from_template("""
Grade the relevance of each document to the question.
Return ONLY JSON mapping document number to relevance ("yes" or "no"):
{{"1": "yes", "2": "no", "3": "yes"}}

Question: {question}

Documents:
{docs_str}
""")
    chain = prompt | llm

    relevant_docs = []
    try:
        raw_res = chain.invoke({"question": question, "docs_str": docs_str}).content
        parsed = extract_json(raw_res)
        for i, doc in enumerate(documents, start=1):
            rel_str = str(parsed.get(str(i), "yes")).lower()
            if rel_str in ["yes", "y", "true"]:
                relevant_docs.append(doc)
    except Exception:
        relevant_docs = documents

    needs_web_search = len(relevant_docs) == 0

    return {
        "documents": relevant_docs,
        "web_search": needs_web_search
    }


def transform_query(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    retries = state.get("retries", 0) + 1
    llm = get_graph_llm()

    prompt = ChatPromptTemplate.from_template("""
Rewrite the question to improve search retrieval performance.
Return ONLY the rewritten search query.

Question: {question}
""")
    chain = prompt | llm
    try:
        rewritten = chain.invoke({"question": question}).content.strip()
    except Exception:
        rewritten = question

    return {
        "question": rewritten,
        "retries": retries
    }


def web_search_node(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    existing_docs = state.get("documents", [])
    web_docs = perform_web_search(question, max_results=3)

    return {
        "documents": existing_docs + web_docs,
        "web_search": False
    }


def generate(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    documents = state.get("documents", [])
    sub_questions = state.get("sub_questions", [])
    llm = get_graph_llm()

    context_parts = []
    sources = []
    for i, doc in enumerate(documents, start=1):
        context_parts.append(f"[Doc {i}]: {doc.page_content}")
        sources.append({
            "citation_id": f"Doc {i}",
            "title": doc.metadata.get("title", f"Document {i}"),
            "source": doc.metadata.get("source", "rag-mini-wikipedia"),
            "content_snippet": doc.page_content[:150] + "..."
        })

    context_str = "\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_template("""
Answer the user question based STRICTLY on the provided context documents.
Cite sources in line using [Doc X].
If context is insufficient, respond: "I don't have enough information in the provided context to answer this question."

Context:
{context}

Question:
{question}

Answer:
""")
    chain = prompt | llm
    try:
        answer = chain.invoke({
            "context": context_str,
            "question": question
        }).content
    except Exception as e:
        answer = f"Error generating answer: {str(e)}"

    return {
        "generation": answer,
        "sources": sources
    }


def grade_generation(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    documents = state.get("documents", [])
    generation = state.get("generation", "")

    if "I don't have enough information" in generation:
        return {"is_grounded": True, "answers_question": True}

    llm = get_graph_llm()
    context_str = "\n\n".join([doc.page_content[:250] for doc in documents])

    prompt = ChatPromptTemplate.from_template("""
Assess answer groundedness and question resolution.
Return ONLY JSON:
{{"is_grounded": true/false, "answers_question": true/false}}

Context:
{context}

Answer:
{generation}

User Question:
{question}
""")
    chain = prompt | llm

    try:
        raw_res = chain.invoke({
            "context": context_str,
            "generation": generation,
            "question": question
        }).content
        parsed = extract_json(raw_res)
        is_grounded = parsed.get("is_grounded", True)
        answers_question = parsed.get("answers_question", True)
    except Exception:
        is_grounded = True
        answers_question = True

    return {
        "is_grounded": is_grounded,
        "answers_question": answers_question
    }
