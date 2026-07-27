from langgraph_rag.state import GraphState


def decide_to_generate(state: GraphState) -> str:
    web_search = state.get("web_search", False)
    documents = state.get("documents", [])
    retries = state.get("retries", 0)
    max_retries = state.get("max_retries", 3)

    if web_search or len(documents) == 0:
        if retries < max_retries:
            return "transform_query"
        else:
            return "web_search"
    return "generate"


def grade_generation_v_documents_and_question(state: GraphState) -> str:
    is_grounded = state.get("is_grounded", True)
    answers_question = state.get("answers_question", True)
    retries = state.get("retries", 0)
    max_retries = state.get("max_retries", 3)

    if is_grounded and answers_question:
        return "useful"
    elif not is_grounded:
        if retries < max_retries:
            return "not_grounded"
        return "abstain"
    else:
        if retries < max_retries:
            return "not_useful"
        return "abstain"
