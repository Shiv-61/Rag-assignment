import os
from langchain_openai import ChatOpenAI
from langgraph_rag.config import config


def get_graph_llm(temperature: float = 0.0, model: str = None):
    model_to_use = model or config.model_name
    api_key = config.openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "dummy_key")

    return ChatOpenAI(
        model=model_to_use,
        api_key=api_key,
        base_url=config.openrouter_base_url,
        temperature=temperature,
        default_headers={
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "LangGraph-RAG"
        }
    )
