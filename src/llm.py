import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_llm():
    return ChatOpenAI(
        model="nvidia/nemotron-3-ultra-550b-a55b:free",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        default_headers={
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Agentic-RAG"
        },
        model_kwargs={
            "reasoning": {
                "enabled": True
            }
        }
    )