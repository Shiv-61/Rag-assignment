import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RAGConfig:
    model_name: str = field(
        default_factory=lambda: os.getenv("RAG_MODEL", "nvidia/nemotron-3-nano-30b-a3b:free")
    )
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dataset_name: str = "rag-datasets/rag-mini-wikipedia"
    k_documents: int = 4
    max_retries: int = 2
    openrouter_api_key: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY", "")
    )
    openrouter_base_url: str = "https://openrouter.ai/api/v1"


config = RAGConfig()
