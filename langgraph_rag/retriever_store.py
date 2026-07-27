import os
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph_rag.config import config

_retriever_instance = None
FAISS_CACHE_DIR = "langgraph_rag/faiss_index"


def get_wikipedia_retriever(k: int = None):
    global _retriever_instance
    if _retriever_instance is not None:
        return _retriever_instance

    k_val = k or config.k_documents
    embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model_name)

    corpus = load_dataset(
        config.dataset_name,
        "text-corpus"
    )["passages"]

    documents = []
    for row in corpus:
        doc = Document(
            page_content=row["passage"],
            metadata={
                "id": row["id"],
                "source": config.dataset_name,
                "title": row.get("title", f"Passage {row['id']}")
            }
        )
        documents.append(doc)

    if os.path.exists(FAISS_CACHE_DIR):
        vector_store = FAISS.load_local(
            FAISS_CACHE_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        vector_store = FAISS.from_documents(documents, embeddings)
        os.makedirs(FAISS_CACHE_DIR, exist_ok=True)
        vector_store.save_local(FAISS_CACHE_DIR)

    dense_retriever = vector_store.as_retriever(search_kwargs={"k": k_val})

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k_val

    _retriever_instance = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.5, 0.5]
    )

    return _retriever_instance
