from datasets import load_dataset
from langchain_core.documents import Document


def load_rag_dataset():

    corpus = load_dataset(
        "rag-datasets/rag-mini-wikipedia",
        "text-corpus"
    )["passages"]

    qa = load_dataset(
        "rag-datasets/rag-mini-wikipedia",
        "question-answer"
    )["test"]

    return corpus, qa


def create_documents(corpus):

    documents = []

    for row in corpus:

        doc = Document(
            page_content=row["passage"],
            metadata={
                "document_id": row["id"],
                "source": "rag-mini-wikipedia",
            },
        )

        documents.append(doc)

    return documents