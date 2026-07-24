from langchain_classic.retrievers import EnsembleRetriever


def build_hybrid_retriever(
    bm25_retriever,
    dense_retriever,
):

    hybrid = EnsembleRetriever(
        retrievers=[
            bm25_retriever,
            dense_retriever,
        ],
        weights=[0.5, 0.5],
    )

    return hybrid
