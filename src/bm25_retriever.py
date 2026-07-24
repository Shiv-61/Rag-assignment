from langchain_community.retrievers import BM25Retriever

def build_bm25_retriever(chunks):

    bm25_retriever = BM25Retriever.from_documents(chunks)

    bm25_retriever.k = 5

    return bm25_retriever