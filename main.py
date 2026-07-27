from urllib import response

from src.embeddings import get_embedding_model
from src.loader import create_documents, load_rag_dataset
from src.rag_chain import ask_question_with_stuff_chain
from src.retriever import get_retriever
from src.splitter import split_documents
from src.vector_store import build_vector_store


def main():

    # ============================
    # Load Dataset
    # ============================

    corpus, qa = load_rag_dataset()

    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    print(f"Corpus Size : {len(corpus)}")
    print(f"QA Size     : {len(qa)}")

    print("\nCorpus Columns:")
    print(corpus.column_names)

    print("\nFirst Corpus Row:")
    print(corpus[0])

    # ============================
    # Create LangChain Documents
    # ============================

    documents = create_documents(corpus)

    print("\n" + "=" * 60)
    print(f"Documents Created : {len(documents)}")

    print("\nFirst Document:")
    print(documents[0])

    # ============================
    # Split Documents
    # ============================

    chunks = split_documents(documents)

    print("\n" + "=" * 60)
    print(f"Total Chunks : {len(chunks)}")

    print("\nFirst Chunk:")
    print(chunks[0])

    # ============================
    # Embedding Model
    # ============================

    embedding_model = get_embedding_model()

    # ============================
    # Build Vector Store
    # ============================

    vector_store = build_vector_store(
        chunks,
        embedding_model,
    )

    print("\nVector Store Created Successfully!")

    # ============================
    # Create Retriever
    # ============================

    retriever = get_retriever(vector_store)

    # ============================
    # Test Questions
    # ============================

    questions = [
        "Where is Uruguay located?",
        "What is the capital of Uruguay?",
        "How many people live in Uruguay?",
        "What language is spoken in Uruguay?",
        "Who is Virat Kohli?",
        "What is LangGraph?",
        "Who is Elon Musk?",
        "What is ChatGPT?",
    ]

    # ============================
    # Ask Questions
    # ============================

    for question in questions:
        print("\n" + "=" * 100)
        print(f"Question: {question}")
        print("=" * 100)

        answer = ask_question_with_stuff_chain(
            retriever,
            question,
        )

        print("\nAnswer:\n")
        print(answer)


def test_retrieval_chain():
    from src.rag_chain import ask_question_with_retrieval_chain

    corpus, qa = load_rag_dataset()
    documents = create_documents(corpus)
    chunks = split_documents(documents)
    embedding_model = get_embedding_model()
    vector_store = build_vector_store(
        chunks,
        embedding_model,
    )

    retriever = get_retriever(vector_store)
    questions = [
        "Where is Uruguay located?",
    ]
    for question in questions:
        print("\n" + "=" * 100)
        print(f"Question: {question}")
        print("=" * 100)
        response = ask_question_with_retrieval_chain(
            retriever,
            question,
        )
        print("\nAnswer:\n")
        print(response)


def evaluate_retrieval():
    # pyrefly: ignore [missing-import]
    from src.eval import (
        get_first_correct_rank,
        hit_at_k,
        reciprocal_rank,
    )

    corpus, qa = load_rag_dataset()
    documents = create_documents(corpus)
    chunks = split_documents(documents)
    embedding_model = get_embedding_model()
    vector_store = build_vector_store(chunks, embedding_model)
    retriever = get_retriever(vector_store)

    k_values = [1, 3, 5, 10]

    ranks = []

    for sample in qa:
        question = sample["question"]
        answer = sample["answer"]

        docs = retriever.invoke(question)

        rank = get_first_correct_rank(docs, answer)
        ranks.append(rank)

    print("\n" + "=" * 60)
    print("RETRIEVAL METRICS")
    print("=" * 60)

    for k in k_values:
        hit_scores = [hit_at_k(r, k) for r in ranks]
        mean_hit = sum(hit_scores) / len(hit_scores)

        rr_scores = [reciprocal_rank(r) for r in ranks]
        mean_rr = sum(rr_scores) / len(rr_scores)

        print(f"\nHit Rate @{k}:")
        print(f"  Mean: {mean_hit:.4f}")

        print(f"\nReciprocal Rank:")
        print(f"  Mean: {mean_rr:.4f}")


def generate_golden():
    from src.golden_dataset import create_golden_dataset

    corpus, qa = load_rag_dataset()
    documents = create_documents(corpus)
    chunks = split_documents(documents)
    embedding_model = get_embedding_model()
    vector_store = build_vector_store(chunks, embedding_model)
    retriever = get_retriever(vector_store)

    create_golden_dataset(
        qa, corpus, retriever, output_path="data/golden_dataset.json", max_samples=50
    )


def hybrid_plus_rerank():
    corpus, qa = load_rag_dataset()
    documents = create_documents(corpus)
    chunks = split_documents(documents)

    from src.bm25_retriever import build_bm25_retriever

    question = "Jewish"

    bm25_retriever = build_bm25_retriever(chunks)
    docs = bm25_retriever.invoke(question)

    embedding_model = get_embedding_model()
    vector_store = build_vector_store(chunks, embedding_model)
    retriever = get_retriever(vector_store)

    from src.hybrid_retriever import build_hybrid_retriever
    from src.reranker import build_reranker

    hybrid_retriever = build_hybrid_retriever(bm25_retriever, retriever)
    reranked_retriever = build_reranker(hybrid_retriever)
    docs = reranked_retriever.invoke(question)

    for doc in docs:
        print(doc.metadata)
        print(doc.page_content[:200])


if __name__ == "__main__":
    from src.llm import get_llm
    from src.querry_transfrom import transform_query

    question = "elon musk"
    llm = get_llm()
    transformed_question = transfor m_query(question, llm)
    print(transformed_question)
