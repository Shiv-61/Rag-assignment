from src.loader import load_rag_dataset, create_documents
from src.splitter import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import build_vector_store
from src.retriever import get_retriever
from src.rag_chain import ask_question_with_stuff_chain


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


if __name__ == "__main__":
    main()