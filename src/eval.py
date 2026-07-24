def inspect_retrieval(retriever, qa, n=5):

    for sample in qa.select(range(n)):

        question = sample["question"]
        answer = sample["answer"]

        print("=" * 80)
        print("Question:", question)
        print("Expected Answer:", answer)
        print("=" * 80)

        docs = retriever.invoke(question)

        for i, doc in enumerate(docs, start=1):
            print(f"\nRank {i}")
            print(doc.metadata)
            print(doc.page_content[:300])
            print("-" * 80)


def get_first_correct_rank(retrieved_docs, expected_answer):

    expected_answer = expected_answer.lower()

    for rank, doc in enumerate(retrieved_docs, start=1):

        if expected_answer in doc.page_content.lower():
            return rank

    return None

def hit_at_k(rank, k):
    return 1 if rank is not None and rank <= k else 0


def reciprocal_rank(rank):
    if rank is None:
        return 0
    return 1 / rank