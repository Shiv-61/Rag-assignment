from datasets import load_dataset


def load_rag_dataset():
    """
    Load the corpus and QA dataset.
    """

    corpus = load_dataset(
        "rag-datasets/rag-mini-wikipedia",
        "text-corpus"
    )["passages"]

    qa = load_dataset(
        "rag-datasets/rag-mini-wikipedia",
        "question-answer"
    )["test"]

    return corpus, qa