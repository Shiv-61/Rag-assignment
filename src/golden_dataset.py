import json
from pathlib import Path
import pandas as pd


def create_golden_dataset(qa_dataset, corpus_dataset, retriever, output_path="data/golden_dataset.json", max_samples=50):
    """
    Creates a Golden Dataset for RAG evaluation.
    
    A golden dataset contains:
    - id: Unique query identifier
    - question: User query
    - expected_answer: Ground truth reference answer
    - relevant_doc_ids: List of ground-truth document IDs containing the answer
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    golden_records = []

    # Map corpus text/id for ground truth verification
    corpus_map = {row["id"]: row["passage"] for row in corpus_dataset}

    for i, sample in enumerate(qa_dataset):
        if len(golden_records) >= max_samples:
            break

        question = sample["question"]
        expected_answer = str(sample["answer"]).strip()

        # Retrieve top documents
        retrieved_docs = retriever.invoke(question)

        # Find ground truth relevant doc IDs from retrieved or corpus match
        relevant_doc_ids = []
        for doc in retrieved_docs:
            doc_id = doc.metadata.get("document_id")
            if expected_answer.lower() in doc.page_content.lower():
                if doc_id not in relevant_doc_ids:
                    relevant_doc_ids.append(doc_id)

        # Record golden sample entry
        record = {
            "id": sample.get("id", i),
            "question": question,
            "expected_answer": expected_answer,
            "relevant_doc_ids": relevant_doc_ids,
        }
        golden_records.append(record)

    # Save as JSON
    with open(path, "w", encoding="utf-8") as f:
        json.dump(golden_records, f, indent=2)

    # Also save as CSV for easy viewing/editing
    df = pd.DataFrame(golden_records)
    df.to_csv(path.with_suffix(".csv"), index=False)

    print(f"Golden dataset created successfully with {len(golden_records)} samples at {path}")
    return golden_records


def load_golden_dataset(file_path="data/golden_dataset.json"):
    """Loads the golden dataset from disk."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
