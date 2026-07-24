from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

from langchain_classic.chains import create_retrieval_chain

from .llm import get_llm


# ---------------------------------------------------
# 1. Manual RAG
# ---------------------------------------------------
def ask_question(retriever, question):

    documents = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in documents
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful AI assistant.

Answer ONLY using the provided context.

If the answer isn't present, say:

"I don't know."

Context:
{context}

Question:
{question}
"""
    )

    llm = get_llm()

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    return answer


# ---------------------------------------------------
# 2. Stuff Documents Chain
# ---------------------------------------------------
def ask_question_with_stuff_chain(retriever, question):

    documents = retriever.invoke(question)

    print("\n")
    print("=" * 80)
    print("Retrieved Documents")
    print("=" * 80)

    for i, doc in enumerate(documents, start=1):
        print(f"\nDocument {i}")
        print(doc.page_content)
        print("\nMetadata:", doc.metadata)
        print("-" * 80)

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful AI assistant.

Answer ONLY using the provided context.

If the answer isn't present, say:

"I don't know."

Context:
{context}

Question:
{input}
"""
    )

    llm = get_llm()

    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
    )

    answer = document_chain.invoke(
        {
            "input": question,
            "context": documents,
        }
    )

    return answer


# ---------------------------------------------------
# 3. Retrieval Chain
# ---------------------------------------------------
def ask_question_with_retrieval_chain(retriever, question):

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful AI assistant.

Answer ONLY using the provided context.

If the answer isn't present, say:

"I don't know."

Context:
{context}

Question:
{input}
"""
    )

    llm = get_llm()

    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
    )

    rag_chain = create_retrieval_chain(
        retriever,
        document_chain,
    )

    response = rag_chain.invoke(
        {
            "input": question
        }
    )

    return response