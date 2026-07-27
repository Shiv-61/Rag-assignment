from langchain_core.prompts import ChatPromptTemplate

rewrite_prompt = ChatPromptTemplate.from_template("""
You are an expert search assistant.

Rewrite the user's question into a better search query.

Do not answer the question.

Return only the rewritten query.

Question:
{question}
""")


def transform_query(question, llm):

    rewrite_chain = rewrite_prompt | llm

    response = rewrite_chain.invoke({"question": question})

    return response.content
