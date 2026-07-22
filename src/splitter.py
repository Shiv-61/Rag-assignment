from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents):
    """
    Split documents into overlapping chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(documents)

    return chunks