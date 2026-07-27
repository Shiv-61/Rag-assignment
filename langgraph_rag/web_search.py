from typing import List
from langchain_core.documents import Document
from duckduckgo_search import DDGS


def perform_web_search(query: str, max_results: int = 3) -> List[Document]:
    documents = []
    try:
        ddgs = DDGS()
        results = list(ddgs.text(query, max_results=max_results))
        for idx, res in enumerate(results):
            content = f"{res.get('title', '')}\n{res.get('body', '')}"
            doc = Document(
                page_content=content,
                metadata={
                    "id": f"web_{idx+1}",
                    "source": res.get("href", "DuckDuckGo Web Search"),
                    "title": res.get("title", "Web Result")
                }
            )
            documents.append(doc)
    except Exception as e:
        doc = Document(
            page_content=f"Web search attempted for query '{query}', but search failed: {str(e)}",
            metadata={
                "id": "web_fallback_err",
                "source": "Web Search (Failed)",
                "title": "Search Error"
            }
        )
        documents.append(doc)

    return documents
