from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document


class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    retries: int
    max_retries: int
    web_search: bool
    sub_questions: List[str]
    sub_answers: List[str]
    is_multihop: bool
    sources: List[Dict[str, Any]]
    is_grounded: bool
    answers_question: bool
