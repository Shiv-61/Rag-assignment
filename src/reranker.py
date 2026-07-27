from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


def build_reranker(hybrid_retriever):

    cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")

    compressor = CrossEncoderReranker(
        model=cross_encoder,
        top_n=5,
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=hybrid_retriever,
    )

    return compression_retriever
