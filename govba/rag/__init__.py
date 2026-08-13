"""Retrieval and evidence infrastructure for GovBA-GAR."""

from govba.rag.evidence import (
    EvidenceChunk,
    compute_chunk_hash,
    compute_chunk_id,
)
from govba.rag.embeddings import (
    EmbeddingProvider,
    EmbeddingVector,
    cosine_similarity,
    cosine_to_unit_interval,
    embedding_norm,
    normalize_embedding_vector,
)
from govba.rag.evaluation import (
    DEFAULT_RETRIEVAL_CUTOFFS,
    GoldRetrievalCase,
    RetrievalBenchmarkEvaluation,
    RetrievalCaseEvaluation,
    evaluate_benchmark,
    evaluate_ranked_results,
    evaluate_retrieval_case,
    normalize_cutoffs,
)
from govba.rag.lexical import (
    LEXICAL_RETRIEVAL_METHOD,
    LexicalRetriever,
    normalize_lexical_text,
    tokenize_lexical,
)
from govba.rag.models import (
    AuthoritativeSource,
    DocumentType,
    SourceLanguage,
    SourceStatus,
    compute_content_hash,
)
from govba.rag.semantic import (
    SEMANTIC_RETRIEVAL_METHOD,
    SemanticRetriever,
)
from govba.rag.retrieval import (
    RetrievalFilters,
    RetrievalQuery,
    RetrievalResult,
    Retriever,
)

__all__ = [
    "SEMANTIC_RETRIEVAL_METHOD",

    "SemanticRetriever",

    "EmbeddingProvider",

    "EmbeddingVector",

    "cosine_similarity",

    "cosine_to_unit_interval",

    "embedding_norm",

    "normalize_embedding_vector",

    "DEFAULT_RETRIEVAL_CUTOFFS",

    "GoldRetrievalCase",

    "RetrievalBenchmarkEvaluation",

    "RetrievalCaseEvaluation",

    "evaluate_benchmark",

    "evaluate_ranked_results",

    "evaluate_retrieval_case",

    "normalize_cutoffs",

    "tokenize_lexical",
    "normalize_lexical_text",
    "LexicalRetriever",
    "LEXICAL_RETRIEVAL_METHOD",
    "AuthoritativeSource",
    "DocumentType",
    "EvidenceChunk",
    "RetrievalFilters",
    "RetrievalQuery",
    "RetrievalResult",
    "Retriever",
    "SourceLanguage",
    "SourceStatus",
    "compute_chunk_hash",
    "compute_chunk_id",
    "compute_content_hash",
]
