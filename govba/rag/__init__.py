"""Retrieval and evidence infrastructure for GovBA-GAR."""

from govba.rag.evidence import (
    EvidenceChunk,
    compute_chunk_hash,
    compute_chunk_id,
)
from govba.rag.benchmarking import (
    BENCHMARK_COMPARISON_SCHEMA_VERSION,
    RetrievalBenchmarkComparison,
    RetrieverBenchmarkRun,
    compare_retrievers,
)
from govba.rag.chunking import (
    CHUNKING_VERSION,
    DEFAULT_OVERLAP_CHARACTERS,
    DEFAULT_TARGET_CHARACTERS,
    ChunkingConfig,
    chunk_extracted_document,
)
from govba.rag.corpus import (
    CORPUS_MANIFEST_SCHEMA_VERSION,
    CorpusEntry,
    CorpusManifest,
    build_ingestion_request,
)
from govba.rag.corpus_store import (
    CORPUS_STORE_VERSION,
    CorpusManifestError,
    canonical_manifest_json,
    compute_manifest_hash,
    manifest_envelope,
    read_manifest,
    write_manifest,
)
from govba.rag.corpus_validation import (
    CORPUS_VALIDATION_VERSION,
    CorpusEntryValidation,
    CorpusValidationReport,
    compute_file_sha256,
    validate_corpus,
)
from govba.rag.docx_extractor import (
    DOCX_EXTRACTION_VERSION,
    DOCXExtractionError,
    DOCXExtractor,
    PythonDocxExtractor,
)
from govba.rag.embedding_config import (
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    EmbeddingProviderStatus,
    get_embedding_provider_status,
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
from govba.rag.hybrid import (
    DEFAULT_HYBRID_CANDIDATE_POOL,
    DEFAULT_RRF_K,
    HYBRID_RETRIEVAL_METHOD,
    HybridRetriever,
)
from govba.rag.evidence_card import (
    EVIDENCE_CARD_SCHEMA_VERSION,
    EvidenceCard,
    EvidenceTemporalState,
    EvidenceVerificationState,
)
from govba.rag.evidence_card_builder import (
    EVIDENCE_CARD_BUILDER_VERSION,
    EvidenceCardBuilder,
    build_evidence_card,
    build_evidence_cards,
)
from govba.rag.ingestion import (
    DocumentExtractor,
    DocumentFormat,
    ExtractedBlock,
    ExtractedDocument,
    IngestionRequest,
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
from govba.rag.normalization import (
    TEXT_NORMALIZATION_VERSION,
    is_normalized_ingested_text,
    normalize_ingested_text,
)
from govba.rag.openai_embeddings import (
    DEFAULT_OPENAI_EMBEDDING_BATCH_SIZE,
    OpenAIEmbeddingProvider,
    OpenAIEmbeddingProviderError,
)
from govba.rag.semantic import (
    SEMANTIC_RETRIEVAL_METHOD,
    SemanticRetriever,
)
from govba.rag.pdf_extractor import (
    PDF_EXTRACTION_VERSION,
    PDFExtractionError,
    PDFExtractor,
    PyMuPDFExtractor,
)
from govba.rag.pipeline import (
    INGESTION_PIPELINE_VERSION,
    IngestionPipeline,
    IngestionPipelineError,
    IngestionPipelineResult,
)
from govba.rag.retrieval import (
    RetrievalFilters,
    RetrievalQuery,
    RetrievalResult,
    Retriever,
)

__all__ = [
    "EVIDENCE_CARD_BUILDER_VERSION",

    "EvidenceCardBuilder",

    "build_evidence_card",

    "build_evidence_cards",

    "EVIDENCE_CARD_SCHEMA_VERSION",

    "EvidenceCard",

    "EvidenceTemporalState",

    "EvidenceVerificationState",

    "CORPUS_VALIDATION_VERSION",

    "CorpusEntryValidation",

    "CorpusValidationReport",

    "compute_file_sha256",

    "validate_corpus",

    "CORPUS_STORE_VERSION",

    "CorpusManifestError",

    "canonical_manifest_json",

    "compute_manifest_hash",

    "manifest_envelope",

    "read_manifest",

    "write_manifest",

    "CORPUS_MANIFEST_SCHEMA_VERSION",

    "CorpusEntry",

    "CorpusManifest",

    "build_ingestion_request",

    "INGESTION_PIPELINE_VERSION",

    "IngestionPipeline",

    "IngestionPipelineError",

    "IngestionPipelineResult",

    "DOCX_EXTRACTION_VERSION",

    "DOCXExtractionError",

    "DOCXExtractor",

    "PythonDocxExtractor",

    "PDF_EXTRACTION_VERSION",

    "PDFExtractionError",

    "PDFExtractor",

    "PyMuPDFExtractor",

    "CHUNKING_VERSION",

    "DEFAULT_OVERLAP_CHARACTERS",

    "DEFAULT_TARGET_CHARACTERS",

    "ChunkingConfig",

    "chunk_extracted_document",

    "TEXT_NORMALIZATION_VERSION",

    "is_normalized_ingested_text",

    "normalize_ingested_text",

    "DocumentExtractor",

    "DocumentFormat",

    "ExtractedBlock",

    "ExtractedDocument",

    "IngestionRequest",

    "EmbeddingProviderStatus",

    "get_embedding_provider_status",

    "DEFAULT_OPENAI_EMBEDDING_BATCH_SIZE",

    "DEFAULT_OPENAI_EMBEDDING_MODEL",

    "OpenAIEmbeddingProvider",

    "OpenAIEmbeddingProviderError",

    "DEFAULT_HYBRID_CANDIDATE_POOL",

    "DEFAULT_RRF_K",

    "HYBRID_RETRIEVAL_METHOD",

    "HybridRetriever",

    "BENCHMARK_COMPARISON_SCHEMA_VERSION",

    "RetrievalBenchmarkComparison",

    "RetrieverBenchmarkRun",

    "compare_retrievers",

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
