"""Controlled LIVE retrieval benchmark for GovBA-GAR.

WARNING:
- Makes real, cost-bearing embedding requests only when
  EMBEDDINGS_ENABLED is explicitly enabled.
- Uses synthetic public-sector text only.
- Never prints API keys or embedding-vector values.
- Not part of routine regression testing.
"""

from __future__ import annotations

import json

from govba.core.settings import read_setting
from govba.rag import (
    AuthoritativeSource,
    DocumentType,
    EvidenceChunk,
    GoldRetrievalCase,
    HybridRetriever,
    LexicalRetriever,
    OpenAIEmbeddingProvider,
    OpenAIEmbeddingProviderError,
    RetrievalFilters,
    RetrievalQuery,
    SemanticRetriever,
    SourceLanguage,
    SourceStatus,
    compare_retrievers,
    get_embedding_provider_status,
)


CUTOFFS = (1, 3, 5)


SYNTHETIC_DOCUMENTS = (
    (
        "DOC-LEAVE",
        "Annual Leave Procedure",
        DocumentType.PROCEDURE,
        (
            "Annual leave requests must be submitted at least "
            "five working days before planned leave begins."
        ),
    ),
    (
        "DOC-PROCUREMENT",
        "Supplier Registration Procedure",
        DocumentType.PROCEDURE,
        (
            "Suppliers must maintain active registration before "
            "participating in government procurement tenders."
        ),
    ),
    (
        "DOC-LICENCE",
        "Electronic Licence Renewal Guideline",
        DocumentType.GUIDELINE,
        (
            "Citizens can renew municipal service licences through "
            "the secure electronic service portal."
        ),
    ),
    (
        "DOC-CONFIDENTIAL",
        "Confidential Correspondence Policy",
        DocumentType.POLICY,
        (
            "Confidential official correspondence must not be "
            "forwarded to personal email accounts."
        ),
    ),
    (
        "DOC-SECURITY",
        "Cybersecurity Incident Procedure",
        DocumentType.PROCEDURE,
        (
            "Employees must report suspected cybersecurity incidents "
            "to the information security team within one hour "
            "of discovery."
        ),
    ),
    (
        "DOC-RECORDS",
        "Personnel Records Policy",
        DocumentType.POLICY,
        (
            "Personnel records must be retained for seven years "
            "after an employee leaves the organization."
        ),
    ),
)


class InMemoryCachingEmbeddingProvider:
    """Process-local embedding cache for controlled benchmarking."""

    def __init__(self, provider):
        self._provider = provider
        self._cache = {}

        self.document_api_batches = 0
        self.query_api_calls = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def embed_documents(self, texts):
        texts = tuple(texts)

        missing = []
        seen_missing = set()

        for text in texts:
            if text in self._cache:
                self.cache_hits += 1
                continue

            if text not in seen_missing:
                missing.append(text)
                seen_missing.add(text)

        if missing:
            vectors = self._provider.embed_documents(
                missing
            )

            self.document_api_batches += 1
            self.cache_misses += len(missing)

            for text, vector in zip(
                missing,
                vectors,
                strict=True,
            ):
                self._cache[text] = tuple(vector)

        return tuple(
            self._cache[text]
            for text in texts
        )

    def embed_query(self, text):
        if text in self._cache:
            self.cache_hits += 1
            return self._cache[text]

        vector = tuple(
            self._provider.embed_query(text)
        )

        self.query_api_calls += 1
        self.cache_misses += 1
        self._cache[text] = vector

        return vector

    @property
    def cached_vector_count(self):
        return len(self._cache)


def make_sources():
    return tuple(
        AuthoritativeSource(
            document_id=document_id,
            title=title,
            issuing_authority=(
                "Synthetic GovBA Test Authority"
            ),
            document_type=document_type,
            language=SourceLanguage.ENGLISH,
            status=SourceStatus.CURRENT,
        )
        for (
            document_id,
            title,
            document_type,
            _,
        ) in SYNTHETIC_DOCUMENTS
    )


def make_chunks():
    return tuple(
        EvidenceChunk(
            document_id=document_id,
            chunk_index=0,
            text=text,
            language=SourceLanguage.ENGLISH,
        )
        for (
            document_id,
            _,
            _,
            text,
        ) in SYNTHETIC_DOCUMENTS
    )


def make_cases(chunks):
    chunk_ids = {
        chunk.document_id: chunk.chunk_id
        for chunk in chunks
    }

    current_only = RetrievalFilters(
        statuses=(
            SourceStatus.CURRENT,
        )
    )

    case_specs = (
        (
            "EN-LEX-LEAVE",
            (
                "How many working days before planned leave "
                "must annual leave requests be submitted?"
            ),
            "DOC-LEAVE",
        ),
        (
            "EN-SEM-LEAVE",
            "How early should staff ask for vacation time?",
            "DOC-LEAVE",
        ),
        (
            "EN-LEX-PROCUREMENT",
            (
                "What must suppliers maintain before "
                "participating in government procurement tenders?"
            ),
            "DOC-PROCUREMENT",
        ),
        (
            "EN-SEM-PROCUREMENT",
            (
                "What prerequisite does a vendor need before "
                "bidding for a public contract?"
            ),
            "DOC-PROCUREMENT",
        ),
        (
            "EN-LEX-CONFIDENTIAL",
            (
                "May confidential official correspondence be "
                "forwarded to personal email accounts?"
            ),
            "DOC-CONFIDENTIAL",
        ),
        (
            "EN-SEM-SECURITY",
            (
                "Who should staff contact after noticing a possible "
                "cyber attack, and how quickly?"
            ),
            "DOC-SECURITY",
        ),
        (
            "EN-LEX-RECORDS",
            (
                "How long must personnel records be retained after "
                "an employee leaves?"
            ),
            "DOC-RECORDS",
        ),
        (
            "EN-SEM-LICENCE",
            (
                "Where can residents extend an existing permit "
                "online?"
            ),
            "DOC-LICENCE",
        ),
    )

    return tuple(
        GoldRetrievalCase(
            case_id=case_id,
            query=RetrievalQuery(
                text=query_text,
                top_k=5,
                filters=current_only,
            ),
            relevant_chunk_ids=(
                chunk_ids[document_id],
            ),
        )
        for (
            case_id,
            query_text,
            document_id,
        ) in case_specs
    )


def print_comparison_summary(
    comparison,
    heading,
):
    print()
    print(heading)
    print("=" * len(heading))

    data = comparison.to_dict()

    print(
        "case_count:",
        data.get("case_count"),
    )
    print(
        "cutoffs:",
        data.get("cutoffs"),
    )

    for run in data.get("runs", []):
        print()
        print(
            f"[{run.get('name', 'Unknown')}]"
        )

        evaluation = dict(
            run.get(
                "evaluation",
                {},
            )
        )

        evaluation.pop(
            "cases",
            None,
        )
        evaluation.pop(
            "case_evaluations",
            None,
        )

        print(
            json.dumps(
                evaluation,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )


def validate_fixture(
    sources,
    chunks,
    cases,
):
    source_ids = [
        source.document_id
        for source in sources
    ]

    chunk_ids = [
        chunk.chunk_id
        for chunk in chunks
    ]

    case_ids = [
        case.case_id
        for case in cases
    ]

    if len(sources) != 6:
        raise RuntimeError(
            "Expected exactly 6 synthetic sources."
        )

    if len(chunks) != 6:
        raise RuntimeError(
            "Expected exactly 6 synthetic chunks."
        )

    if len(cases) != 8:
        raise RuntimeError(
            "Expected exactly 8 gold cases."
        )

    if len(set(source_ids)) != len(source_ids):
        raise RuntimeError(
            "Duplicate source IDs detected."
        )

    if len(set(chunk_ids)) != len(chunk_ids):
        raise RuntimeError(
            "Duplicate chunk IDs detected."
        )

    if len(set(case_ids)) != len(case_ids):
        raise RuntimeError(
            "Duplicate case IDs detected."
        )

    known_chunks = set(chunk_ids)

    for case in cases:
        unknown = (
            set(case.relevant_chunk_ids)
            - known_chunks
        )

        if unknown:
            raise RuntimeError(
                "Gold case references unknown chunk."
            )


def main():
    sources = make_sources()
    chunks = make_chunks()
    cases = make_cases(chunks)

    validate_fixture(
        sources,
        chunks,
        cases,
    )

    print(
        "GovBA-GAR controlled retrieval benchmark"
    )
    print("-" * 48)
    print("synthetic_sources:", len(sources))
    print("synthetic_chunks:", len(chunks))
    print("gold_cases:", len(cases))
    print("raw_vectors_printed: False")
    print("synthetic_data_only: True")

    lexical = LexicalRetriever(
        sources=sources,
        chunks=chunks,
    )

    lexical_comparison = compare_retrievers(
        retrievers={
            "Lexical": lexical,
        },
        cases=cases,
        cutoffs=CUTOFFS,
    )

    print_comparison_summary(
        lexical_comparison,
        "OFFLINE LEXICAL BASELINE",
    )

    status = get_embedding_provider_status()

    print()
    print("Embedding provider status")
    print("-" * 48)

    for key in (
        "provider",
        "configured",
        "enabled",
        "sdk_available",
        "ready",
        "model",
        "mode",
    ):
        print(
            f"{key}: {status[key]}"
        )

    if not status["ready"]:
        print()
        print(
            "LIVE semantic benchmark blocked safely."
        )
        print(
            "No external embedding request was made."
        )
        return 2

    api_key = read_setting(
        "OPENAI_API_KEY"
    )

    if not api_key:
        print(
            "LIVE benchmark blocked: API key unavailable."
        )
        return 2

    provider = OpenAIEmbeddingProvider(
        api_key=api_key,
        model=str(
            status["model"]
        ),
    )

    cached_provider = (
        InMemoryCachingEmbeddingProvider(
            provider
        )
    )

    try:
        semantic = SemanticRetriever(
            sources=sources,
            chunks=chunks,
            embedding_provider=(
                cached_provider
            ),
        )

        hybrid = HybridRetriever(
            retrievers={
                "Lexical": lexical,
                "Semantic": semantic,
            },
            candidate_pool_size=6,
        )

        comparison = compare_retrievers(
            retrievers={
                "Lexical": lexical,
                "Semantic": semantic,
                "Hybrid": hybrid,
            },
            cases=cases,
            cutoffs=CUTOFFS,
        )

    except OpenAIEmbeddingProviderError as exc:
        print()
        print(
            "LIVE retrieval benchmark failed safely."
        )
        print(
            "Error type:",
            type(exc).__name__,
        )
        print(
            "Safe message:",
            str(exc),
        )
        return 1

    print_comparison_summary(
        comparison,
        "LIVE RETRIEVAL COMPARISON",
    )

    print()
    print("Embedding execution statistics")
    print("-" * 48)
    print(
        "embedding_dimension:",
        semantic.embedding_dimension,
    )
    print(
        "document_api_batches:",
        cached_provider.document_api_batches,
    )
    print(
        "query_api_calls:",
        cached_provider.query_api_calls,
    )
    print(
        "cache_hits:",
        cached_provider.cache_hits,
    )
    print(
        "cache_misses:",
        cached_provider.cache_misses,
    )
    print(
        "cached_vector_count:",
        cached_provider.cached_vector_count,
    )
    print(
        "vector_values_printed: False"
    )

    print()
    print(
        "CONTROLLED LIVE RETRIEVAL BENCHMARK PASSED."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
