from __future__ import annotations

import hashlib
import json
import statistics
import sys
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from govba.core.settings import read_setting

from govba.rag import (
    AuthoritativeSource,
    DocumentType,
    EvidenceChunk,
    HybridRetriever,
    LexicalRetriever,
    OpenAIEmbeddingProvider,
    OpenAIEmbeddingProviderError,
    SemanticRetriever,
    SourceLanguage,
    SourceStatus,
    get_embedding_provider_status,
)

from govba.rag.bilingual_evaluation import (
    BilingualGoldCase,
    BilingualLanguage,
    evaluate_bilingual_retrieval,
)


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "benchmark_manifest.json"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT = RESULTS_DIR / "retrieval_comparison_results.json"


def digest(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def src_language(code: str):
    return (
        SourceLanguage.ARABIC
        if code == "ar"
        else SourceLanguage.ENGLISH
    )


def bilingual_language(code: str):
    return (
        BilingualLanguage.ARABIC
        if code == "ar"
        else BilingualLanguage.ENGLISH
    )


class InMemoryCachingEmbeddingProvider:
    """
    Independent cache/accounting wrapper.

    Each semantic configuration receives its own instance so
    B2 cannot make B3 artificially faster through shared
    query-embedding cache hits.
    """

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
        seen = set()

        for text in texts:
            if text in self._cache:
                self.cache_hits += 1
            elif text not in seen:
                missing.append(text)
                seen.add(text)

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


def jsonable(value):
    if isinstance(value, Enum):
        return value.value

    if is_dataclass(value):
        return jsonable(asdict(value))

    if isinstance(value, dict):
        return {
            str(k): jsonable(v)
            for k, v in value.items()
        }

    if isinstance(value, (tuple, list)):
        return [
            jsonable(v)
            for v in value
        ]

    return value


def percentile(values, p):
    values = sorted(values)

    if not values:
        return None

    if len(values) == 1:
        return values[0]

    position = (
        (len(values) - 1) * p
    )

    low = int(position)
    high = min(
        low + 1,
        len(values) - 1,
    )

    fraction = position - low

    return (
        values[low]
        + (
            values[high]
            - values[low]
        ) * fraction
    )


def load_fixture():
    manifest = json.loads(
        MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    rag_cases = [
        case
        for case in manifest["cases"]
        if case["family"] == "rag"
    ]

    if len(rag_cases) != 30:
        raise RuntimeError(
            "Expected exactly 30 RAG benchmark cases."
        )

    sources = []
    chunks = []
    gold = []

    for case in rag_cases:
        document_id = (
            f"BENCH-{case['case_id']}"
        )

        language = src_language(
            case["language"]
        )

        evidence = case["evidence"]

        source = AuthoritativeSource(
            document_id=document_id,
            title=(
                f"Benchmark Source "
                f"{case['case_id']}"
            ),
            issuing_authority=(
                "Synthetic Jordanian Authority"
            ),
            document_type=(
                DocumentType.GUIDELINE
            ),
            language=language,
            jurisdiction="Jordan",
            status=SourceStatus.CURRENT,
            official_source_url=(
                "https://benchmark.gov.jo/"
                f"{case['case_id'].lower()}"
            ),
            content_hash=digest(evidence),
        )

        chunk = EvidenceChunk(
            document_id=document_id,
            chunk_index=0,
            text=evidence,
            language=language,
        )

        sources.append(source)
        chunks.append(chunk)

        gold.append(
            BilingualGoldCase(
                case_id=case["case_id"],
                pair_id=case["pair_id"],
                query_text=case["query"],
                query_language=(
                    bilingual_language(
                        case["language"]
                    )
                ),
                relevant_chunk_ids=(
                    chunk.chunk_id,
                ),
                top_k=5,
            )
        )

    return (
        tuple(sources),
        tuple(chunks),
        tuple(gold),
    )


def latency_summary(result):
    latencies = [
        float(case.latency_ms)
        for case in result.cases
        if getattr(
            case,
            "latency_ms",
            None,
        ) is not None
    ]

    if not latencies:
        return {}

    return {
        "mean_ms": statistics.mean(
            latencies
        ),
        "median_ms": statistics.median(
            latencies
        ),
        "p95_ms": percentile(
            latencies,
            0.95,
        ),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
    }


def summarize(
    label,
    result,
    provider=None,
):
    payload = {
        "configuration": label,
        "case_count": len(
            result.cases
        ),
        "arabic": jsonable(
            result.arabic
        ),
        "english": jsonable(
            result.english
        ),
        "overall_latency": (
            latency_summary(result)
        ),
        "cases": jsonable(
            result.cases
        ),
    }

    if provider is not None:
        payload[
            "embedding_execution"
        ] = {
            "document_api_batches":
                provider.document_api_batches,
            "query_api_calls":
                provider.query_api_calls,
            "cache_hits":
                provider.cache_hits,
            "cache_misses":
                provider.cache_misses,
            "cached_vector_count":
                provider.cached_vector_count,
        }

    return payload


def main():
    status = get_embedding_provider_status()

    print("Embedding status:")
    print(
        json.dumps(
            status,
            indent=2,
            default=str,
        )
    )

    if not status.get("ready"):
        raise SystemExit(
            "Embedding provider is not ready."
        )

    api_key = read_setting(
        "OPENAI_API_KEY"
    )

    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY unavailable."
        )

    sources, chunks, cases = (
        load_fixture()
    )

    print()
    print("Research retrieval comparison")
    print("-" * 52)
    print("cases:", len(cases))
    print("arabic_cases:", 15)
    print("english_cases:", 15)
    print(
        "embedding_model:",
        status["model"],
    )
    print(
        "raw_vectors_printed:",
        False,
    )

    # -------------------------------------------------
    # B1: lexical baseline
    # -------------------------------------------------

    lexical = LexicalRetriever(
        sources=sources,
        chunks=chunks,
    )

    b1 = evaluate_bilingual_retrieval(
        lexical,
        cases,
        allow_cross_lingual=False,
        candidate_pool_size=30,
    )

    # -------------------------------------------------
    # B2: semantic retrieval
    #
    # Independent provider/cache prevents B2 query
    # embeddings from being reused by B3.
    # -------------------------------------------------

    b2_base_provider = (
        OpenAIEmbeddingProvider(
            api_key=api_key,
            model=str(
                status["model"]
            ),
        )
    )

    b2_provider = (
        InMemoryCachingEmbeddingProvider(
            b2_base_provider
        )
    )

    try:
        semantic_b2 = SemanticRetriever(
            sources=sources,
            chunks=chunks,
            embedding_provider=(
                b2_provider
            ),
        )

        b2 = (
            evaluate_bilingual_retrieval(
                semantic_b2,
                cases,
                allow_cross_lingual=False,
                candidate_pool_size=30,
            )
        )

    except OpenAIEmbeddingProviderError as exc:
        raise SystemExit(
            "B2 semantic retrieval failed safely: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    # -------------------------------------------------
    # B3: GovBA-GAR hybrid retrieval
    #
    # Separate semantic index/cache to make latency
    # comparison independent from B2.
    # Equal weights are pre-specified.
    # -------------------------------------------------

    b3_base_provider = (
        OpenAIEmbeddingProvider(
            api_key=api_key,
            model=str(
                status["model"]
            ),
        )
    )

    b3_provider = (
        InMemoryCachingEmbeddingProvider(
            b3_base_provider
        )
    )

    try:
        semantic_b3 = SemanticRetriever(
            sources=sources,
            chunks=chunks,
            embedding_provider=(
                b3_provider
            ),
        )

        hybrid = HybridRetriever(
            retrievers={
                "Lexical": lexical,
                "Semantic": semantic_b3,
            },
            weights={
                "Lexical": 1.0,
                "Semantic": 1.0,
            },
            candidate_pool_size=30,
        )

        b3 = (
            evaluate_bilingual_retrieval(
                hybrid,
                cases,
                allow_cross_lingual=False,
                candidate_pool_size=30,
            )
        )

    except OpenAIEmbeddingProviderError as exc:
        raise SystemExit(
            "B3 hybrid retrieval failed safely: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    results = {
        "benchmark_version":
            "govba-gar-retrieval-comparison-v1",
        "source_manifest":
            "govba-gar-benchmark-v1",
        "case_count": 30,
        "language_balance": {
            "arabic": 15,
            "english": 15,
        },
        "embedding_model":
            status["model"],
        "cross_lingual": False,
        "hybrid_weights": {
            "Lexical": 1.0,
            "Semantic": 1.0,
        },
        "B1": summarize(
            "B1_lexical",
            b1,
        ),
        "B2": summarize(
            "B2_semantic",
            b2,
            b2_provider,
        ),
        "B3": summarize(
            "B3_govba_gar_hybrid",
            b3,
            b3_provider,
        ),
    }

    OUTPUT.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print()
    print(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print("results_file:", OUTPUT)


if __name__ == "__main__":
    main()