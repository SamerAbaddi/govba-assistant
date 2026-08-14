from __future__ import annotations

import hashlib
import json
import statistics
import time
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
import sys


# Make the repository root importable when this file is
# executed directly from benchmarks/govba_gar_v1/.
REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from govba.change.briefing import (
    ChangeBriefingDecision,
)
from govba.change.contract import (
    PolicyChangeImpact,
)
from govba.change.evaluation import (
    ChangeIntelligenceGoldCase,
    evaluate_change_intelligence,
)
from govba.change.version_matching import (
    DocumentVersionMatchDecision,
)

from govba.correspondence.briefing import (
    CorrespondenceBriefingDecision,
)
from govba.correspondence.contract import (
    CorrespondenceChannel,
    CorrespondenceDirection,
    CorrespondenceIntent,
    CorrespondencePriority,
    CorrespondenceRequest,
)
from govba.correspondence.evaluation import (
    CorrespondenceEvaluationGoldCase,
    evaluate_correspondence_intelligence,
)

from govba.rag.bilingual_evaluation import (
    BilingualGoldCase,
    evaluate_bilingual_retrieval,
)
from govba.rag.evidence import EvidenceChunk
from govba.rag.language import BilingualLanguage
from govba.rag.models import (
    AuthoritativeSource,
    DocumentType,
    SourceLanguage,
    SourceStatus,
)
from govba.rag.lexical import LexicalRetriever

from govba.requirements.acceptance import (
    AcceptanceCriteriaStatus,
)
from govba.requirements.contract import (
    RequirementOrigin,
    RequirementPriority,
    RequirementType,
    RequirementsRequest,
)
from govba.requirements.evaluation import (
    RequirementsEvaluationGoldCase,
    evaluate_requirements_intelligence,
)
from govba.requirements.governed_brd import (
    GovernedBRDDecision,
)
from govba.requirements.quality import (
    RequirementQualityDecision,
)


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "benchmark_manifest.json"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

UTC = timezone.utc
ASSESSED_AT = datetime(
    2026,
    8,
    14,
    10,
    0,
    tzinfo=UTC,
)


def digest(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def source_language(code):
    return (
        SourceLanguage.ARABIC
        if code == "ar"
        else SourceLanguage.ENGLISH
    )


def bilingual_language(code):
    return (
        BilingualLanguage.ARABIC
        if code == "ar"
        else BilingualLanguage.ENGLISH
    )


def load_cases():
    data = json.loads(
        MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    cases = data["cases"]

    if len(cases) != 120:
        raise RuntimeError(
            "Benchmark must contain exactly 120 cases."
        )

    return data, cases


def percentile(values, p):
    if not values:
        return 0.0

    values = sorted(values)

    index = (
        (len(values) - 1)
        * p
    )

    lower = int(index)
    upper = min(
        lower + 1,
        len(values) - 1,
    )

    fraction = index - lower

    return (
        values[lower]
        + (
            values[upper]
            - values[lower]
        )
        * fraction
    )


# ---------------------------------------------------------
# RAG lexical baseline
# ---------------------------------------------------------

def run_rag(cases):
    rag_cases = [
        case
        for case in cases
        if case["family"] == "rag"
    ]

    sources = []
    chunks = []
    gold_cases = []

    for index, case in enumerate(
        rag_cases
    ):
        document_id = (
            f"BENCH-{case['case_id']}"
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
            language=source_language(
                case["language"]
            ),
            jurisdiction="Jordan",
            effective_from=date(
                2026,
                1,
                1,
            ),
            status=SourceStatus.CURRENT,
            official_source_url=(
                "https://benchmark.gov.jo/"
                f"{case['case_id'].lower()}"
            ),
            content_hash=digest(
                evidence
            ),
        )

        chunk = EvidenceChunk(
            document_id=document_id,
            chunk_index=0,
            text=evidence,
            language=source_language(
                case["language"]
            ),
        )

        sources.append(source)
        chunks.append(chunk)

        gold_cases.append(
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

    retriever = LexicalRetriever(
        sources=tuple(sources),
        chunks=tuple(chunks),
    )

    result = evaluate_bilingual_retrieval(
        retriever,
        tuple(gold_cases),
        allow_cross_lingual=False,
    )

    return {
        "configuration": (
            "B1_lexical_baseline"
        ),
        "case_count": len(rag_cases),
        "arabic": asdict(
            result.arabic
        ),
        "english": asdict(
            result.english
        ),
    }


# ---------------------------------------------------------
# Change intelligence
# ---------------------------------------------------------

def make_change_source(
    case,
    prefix,
    texts,
):
    joined = "\n".join(texts)

    return AuthoritativeSource(
        document_id=(
            f"{prefix}-{case['case_id']}"
        ),
        title="Benchmark Policy",
        issuing_authority=(
            "Synthetic Jordanian Authority"
        ),
        document_type=(
            DocumentType.POLICY
        ),
        language=source_language(
            case["language"]
        ),
        jurisdiction="Jordan",
        effective_from=date(
            2026,
            1,
            1,
        ),
        status=SourceStatus.CURRENT,
        official_source_url=(
            "https://benchmark.gov.jo/"
            f"{prefix.lower()}-"
            f"{case['case_id'].lower()}"
        ),
        content_hash=digest(
            joined
        ),
    )


def make_chunks(
    source,
    texts,
    language,
):
    return tuple(
        EvidenceChunk(
            document_id=source.document_id,
            chunk_index=index,
            text=text,
            language=source_language(
                language
            ),
        )
        for index, text in enumerate(
            texts
        )
    )


def run_change(cases):
    gold_cases = []

    for case in cases:
        if case["family"] != "change":
            continue

        baseline = make_change_source(
            case,
            "OLD",
            case["old"],
        )

        candidate = make_change_source(
            case,
            "NEW",
            case["new"],
        )

        expected = case["expected"]

        gold_cases.append(
            ChangeIntelligenceGoldCase(
                case_id=case["case_id"],
                baseline_source=baseline,
                candidate_source=candidate,
                baseline_chunks=make_chunks(
                    baseline,
                    case["old"],
                    case["language"],
                ),
                candidate_chunks=make_chunks(
                    candidate,
                    case["new"],
                    case["language"],
                ),
                expected_match_decision=(
                    DocumentVersionMatchDecision[
                        expected["match"]
                    ]
                ),
                expected_added=(
                    expected["added"]
                ),
                expected_removed=(
                    expected["removed"]
                ),
                expected_modified=(
                    expected["modified"]
                ),
                expected_unchanged=(
                    expected["unchanged"]
                ),
                expected_maximum_impact=(
                    PolicyChangeImpact[
                        expected["impact"]
                    ]
                ),
                expected_briefing_decision=(
                    ChangeBriefingDecision[
                        expected["briefing"]
                    ]
                ),
            )
        )

    result = evaluate_change_intelligence(
        tuple(gold_cases)
    )

    return asdict(
        result.metrics
    )


# ---------------------------------------------------------
# Correspondence intelligence
# ---------------------------------------------------------

def run_correspondence(cases):
    gold_cases = []

    for case in cases:
        if (
            case["family"]
            != "correspondence"
        ):
            continue

        expected = case["expected"]

        request = CorrespondenceRequest(
            body_text=case["text"],
            language=source_language(
                case["language"]
            ),
            direction=(
                CorrespondenceDirection.INBOUND
            ),
            channel=(
                CorrespondenceChannel.EMAIL
            ),
            received_at=ASSESSED_AT,
            trace_id=(
                f"BENCH-{case['case_id']}"
            ),
        )

        gold_cases.append(
            CorrespondenceEvaluationGoldCase(
                case_id=case["case_id"],
                request=request,
                assessed_at=ASSESSED_AT,
                expected_intent=(
                    CorrespondenceIntent[
                        expected["intent"]
                    ]
                ),
                expected_priority=(
                    CorrespondencePriority[
                        expected["priority"]
                    ]
                ),
                expected_action_count=(
                    expected["actions"]
                ),
                expected_commitment_count=(
                    expected["commitments"]
                ),
                expected_deadline_count=(
                    expected["deadlines"]
                ),
                expected_briefing_decision=(
                    CorrespondenceBriefingDecision[
                        expected["briefing"]
                    ]
                ),
            )
        )

    result = (
        evaluate_correspondence_intelligence(
            tuple(gold_cases)
        )
    )

    return asdict(
        result.metrics
    )


# ---------------------------------------------------------
# Requirements / BRD intelligence
# ---------------------------------------------------------

def run_requirements(cases):
    gold_cases = []

    for case in cases:
        if (
            case["family"]
            != "requirements"
        ):
            continue

        expected = case["expected"]

        request = RequirementsRequest(
            source_text=case["text"],
            language=source_language(
                case["language"]
            ),
            project_id="GOVBA-BENCH",
            trace_id=(
                f"BENCH-{case['case_id']}"
            ),
            origin=(
                RequirementOrigin.USER_INPUT
            ),
        )

        gold_cases.append(
            RequirementsEvaluationGoldCase(
                case_id=case["case_id"],
                request=request,
                expected_requirement_types=tuple(
                    RequirementType[value]
                    for value
                    in expected["types"]
                ),
                expected_priorities=tuple(
                    RequirementPriority[value]
                    for value
                    in expected[
                        "priorities"
                    ]
                ),
                expected_acceptance_statuses=tuple(
                    AcceptanceCriteriaStatus[
                        value
                    ]
                    for value
                    in expected[
                        "acceptance"
                    ]
                ),
                expected_quality_decisions=tuple(
                    RequirementQualityDecision[
                        value
                    ]
                    for value
                    in expected["quality"]
                ),
                expected_governed_decision=(
                    GovernedBRDDecision[
                        expected["decision"]
                    ]
                ),
            )
        )

    result = (
        evaluate_requirements_intelligence(
            tuple(gold_cases)
        )
    )

    return asdict(
        result.metrics
    )


def main():
    manifest, cases = load_cases()

    started = time.perf_counter()

    results = {
        "benchmark_version": (
            manifest["version"]
        ),
        "case_count": 120,
        "rag_lexical": run_rag(
            cases
        ),
        "change": run_change(
            cases
        ),
        "correspondence": (
            run_correspondence(
                cases
            )
        ),
        "requirements": (
            run_requirements(
                cases
            )
        ),
    }

    total_ms = (
        time.perf_counter()
        - started
    ) * 1000.0

    results[
        "offline_total_latency_ms"
    ] = total_ms

    output = (
        RESULTS_DIR
        / "offline_results.json"
    )

    output.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print("results_file:", output)


if __name__ == "__main__":
    main()