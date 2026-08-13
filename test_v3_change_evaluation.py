"""Offline tests for GovBA-GAR change-intelligence evaluation."""

from __future__ import annotations

import unittest
from datetime import date

from govba.change.briefing import (
    ChangeBriefingDecision,
)
from govba.change.contract import (
    PolicyChangeImpact,
)
from govba.change.evaluation import (
    CHANGE_INTELLIGENCE_EVALUATION_VERSION,
    ChangeIntelligenceGoldCase,
    evaluate_change_intelligence,
)
from govba.change.version_matching import (
    DocumentVersionMatchDecision,
)
from govba.rag.evidence import EvidenceChunk
from govba.rag.models import (
    AuthoritativeSource,
    DocumentType,
    SourceLanguage,
    SourceStatus,
)


def source(
    document_id,
    *,
    title="Synthetic Policy",
    authority="Synthetic Ministry",
    content_hash,
):
    return AuthoritativeSource(
        document_id=document_id,
        title=title,
        issuing_authority=authority,
        document_type=(
            DocumentType.POLICY
        ),
        language=(
            SourceLanguage.ENGLISH
        ),
        jurisdiction="Jordan",
        effective_from=date(
            2026,
            1,
            1,
        ),
        status=(
            SourceStatus.CURRENT
        ),
        official_source_url=(
            "https://example.gov.jo/"
            f"{document_id}"
        ),
        content_hash=content_hash,
    )


def chunk(
    document_id,
    index,
    text,
):
    return EvidenceChunk(
        document_id=document_id,
        chunk_index=index,
        text=text,
        language=(
            SourceLanguage.ENGLISH
        ),
    )


def allowed_case(
    *,
    case_id="CASE-001",
    old_text="Employees may submit the form.",
    new_text="Employees must submit the form.",
    expected_added=0,
    expected_removed=0,
    expected_modified=1,
    expected_unchanged=0,
    expected_impact=(
        PolicyChangeImpact.MEDIUM
    ),
    expected_briefing=(
        ChangeBriefingDecision.READY
    ),
):
    baseline = source(
        "OLD-" + case_id,
        content_hash="a" * 64,
    )

    candidate = source(
        "NEW-" + case_id,
        content_hash="b" * 64,
    )

    return ChangeIntelligenceGoldCase(
        case_id=case_id,
        baseline_source=baseline,
        candidate_source=candidate,
        baseline_chunks=(
            chunk(
                baseline.document_id,
                0,
                old_text,
            ),
        ),
        candidate_chunks=(
            chunk(
                candidate.document_id,
                0,
                new_text,
            ),
        ),
        expected_match_decision=(
            DocumentVersionMatchDecision.ALLOW
        ),
        expected_added=expected_added,
        expected_removed=expected_removed,
        expected_modified=expected_modified,
        expected_unchanged=expected_unchanged,
        expected_maximum_impact=(
            expected_impact
        ),
        expected_briefing_decision=(
            expected_briefing
        ),
    )


class TestChangeIntelligenceEvaluation(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            CHANGE_INTELLIGENCE_EVALUATION_VERSION,
            "govba-change-intelligence-evaluation-v1",
        )

    def test_allowed_case_runs_end_to_end(self):
        benchmark = (
            evaluate_change_intelligence(
                (
                    allowed_case(),
                )
            )
        )

        result = benchmark.cases[0]

        self.assertTrue(
            result.match_correct
        )

        self.assertTrue(
            result.change_counts_correct
        )

        self.assertTrue(
            result.impact_correct
        )

        self.assertTrue(
            result.briefing_correct
        )

        self.assertTrue(
            result.case_success
        )

    def test_perfect_metrics_are_one(self):
        benchmark = (
            evaluate_change_intelligence(
                (
                    allowed_case(),
                )
            )
        )

        self.assertEqual(
            benchmark.metrics
            .version_match_accuracy,
            1.0,
        )

        self.assertEqual(
            benchmark.metrics
            .change_count_accuracy,
            1.0,
        )

        self.assertEqual(
            benchmark.metrics
            .impact_accuracy,
            1.0,
        )

        self.assertEqual(
            benchmark.metrics
            .briefing_accuracy,
            1.0,
        )

        self.assertEqual(
            benchmark.metrics
            .end_to_end_success_rate,
            1.0,
        )

    def test_critical_change_requires_review(self):
        case = allowed_case(
            case_id="CRITICAL",
            old_text=(
                "A penalty of 50 JOD "
                "shall apply."
            ),
            new_text=(
                "A penalty of 100 JOD "
                "shall apply."
            ),
            expected_impact=(
                PolicyChangeImpact.CRITICAL
            ),
            expected_briefing=(
                ChangeBriefingDecision.REVIEW
            ),
        )

        benchmark = (
            evaluate_change_intelligence(
                (
                    case,
                )
            )
        )

        self.assertTrue(
            benchmark.cases[
                0
            ].case_success
        )

    def test_exact_unchanged_clause(self):
        case = allowed_case(
            case_id="UNCHANGED",
            old_text=(
                "Employees receive annual leave."
            ),
            new_text=(
                "Employees receive annual leave."
            ),
            expected_added=0,
            expected_removed=0,
            expected_modified=0,
            expected_unchanged=1,
            expected_impact=(
                PolicyChangeImpact.INFORMATIONAL
            ),
            expected_briefing=(
                ChangeBriefingDecision.READY
            ),
        )

        benchmark = (
            evaluate_change_intelligence(
                (
                    case,
                )
            )
        )

        self.assertTrue(
            benchmark.cases[
                0
            ].case_success
        )

    def test_rejected_pair_skips_downstream(self):
        baseline = source(
            "OLD-REJECT",
            title="Leave Policy",
            authority="Ministry A",
            content_hash="a" * 64,
        )

        candidate = source(
            "NEW-REJECT",
            title="Cybersecurity Manual",
            authority="Ministry B",
            content_hash="b" * 64,
        )

        case = ChangeIntelligenceGoldCase(
            case_id="REJECT",
            baseline_source=baseline,
            candidate_source=candidate,
            baseline_chunks=(
                chunk(
                    baseline.document_id,
                    0,
                    "Old evidence.",
                ),
            ),
            candidate_chunks=(
                chunk(
                    candidate.document_id,
                    0,
                    "New evidence.",
                ),
            ),
            expected_match_decision=(
                DocumentVersionMatchDecision.REJECT
            ),
        )

        benchmark = (
            evaluate_change_intelligence(
                (
                    case,
                )
            )
        )

        result = benchmark.cases[0]

        self.assertTrue(
            result.match_correct
        )

        self.assertIsNone(
            result.change_counts_correct
        )

        self.assertTrue(
            result.case_success
        )

    def test_mixed_cases_are_aggregated(self):
        good = allowed_case(
            case_id="GOOD"
        )

        baseline = source(
            "OLD-R",
            title="Policy A",
            authority="Authority A",
            content_hash="c" * 64,
        )

        candidate = source(
            "NEW-R",
            title="Policy B",
            authority="Authority B",
            content_hash="d" * 64,
        )

        rejected = ChangeIntelligenceGoldCase(
            case_id="R",
            baseline_source=baseline,
            candidate_source=candidate,
            baseline_chunks=(
                chunk(
                    baseline.document_id,
                    0,
                    "Baseline.",
                ),
            ),
            candidate_chunks=(
                chunk(
                    candidate.document_id,
                    0,
                    "Candidate.",
                ),
            ),
            expected_match_decision=(
                DocumentVersionMatchDecision.REJECT
            ),
        )

        benchmark = (
            evaluate_change_intelligence(
                (
                    good,
                    rejected,
                )
            )
        )

        self.assertEqual(
            benchmark.metrics.case_count,
            2,
        )

        self.assertEqual(
            benchmark.metrics
            .downstream_case_count,
            1,
        )

        self.assertEqual(
            benchmark.metrics
            .end_to_end_success_rate,
            1.0,
        )

    def test_wrong_expected_impact_fails_case(self):
        case = allowed_case(
            case_id="WRONG",
            expected_impact=(
                PolicyChangeImpact.HIGH
            ),
        )

        benchmark = (
            evaluate_change_intelligence(
                (
                    case,
                )
            )
        )

        self.assertFalse(
            benchmark.cases[
                0
            ].impact_correct
        )

        self.assertFalse(
            benchmark.cases[
                0
            ].case_success
        )

    def test_duplicate_case_ids_are_rejected(self):
        case = allowed_case(
            case_id="DUP"
        )

        with self.assertRaises(
            ValueError
        ):
            evaluate_change_intelligence(
                (
                    case,
                    case,
                )
            )

    def test_empty_cases_are_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            evaluate_change_intelligence(
                ()
            )

    def test_allow_case_requires_downstream_labels(self):
        baseline = source(
            "OLD",
            content_hash="a" * 64,
        )

        candidate = source(
            "NEW",
            content_hash="b" * 64,
        )

        with self.assertRaises(
            ValueError
        ):
            ChangeIntelligenceGoldCase(
                case_id="INVALID",
                baseline_source=baseline,
                candidate_source=candidate,
                baseline_chunks=(),
                candidate_chunks=(),
                expected_match_decision=(
                    DocumentVersionMatchDecision.ALLOW
                ),
            )

    def test_rejected_case_disallows_downstream_labels(self):
        baseline = source(
            "OLD",
            content_hash="a" * 64,
        )

        candidate = source(
            "NEW",
            title="Different",
            authority="Different",
            content_hash="b" * 64,
        )

        with self.assertRaises(
            ValueError
        ):
            ChangeIntelligenceGoldCase(
                case_id="INVALID-2",
                baseline_source=baseline,
                candidate_source=candidate,
                baseline_chunks=(),
                candidate_chunks=(),
                expected_match_decision=(
                    DocumentVersionMatchDecision.REJECT
                ),
                expected_added=0,
            )

    def test_benchmark_id_is_deterministic(self):
        case = allowed_case(
            case_id="DET"
        )

        first = (
            evaluate_change_intelligence(
                (
                    case,
                )
            )
        )

        second = (
            evaluate_change_intelligence(
                (
                    case,
                )
            )
        )

        self.assertEqual(
            first.benchmark_id,
            second.benchmark_id,
        )

    def test_latency_does_not_change_identity(self):
        case = allowed_case(
            case_id="LATENCY"
        )

        first = (
            evaluate_change_intelligence(
                (
                    case,
                )
            )
        )

        second = (
            evaluate_change_intelligence(
                (
                    case,
                )
            )
        )

        self.assertEqual(
            first.cases[0].result_id,
            second.cases[0].result_id,
        )

    def test_serialization_contains_no_clause_text(self):
        raw_text = (
            "Employees must submit "
            "the unique synthetic form."
        )

        case = allowed_case(
            case_id="PRIVACY",
            new_text=raw_text,
        )

        benchmark = (
            evaluate_change_intelligence(
                (
                    case,
                )
            )
        )

        data = benchmark.to_dict()

        self.assertNotIn(
            raw_text,
            repr(data),
        )

        self.assertNotIn(
            "https://",
            repr(data),
        )

    def test_latency_is_recorded(self):
        benchmark = (
            evaluate_change_intelligence(
                (
                    allowed_case(),
                )
            )
        )

        self.assertGreaterEqual(
            benchmark.cases[
                0
            ].latency_ms,
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
