"""Offline tests for GovBA-GAR requirements intelligence evaluation."""

from __future__ import annotations

import unittest

from govba.rag.models import (
    SourceLanguage,
)
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
    REQUIREMENTS_EVALUATION_VERSION,
    RequirementsEvaluationGoldCase,
    evaluate_requirements_intelligence,
)
from govba.requirements.governed_brd import (
    GovernedBRDDecision,
)
from govba.requirements.quality import (
    RequirementQualityDecision,
)


def request(
    text,
    *,
    language=SourceLanguage.ENGLISH,
):
    return RequirementsRequest(
        source_text=text,
        language=language,
        project_id="PROJECT-EVAL",
        trace_id="TRACE-EVAL",
        origin=(
            RequirementOrigin.USER_INPUT
        ),
    )


def gold(
    *,
    case_id="FUNCTIONAL",
    text=(
        "The system must allow users "
        "to submit requests."
    ),
    language=SourceLanguage.ENGLISH,
    types=(
        RequirementType.FUNCTIONAL,
    ),
    priorities=(
        RequirementPriority.MUST,
    ),
    acceptance=(
        AcceptanceCriteriaStatus.DRAFT_REVIEW,
    ),
    quality=(
        RequirementQualityDecision.PASS,
    ),
    decision=(
        GovernedBRDDecision.REVIEW
    ),
):
    return RequirementsEvaluationGoldCase(
        case_id=case_id,
        request=request(
            text,
            language=language,
        ),
        expected_requirement_types=(
            types
        ),
        expected_priorities=(
            priorities
        ),
        expected_acceptance_statuses=(
            acceptance
        ),
        expected_quality_decisions=(
            quality
        ),
        expected_governed_decision=(
            decision
        ),
    )


class TestRequirementsEvaluation(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            REQUIREMENTS_EVALUATION_VERSION,
            "govba-requirements-evaluation-v1",
        )

    def test_clean_functional_case_passes(self):
        benchmark = (
            evaluate_requirements_intelligence(
                (
                    gold(),
                )
            )
        )

        self.assertTrue(
            benchmark.cases[
                0
            ].case_success
        )

    def test_security_case_passes(self):
        case = gold(
            case_id="SECURITY",
            text=(
                "The system must use encryption."
            ),
            types=(
                RequirementType.SECURITY,
            ),
        )

        benchmark = (
            evaluate_requirements_intelligence(
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

    def test_reporting_should_case_passes(self):
        case = gold(
            case_id="REPORTING",
            text=(
                "The system should generate "
                "monthly reports."
            ),
            types=(
                RequirementType.REPORTING,
            ),
            priorities=(
                RequirementPriority.SHOULD,
            ),
        )

        benchmark = (
            evaluate_requirements_intelligence(
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

    def test_quality_review_case_passes(self):
        case = gold(
            case_id="QUALITY-REVIEW",
            text=(
                "The system should provide "
                "a user-friendly interface."
            ),
            priorities=(
                RequirementPriority.SHOULD,
            ),
            quality=(
                RequirementQualityDecision.REVIEW,
            ),
        )

        benchmark = (
            evaluate_requirements_intelligence(
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

    def test_quality_rewrite_case_passes(self):
        case = gold(
            case_id="QUALITY-REWRITE",
            text=(
                "The system must provide "
                "high availability."
            ),
            types=(
                RequirementType.NON_FUNCTIONAL,
            ),
            quality=(
                RequirementQualityDecision.REWRITE,
            ),
        )

        benchmark = (
            evaluate_requirements_intelligence(
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

    def test_wont_case_passes(self):
        case = gold(
            case_id="WONT",
            text=(
                "Mobile payments are out of scope."
            ),
            priorities=(
                RequirementPriority.WONT,
            ),
            acceptance=(
                AcceptanceCriteriaStatus
                .NOT_APPLICABLE,
            ),
            decision=(
                GovernedBRDDecision.READY
            ),
        )

        benchmark = (
            evaluate_requirements_intelligence(
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

    def test_empty_case_abstains(self):
        case = gold(
            case_id="EMPTY",
            text=(
                "This paragraph contains general "
                "background information only."
            ),
            types=(),
            priorities=(),
            acceptance=(),
            quality=(),
            decision=(
                GovernedBRDDecision.ABSTAIN
            ),
        )

        benchmark = (
            evaluate_requirements_intelligence(
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

    def test_arabic_case_passes(self):
        case = gold(
            case_id="ARABIC",
            text=(
                "يجب أن يسمح النظام بتقديم الطلبات."
            ),
            language=(
                SourceLanguage.ARABIC
            ),
        )

        benchmark = (
            evaluate_requirements_intelligence(
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

    def test_multiple_requirements_preserve_order(self):
        case = gold(
            case_id="MULTIPLE",
            text=(
                "The system must allow login. "
                "The system should generate reports."
            ),
            types=(
                RequirementType.FUNCTIONAL,
                RequirementType.REPORTING,
            ),
            priorities=(
                RequirementPriority.MUST,
                RequirementPriority.SHOULD,
            ),
            acceptance=(
                AcceptanceCriteriaStatus.DRAFT_REVIEW,
                AcceptanceCriteriaStatus.DRAFT_REVIEW,
            ),
            quality=(
                RequirementQualityDecision.PASS,
                RequirementQualityDecision.PASS,
            ),
        )

        benchmark = (
            evaluate_requirements_intelligence(
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

    def test_wrong_type_gold_fails_case(self):
        case = gold(
            case_id="WRONG-TYPE",
            types=(
                RequirementType.SECURITY,
            ),
        )

        benchmark = (
            evaluate_requirements_intelligence(
                (
                    case,
                )
            )
        )

        self.assertFalse(
            benchmark.cases[
                0
            ].types_correct
        )

        self.assertFalse(
            benchmark.cases[
                0
            ].case_success
        )

    def test_perfect_metrics(self):
        benchmark = (
            evaluate_requirements_intelligence(
                (
                    gold(
                        case_id="A"
                    ),
                    gold(
                        case_id="B",
                        text=(
                            "The system must use "
                            "encryption."
                        ),
                        types=(
                            RequirementType.SECURITY,
                        ),
                    ),
                )
            )
        )

        self.assertEqual(
            benchmark.metrics
            .requirement_count_accuracy,
            1.0,
        )

        self.assertEqual(
            benchmark.metrics
            .type_sequence_accuracy,
            1.0,
        )

        self.assertEqual(
            benchmark.metrics
            .priority_sequence_accuracy,
            1.0,
        )

        self.assertEqual(
            benchmark.metrics
            .acceptance_sequence_accuracy,
            1.0,
        )

        self.assertEqual(
            benchmark.metrics
            .quality_sequence_accuracy,
            1.0,
        )

        self.assertEqual(
            benchmark.metrics
            .governed_decision_accuracy,
            1.0,
        )

        self.assertEqual(
            benchmark.metrics
            .end_to_end_success_rate,
            1.0,
        )

    def test_gold_tuple_lengths_must_match(self):
        with self.assertRaises(
            ValueError
        ):
            gold(
                types=(
                    RequirementType.FUNCTIONAL,
                    RequirementType.REPORTING,
                ),
            )

    def test_duplicate_case_ids_rejected(self):
        case = gold(
            case_id="DUP"
        )

        with self.assertRaises(
            ValueError
        ):
            evaluate_requirements_intelligence(
                (
                    case,
                    case,
                )
            )

    def test_empty_benchmark_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            evaluate_requirements_intelligence(
                ()
            )

    def test_wrong_case_type_rejected(self):
        with self.assertRaises(
            TypeError
        ):
            evaluate_requirements_intelligence(
                (
                    "not-a-gold-case",
                )
            )

    def test_case_hash_is_deterministic(self):
        first = gold(
            case_id="DET"
        )

        second = gold(
            case_id="DET"
        )

        self.assertEqual(
            first.case_hash,
            second.case_hash,
        )

    def test_result_identity_ignores_latency(self):
        case = gold(
            case_id="LATENCY"
        )

        first = (
            evaluate_requirements_intelligence(
                (
                    case,
                )
            )
        )

        second = (
            evaluate_requirements_intelligence(
                (
                    case,
                )
            )
        )

        self.assertEqual(
            first.cases[
                0
            ].result_id,
            second.cases[
                0
            ].result_id,
        )

    def test_benchmark_identity_is_deterministic(self):
        case = gold(
            case_id="BENCHMARK"
        )

        first = (
            evaluate_requirements_intelligence(
                (
                    case,
                )
            )
        )

        second = (
            evaluate_requirements_intelligence(
                (
                    case,
                )
            )
        )

        self.assertEqual(
            first.benchmark_id,
            second.benchmark_id,
        )

    def test_latency_is_recorded(self):
        benchmark = (
            evaluate_requirements_intelligence(
                (
                    gold(),
                )
            )
        )

        self.assertGreaterEqual(
            benchmark.cases[
                0
            ].latency_ms,
            0.0,
        )

        self.assertGreaterEqual(
            benchmark.metrics
            .mean_latency_ms,
            0.0,
        )

    def test_serialization_excludes_raw_source_text(self):
        raw = (
            "The system must support a unique "
            "private government workflow."
        )

        case = gold(
            case_id="PRIVACY",
            text=raw,
        )

        benchmark = (
            evaluate_requirements_intelligence(
                (
                    case,
                )
            )
        )

        data = benchmark.to_dict()

        self.assertNotIn(
            raw,
            repr(data),
        )

        self.assertNotIn(
            "source_text",
            repr(data),
        )

    def test_gold_serialization_is_privacy_safe(self):
        raw = (
            "The system must support a "
            "confidential workflow."
        )

        case = gold(
            case_id="GOLD-PRIVACY",
            text=raw,
        )

        data = case.to_dict()

        self.assertNotIn(
            raw,
            repr(data),
        )

        self.assertNotIn(
            "source_text",
            data,
        )


if __name__ == "__main__":
    unittest.main()
