"""Offline tests for GovBA-GAR acceptance-criteria intelligence."""

from __future__ import annotations

import unittest

from govba.rag.models import SourceLanguage
from govba.requirements.acceptance import (
    ACCEPTANCE_CRITERIA_VERSION,
    AcceptanceCriteriaStatus,
    AcceptanceCriterionKind,
    AcceptanceCriterionOrigin,
    enrich_requirements_with_acceptance_criteria,
    generate_acceptance_criteria,
)
from govba.requirements.contract import (
    RequirementItem,
    RequirementOrigin,
    RequirementPriority,
    RequirementStatus,
    RequirementType,
    RequirementsRequest,
)
from govba.requirements.extraction import (
    RequirementExtractionResult,
    extract_requirements,
    parse_requirements_source,
)


def request(
    text,
    *,
    language=SourceLanguage.ENGLISH,
):
    return RequirementsRequest(
        source_text=text,
        language=language,
        project_id="PROJECT-001",
        trace_id="TRACE-001",
        origin=RequirementOrigin.USER_INPUT,
    )


def extraction_with_explicit_criteria():
    value = request(
        "The system must allow users to submit requests."
    )

    parsed = parse_requirements_source(
        value
    )

    item = RequirementItem(
        request_id=value.request_id,
        sequence=0,
        requirement_type=RequirementType.FUNCTIONAL,
        priority=RequirementPriority.MUST,
        status=RequirementStatus.DRAFT,
        statement=(
            "The system must allow users "
            "to submit requests."
        ),
        acceptance_criteria=(
            "A valid request is accepted.",
            "A request identifier is returned.",
        ),
    )

    extraction = RequirementExtractionResult(
        request_id=value.request_id,
        project_id=value.project_id,
        trace_id=value.trace_id,
        parser_id=parsed.parser_id,
        requirements=(
            item,
        ),
    )

    return value, extraction


class TestAcceptanceCriteria(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            ACCEPTANCE_CRITERIA_VERSION,
            "govba-acceptance-criteria-v1",
        )

    def test_functional_draft_is_generated(self):
        value = request(
            "The system must allow users to submit requests."
        )

        extraction = extract_requirements(
            value
        )

        result = generate_acceptance_criteria(
            value,
            extraction,
        )

        assessment = result.assessments[0]

        self.assertEqual(
            assessment.status,
            AcceptanceCriteriaStatus.DRAFT_REVIEW,
        )

        self.assertEqual(
            assessment.criteria[0].kind,
            AcceptanceCriterionKind.FUNCTIONAL_BEHAVIOR,
        )

        self.assertTrue(
            assessment.requires_human_review
        )

    def test_security_criterion_kind(self):
        value = request(
            "The system must use encryption."
        )

        extraction = extract_requirements(
            value
        )

        result = generate_acceptance_criteria(
            value,
            extraction,
        )

        self.assertEqual(
            result.assessments[0]
            .criteria[0].kind,
            AcceptanceCriterionKind.SECURITY_CONTROL,
        )

    def test_explicit_criteria_are_preserved(self):
        value, extraction = (
            extraction_with_explicit_criteria()
        )

        result = generate_acceptance_criteria(
            value,
            extraction,
        )

        assessment = result.assessments[0]

        self.assertEqual(
            assessment.status,
            AcceptanceCriteriaStatus.EXPLICIT,
        )

        self.assertEqual(
            len(
                assessment.criteria
            ),
            2,
        )

        self.assertEqual(
            assessment.criteria[0].origin,
            AcceptanceCriterionOrigin.EXPLICIT,
        )

        self.assertFalse(
            assessment.requires_human_review
        )

    def test_wont_requirement_is_not_applicable(self):
        value = request(
            "Mobile payments are out of scope."
        )

        extraction = extract_requirements(
            value
        )

        result = generate_acceptance_criteria(
            value,
            extraction,
        )

        self.assertEqual(
            result.assessments[0].status,
            AcceptanceCriteriaStatus.NOT_APPLICABLE,
        )

        self.assertEqual(
            result.assessments[0].criteria,
            (),
        )

    def test_arabic_template_is_generated(self):
        value = request(
            "يجب أن يسمح النظام بتقديم الطلبات.",
            language=SourceLanguage.ARABIC,
        )

        extraction = extract_requirements(
            value
        )

        result = generate_acceptance_criteria(
            value,
            extraction,
        )

        text = (
            result.assessments[0]
            .criteria[0]
            .criterion_text
        )

        self.assertIn(
            "التحقق",
            text,
        )

    def test_generated_criterion_requires_review(self):
        value = request(
            "The system must generate reports."
        )

        extraction = extract_requirements(
            value
        )

        result = generate_acceptance_criteria(
            value,
            extraction,
        )

        criterion = (
            result.assessments[0]
            .criteria[0]
        )

        self.assertEqual(
            criterion.origin,
            AcceptanceCriterionOrigin.TEMPLATE_DERIVED,
        )

        self.assertTrue(
            criterion.requires_human_review
        )

    def test_generation_is_deterministic(self):
        value = request(
            "The system must allow login."
        )

        extraction = extract_requirements(
            value
        )

        first = generate_acceptance_criteria(
            value,
            extraction,
        )

        second = generate_acceptance_criteria(
            value,
            extraction,
        )

        self.assertEqual(
            first.result_id,
            second.result_id,
        )

        self.assertEqual(
            first.assessments[0]
            .criteria[0]
            .criterion_id,
            second.assessments[0]
            .criteria[0]
            .criterion_id,
        )

    def test_default_serialization_excludes_criterion_text(self):
        value = request(
            "The system must allow login."
        )

        extraction = extract_requirements(
            value
        )

        result = generate_acceptance_criteria(
            value,
            extraction,
        )

        raw = (
            result.assessments[0]
            .criteria[0]
            .criterion_text
        )

        data = result.to_dict()

        self.assertNotIn(
            raw,
            repr(data),
        )

        self.assertNotIn(
            "criterion_text",
            repr(data),
        )

    def test_enrichment_adds_criteria(self):
        value = request(
            "The system must allow login."
        )

        extraction = extract_requirements(
            value
        )

        result = generate_acceptance_criteria(
            value,
            extraction,
        )

        enriched = (
            enrich_requirements_with_acceptance_criteria(
                extraction,
                result,
            )
        )

        self.assertEqual(
            len(
                enriched.requirements[0]
                .acceptance_criteria
            ),
            1,
        )

    def test_enrichment_preserves_requirement_metadata(self):
        value = request(
            "The system must use encryption."
        )

        extraction = extract_requirements(
            value
        )

        result = generate_acceptance_criteria(
            value,
            extraction,
        )

        enriched = (
            enrich_requirements_with_acceptance_criteria(
                extraction,
                result,
            )
        )

        original = extraction.requirements[0]
        updated = enriched.requirements[0]

        self.assertEqual(
            updated.requirement_type,
            original.requirement_type,
        )

        self.assertEqual(
            updated.priority,
            original.priority,
        )

        self.assertEqual(
            updated.statement,
            original.statement,
        )

    def test_mismatched_request_is_rejected(self):
        first = request(
            "The system must allow login."
        )

        second = request(
            "The system must generate reports."
        )

        extraction = extract_requirements(
            second
        )

        with self.assertRaises(
            ValueError
        ):
            generate_acceptance_criteria(
                first,
                extraction,
            )

    def test_result_counts(self):
        value = request(
            "The system must allow login. "
            "The system should generate reports. "
            "Mobile payments are out of scope."
        )

        extraction = extract_requirements(
            value
        )

        result = generate_acceptance_criteria(
            value,
            extraction,
        )

        self.assertEqual(
            len(
                result.assessments
            ),
            3,
        )

        self.assertEqual(
            result.criterion_count,
            2,
        )

        self.assertEqual(
            result.review_count,
            2,
        )

        self.assertEqual(
            result.not_applicable_count,
            1,
        )


if __name__ == "__main__":
    unittest.main()
