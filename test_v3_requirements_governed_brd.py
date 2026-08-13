"""Offline tests for GovBA-GAR governed BRD generation."""

from __future__ import annotations

import unittest

from govba.rag.models import (
    SourceLanguage,
)
from govba.requirements.acceptance import (
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
from govba.requirements.governed_brd import (
    GOVERNED_BRD_VERSION,
    GovernedBRDDecision,
    GovernedBRDReason,
    build_governed_brd,
)
from govba.requirements.quality import (
    assess_requirements_quality,
)


def request(
    text,
    *,
    language=SourceLanguage.ENGLISH,
    document_title="",
):
    return RequirementsRequest(
        source_text=text,
        language=language,
        project_id="PROJECT-001",
        trace_id="TRACE-001",
        origin=(
            RequirementOrigin.USER_INPUT
        ),
        document_title=(
            document_title
        ),
    )


def build(
    value,
    extraction=None,
    *,
    title=None,
):
    if extraction is None:
        extraction = (
            extract_requirements(
                value
            )
        )

    acceptance = (
        generate_acceptance_criteria(
            value,
            extraction,
        )
    )

    quality = (
        assess_requirements_quality(
            extraction
        )
    )

    return build_governed_brd(
        value,
        extraction,
        acceptance,
        quality,
        title=title,
    )


def explicit_extraction(
    value,
):
    parsed = parse_requirements_source(
        value
    )

    item = RequirementItem(
        request_id=(
            value.request_id
        ),
        sequence=0,
        requirement_type=(
            RequirementType.FUNCTIONAL
        ),
        priority=(
            RequirementPriority.MUST
        ),
        status=(
            RequirementStatus.DRAFT
        ),
        statement=(
            "The system must allow users "
            "to submit requests."
        ),
        acceptance_criteria=(
            "A valid request can be submitted.",
            "The system returns a request identifier.",
        ),
    )

    return RequirementExtractionResult(
        request_id=(
            value.request_id
        ),
        project_id=(
            value.project_id
        ),
        trace_id=(
            value.trace_id
        ),
        parser_id=(
            parsed.parser_id
        ),
        requirements=(
            item,
        ),
    )


class TestGovernedBRD(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            GOVERNED_BRD_VERSION,
            "govba-governed-brd-v1",
        )

    def test_clean_explicit_requirement_is_ready(self):
        value = request(
            "The system must allow users "
            "to submit requests."
        )

        extraction = explicit_extraction(
            value
        )

        result = build(
            value,
            extraction,
        )

        self.assertEqual(
            result.decision,
            GovernedBRDDecision.READY,
        )

        self.assertIn(
            GovernedBRDReason
            .READY_FOR_HUMAN_REVIEW,
            result.reasons,
        )

    def test_template_criteria_requires_review(self):
        value = request(
            "The system must allow users "
            "to submit requests."
        )

        result = build(
            value
        )

        self.assertEqual(
            result.decision,
            GovernedBRDDecision.REVIEW,
        )

        self.assertIn(
            GovernedBRDReason
            .DRAFT_ACCEPTANCE_CRITERIA,
            result.reasons,
        )

    def test_quality_review_is_propagated(self):
        value = request(
            "The system should provide "
            "a user-friendly interface."
        )

        result = build(
            value
        )

        self.assertTrue(
            result.requires_review
        )

        self.assertIn(
            GovernedBRDReason
            .QUALITY_REVIEW_REQUIRED,
            result.reasons,
        )

    def test_quality_rewrite_is_propagated(self):
        value = request(
            "The system must provide "
            "high availability."
        )

        result = build(
            value
        )

        self.assertTrue(
            result.requires_review
        )

        self.assertIn(
            GovernedBRDReason
            .QUALITY_REWRITE_REQUIRED,
            result.reasons,
        )

    def test_empty_extraction_abstains(self):
        value = request(
            "This text provides general "
            "background information only."
        )

        result = build(
            value
        )

        self.assertTrue(
            result.abstained
        )

        self.assertEqual(
            result.decision,
            GovernedBRDDecision.ABSTAIN,
        )

        self.assertEqual(
            result.reasons,
            (
                GovernedBRDReason
                .NO_REQUIREMENTS,
            ),
        )

        self.assertEqual(
            result.requirement_count,
            0,
        )

    def test_traceability_link_is_created(self):
        value = request(
            "The system must allow login."
        )

        extraction = (
            extract_requirements(
                value
            )
        )

        result = build(
            value,
            extraction,
        )

        self.assertEqual(
            len(
                result.links
            ),
            1,
        )

        self.assertEqual(
            result.links[0]
            .source_requirement_id,
            extraction.requirements[0]
            .requirement_id,
        )

        self.assertEqual(
            result.links[0]
            .brd_requirement_id,
            result.brd.requirements[0]
            .requirement_id,
        )

    def test_enriched_brd_contains_acceptance_criteria(self):
        value = request(
            "The system must allow login."
        )

        result = build(
            value
        )

        self.assertEqual(
            len(
                result.brd.requirements[0]
                .acceptance_criteria
            ),
            1,
        )

    def test_document_title_is_used(self):
        value = request(
            "The system must allow login.",
            document_title=(
                "Citizen Services BRD"
            ),
        )

        result = build(
            value
        )

        self.assertEqual(
            result.brd.title,
            "Citizen Services BRD",
        )

    def test_default_title_is_used(self):
        value = request(
            "The system must allow login."
        )

        result = build(
            value
        )

        self.assertEqual(
            result.brd.title,
            "Business Requirements Document",
        )

    def test_custom_title_overrides_request_title(self):
        value = request(
            "The system must allow login.",
            document_title="Original BRD",
        )

        result = build(
            value,
            title="Governed Citizen Portal BRD",
        )

        self.assertEqual(
            result.brd.title,
            "Governed Citizen Portal BRD",
        )

    def test_result_is_deterministic(self):
        value = request(
            "The system must allow login."
        )

        extraction = (
            extract_requirements(
                value
            )
        )

        first = build(
            value,
            extraction,
        )

        second = build(
            value,
            extraction,
        )

        self.assertEqual(
            first.result_id,
            second.result_id,
        )

        self.assertEqual(
            first.brd.brd_id,
            second.brd.brd_id,
        )

        self.assertEqual(
            first.links[0].link_id,
            second.links[0].link_id,
        )

    def test_default_serialization_is_privacy_safe(self):
        raw = (
            "The system must support a unique "
            "private government workflow."
        )

        value = request(
            raw
        )

        result = build(
            value
        )

        data = result.to_dict()

        self.assertNotIn(
            raw,
            repr(data),
        )

        self.assertNotIn(
            "title",
            data["brd"],
        )

        for requirement in (
            data["brd"]["requirements"]
        ):
            self.assertNotIn(
                "statement",
                requirement,
            )

            self.assertNotIn(
                "acceptance_criteria",
                requirement,
            )

    def test_content_can_be_explicitly_serialized(self):
        value = request(
            "The system must allow login."
        )

        result = build(
            value
        )

        data = result.to_dict(
            include_content=True
        )

        self.assertIn(
            "title",
            data["brd"],
        )

        self.assertIn(
            "statement",
            data["brd"]
            ["requirements"][0],
        )

        self.assertIn(
            "acceptance_criteria",
            data["brd"]
            ["requirements"][0],
        )

    def test_source_text_is_not_serialized_even_with_brd_content(self):
        source = (
            "Background text. "
            "The system must allow login."
        )

        value = request(
            source
        )

        result = build(
            value
        )

        data = result.to_dict(
            include_content=True
        )

        self.assertNotIn(
            "source_text",
            repr(data),
        )

    def test_arabic_brd_is_supported(self):
        value = request(
            "يجب أن يسمح النظام بتقديم الطلبات.",
            language=(
                SourceLanguage.ARABIC
            ),
        )

        result = build(
            value
        )

        self.assertEqual(
            result.requirement_count,
            1,
        )

        self.assertTrue(
            result.requires_review
        )

    def test_mismatched_acceptance_is_rejected(self):
        first = request(
            "The system must allow login."
        )

        second = request(
            "The system must generate reports."
        )

        first_extraction = (
            extract_requirements(
                first
            )
        )

        second_extraction = (
            extract_requirements(
                second
            )
        )

        wrong_acceptance = (
            generate_acceptance_criteria(
                second,
                second_extraction,
            )
        )

        quality = (
            assess_requirements_quality(
                first_extraction
            )
        )

        with self.assertRaises(
            ValueError
        ):
            build_governed_brd(
                first,
                first_extraction,
                wrong_acceptance,
                quality,
            )

    def test_mismatched_quality_is_rejected(self):
        first = request(
            "The system must allow login."
        )

        second = request(
            "The system must generate reports."
        )

        first_extraction = (
            extract_requirements(
                first
            )
        )

        second_extraction = (
            extract_requirements(
                second
            )
        )

        acceptance = (
            generate_acceptance_criteria(
                first,
                first_extraction,
            )
        )

        wrong_quality = (
            assess_requirements_quality(
                second_extraction
            )
        )

        with self.assertRaises(
            ValueError
        ):
            build_governed_brd(
                first,
                first_extraction,
                acceptance,
                wrong_quality,
            )

    def test_human_review_count(self):
        value = request(
            "The system must allow login. "
            "The system should provide "
            "a user-friendly interface."
        )

        result = build(
            value
        )

        self.assertEqual(
            result.human_review_count,
            2,
        )


if __name__ == "__main__":
    unittest.main()
