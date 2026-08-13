"""Offline tests for GovBA-GAR requirement extraction."""

from __future__ import annotations

import unittest

from govba.rag.models import (
    SourceLanguage,
)
from govba.requirements.contract import (
    RequirementOrigin,
    RequirementPriority,
    RequirementType,
    RequirementsRequest,
)
from govba.requirements.extraction import (
    REQUIREMENT_EXTRACTION_VERSION,
    build_extracted_brd,
    classify_requirement_type,
    detect_requirement_priority,
    extract_requirements,
    parse_requirements_source,
)


def request(
    text,
    *,
    language=SourceLanguage.ENGLISH,
    source_reference="SRC-001",
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
            "Digital Services BRD"
        ),
        source_reference=(
            source_reference
        ),
    )


class TestRequirementsParsing(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            REQUIREMENT_EXTRACTION_VERSION,
            "govba-requirement-extraction-v1",
        )

    def test_multiple_units_are_parsed(self):
        value = request(
            "The system must allow login.\n"
            "The system should provide reports."
        )

        parsed = parse_requirements_source(
            value
        )

        self.assertEqual(
            len(parsed.units),
            2,
        )

    def test_list_marker_is_removed(self):
        value = request(
            "- The system must allow login."
        )

        parsed = parse_requirements_source(
            value
        )

        self.assertEqual(
            parsed.units[0],
            "The system must allow login.",
        )

    def test_parser_is_deterministic(self):
        value = request(
            "The system must allow login."
        )

        first = parse_requirements_source(
            value
        )

        second = parse_requirements_source(
            value
        )

        self.assertEqual(
            first.parser_id,
            second.parser_id,
        )

    def test_default_serialization_excludes_text(self):
        raw = (
            "The system must support "
            "a unique private workflow."
        )

        parsed = parse_requirements_source(
            request(raw)
        )

        data = parsed.to_dict()

        self.assertNotIn(
            raw,
            repr(data),
        )

        self.assertNotIn(
            "normalized_text",
            data,
        )


class TestRequirementPriorityDetection(
    unittest.TestCase
):
    def test_must_priority(self):
        self.assertEqual(
            detect_requirement_priority(
                "The system must allow login."
            ),
            RequirementPriority.MUST,
        )

    def test_shall_priority(self):
        self.assertEqual(
            detect_requirement_priority(
                "The system shall store the request."
            ),
            RequirementPriority.MUST,
        )

    def test_should_priority(self):
        self.assertEqual(
            detect_requirement_priority(
                "The system should provide exports."
            ),
            RequirementPriority.SHOULD,
        )

    def test_could_priority(self):
        self.assertEqual(
            detect_requirement_priority(
                "The system could provide themes."
            ),
            RequirementPriority.COULD,
        )

    def test_out_of_scope_is_wont(self):
        self.assertEqual(
            detect_requirement_priority(
                "Mobile payments are out of scope."
            ),
            RequirementPriority.WONT,
        )

    def test_must_not_remains_must(self):
        self.assertEqual(
            detect_requirement_priority(
                "The system must not expose passwords."
            ),
            RequirementPriority.MUST,
        )

    def test_non_requirement_returns_none(self):
        self.assertIsNone(
            detect_requirement_priority(
                "The office opens at eight."
            )
        )

    def test_arabic_must(self):
        self.assertEqual(
            detect_requirement_priority(
                "يجب أن يسمح النظام بتقديم الطلب."
            ),
            RequirementPriority.MUST,
        )

    def test_arabic_should(self):
        self.assertEqual(
            detect_requirement_priority(
                "ينبغي أن يوفر النظام تقارير دورية."
            ),
            RequirementPriority.SHOULD,
        )


class TestRequirementTypeClassification(
    unittest.TestCase
):
    def test_security_type(self):
        self.assertEqual(
            classify_requirement_type(
                "The system must use multi-factor authentication."
            ),
            RequirementType.SECURITY,
        )

    def test_integration_type(self):
        self.assertEqual(
            classify_requirement_type(
                "The system must integrate with the payment API."
            ),
            RequirementType.INTEGRATION,
        )

    def test_reporting_type(self):
        self.assertEqual(
            classify_requirement_type(
                "The system shall generate monthly reports."
            ),
            RequirementType.REPORTING,
        )

    def test_data_type(self):
        self.assertEqual(
            classify_requirement_type(
                "The system must retain customer data."
            ),
            RequirementType.DATA,
        )

    def test_non_functional_type(self):
        self.assertEqual(
            classify_requirement_type(
                "The system must achieve 99.9 percent availability."
            ),
            RequirementType.NON_FUNCTIONAL,
        )

    def test_compliance_type(self):
        self.assertEqual(
            classify_requirement_type(
                "The system must comply with the applicable regulation."
            ),
            RequirementType.COMPLIANCE,
        )

    def test_default_type_is_functional(self):
        self.assertEqual(
            classify_requirement_type(
                "The system must allow users to submit requests."
            ),
            RequirementType.FUNCTIONAL,
        )

    def test_arabic_security_type(self):
        self.assertEqual(
            classify_requirement_type(
                "يجب أن يستخدم النظام التشفير لحماية البيانات."
            ),
            RequirementType.SECURITY,
        )


class TestRequirementExtraction(
    unittest.TestCase
):
    def test_explicit_requirements_are_extracted(self):
        value = request(
            "The system must allow login. "
            "The system should generate reports."
        )

        result = extract_requirements(
            value
        )

        self.assertEqual(
            result.requirement_count,
            2,
        )

    def test_non_requirement_text_is_ignored(self):
        value = request(
            "The project started in January. "
            "The system must allow login."
        )

        result = extract_requirements(
            value
        )

        self.assertEqual(
            result.requirement_count,
            1,
        )

    def test_types_and_priorities_are_preserved(self):
        value = request(
            "The system must use encryption. "
            "The system should generate reports."
        )

        result = extract_requirements(
            value
        )

        self.assertEqual(
            result.requirements[
                0
            ].requirement_type,
            RequirementType.SECURITY,
        )

        self.assertEqual(
            result.requirements[
                0
            ].priority,
            RequirementPriority.MUST,
        )

        self.assertEqual(
            result.requirements[
                1
            ].requirement_type,
            RequirementType.REPORTING,
        )

    def test_sequences_are_contiguous(self):
        value = request(
            "The system must allow login. "
            "General background statement. "
            "The system should generate reports."
        )

        result = extract_requirements(
            value
        )

        self.assertEqual(
            tuple(
                requirement.sequence
                for requirement
                in result.requirements
            ),
            (
                0,
                1,
            ),
        )

    def test_duplicate_requirement_is_removed(self):
        value = request(
            "The system must allow login.\n"
            "The system must allow login."
        )

        result = extract_requirements(
            value
        )

        self.assertEqual(
            result.requirement_count,
            1,
        )

    def test_source_reference_is_propagated(self):
        value = request(
            "The system must allow login.",
            source_reference="POLICY-17",
        )

        result = extract_requirements(
            value
        )

        self.assertEqual(
            result.requirements[
                0
            ].source_references,
            (
                "POLICY-17",
            ),
        )

    def test_empty_source_reference_is_supported(self):
        value = request(
            "The system must allow login.",
            source_reference="",
        )

        result = extract_requirements(
            value
        )

        self.assertEqual(
            result.requirements[
                0
            ].source_references,
            (),
        )

    def test_arabic_requirement_is_extracted(self):
        value = request(
            "يجب أن يسمح النظام بتقديم الطلبات.",
            language=(
                SourceLanguage.ARABIC
            ),
        )

        result = extract_requirements(
            value
        )

        self.assertEqual(
            result.requirement_count,
            1,
        )

        self.assertEqual(
            result.requirements[
                0
            ].priority,
            RequirementPriority.MUST,
        )

    def test_extraction_is_deterministic(self):
        value = request(
            "The system must allow login."
        )

        first = extract_requirements(
            value
        )

        second = extract_requirements(
            value
        )

        self.assertEqual(
            first.extraction_id,
            second.extraction_id,
        )

        self.assertEqual(
            first.requirements[
                0
            ].requirement_id,
            second.requirements[
                0
            ].requirement_id,
        )

    def test_serialization_excludes_requirement_text(self):
        raw = (
            "The system must support a unique "
            "private government workflow."
        )

        result = extract_requirements(
            request(raw)
        )

        data = result.to_dict()

        self.assertNotIn(
            raw,
            repr(data),
        )

        self.assertNotIn(
            "statement",
            data[
                "requirements"
            ][0],
        )

    def test_preparsed_source_is_supported(self):
        value = request(
            "The system must allow login."
        )

        parsed = parse_requirements_source(
            value
        )

        result = extract_requirements(
            value,
            parsed=parsed,
        )

        self.assertEqual(
            result.parser_id,
            parsed.parser_id,
        )

    def test_mismatched_parsed_source_is_rejected(self):
        first = request(
            "The system must allow login."
        )

        second = request(
            "The system must generate reports."
        )

        parsed = parse_requirements_source(
            second
        )

        with self.assertRaises(
            ValueError
        ):
            extract_requirements(
                first,
                parsed=parsed,
            )


class TestExtractedBRD(
    unittest.TestCase
):
    def test_brd_is_built_from_extraction(self):
        value = request(
            "The system must allow login. "
            "The system should generate reports."
        )

        extraction = extract_requirements(
            value
        )

        brd = build_extracted_brd(
            value,
            extraction,
        )

        self.assertEqual(
            brd.requirement_count,
            2,
        )

        self.assertEqual(
            brd.title,
            "Digital Services BRD",
        )

    def test_default_title_is_supported(self):
        value = RequirementsRequest(
            source_text=(
                "The system must allow login."
            ),
            language=(
                SourceLanguage.ENGLISH
            ),
            project_id="PROJECT-001",
            trace_id="TRACE-001",
        )

        extraction = extract_requirements(
            value
        )

        brd = build_extracted_brd(
            value,
            extraction,
        )

        self.assertEqual(
            brd.title,
            "Business Requirements Document",
        )

    def test_custom_title_is_supported(self):
        value = request(
            "The system must allow login."
        )

        extraction = extract_requirements(
            value
        )

        brd = build_extracted_brd(
            value,
            extraction,
            title="Citizen Portal BRD",
        )

        self.assertEqual(
            brd.title,
            "Citizen Portal BRD",
        )

    def test_mismatched_extraction_is_rejected(self):
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
            build_extracted_brd(
                first,
                extraction,
            )


if __name__ == "__main__":
    unittest.main()
