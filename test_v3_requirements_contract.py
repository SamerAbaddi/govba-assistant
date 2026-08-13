"""Offline tests for GovBA-GAR requirements and BRD contract."""

from __future__ import annotations

import unittest

from govba.rag.models import (
    SourceLanguage,
)
from govba.requirements.contract import (
    REQUIREMENTS_CONTRACT_VERSION,
    BusinessRequirementsDocument,
    RequirementItem,
    RequirementOrigin,
    RequirementPriority,
    RequirementStatus,
    RequirementType,
    RequirementsRequest,
)


def request(
    *,
    text=(
        "The system must allow users "
        "to submit service requests."
    ),
):
    return RequirementsRequest(
        source_text=text,
        language=(
            SourceLanguage.ENGLISH
        ),
        project_id="PROJECT-001",
        trace_id="TRACE-001",
        origin=(
            RequirementOrigin.USER_INPUT
        ),
        document_title=(
            "Digital Services BRD"
        ),
        source_reference="SRC-001",
    )


def requirement(
    request_id,
    *,
    sequence=0,
    statement=(
        "The system shall allow users "
        "to submit service requests."
    ),
    requirement_type=(
        RequirementType.FUNCTIONAL
    ),
    priority=(
        RequirementPriority.MUST
    ),
    status=(
        RequirementStatus.DRAFT
    ),
):
    return RequirementItem(
        request_id=request_id,
        sequence=sequence,
        requirement_type=(
            requirement_type
        ),
        priority=priority,
        status=status,
        statement=statement,
        rationale=(
            "Supports digital service delivery."
        ),
        acceptance_criteria=(
            "User can create a request.",
            "System assigns a request ID.",
        ),
        source_references=(
            "SRC-001",
        ),
    )


class TestRequirementsRequest(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            REQUIREMENTS_CONTRACT_VERSION,
            "govba-requirements-contract-v1",
        )

    def test_request_is_created(self):
        value = request()

        self.assertEqual(
            value.project_id,
            "PROJECT-001",
        )

    def test_source_hash_is_sha256(self):
        value = request()

        self.assertEqual(
            len(
                value.source_sha256
            ),
            64,
        )

        int(
            value.source_sha256,
            16,
        )

    def test_request_id_is_deterministic(self):
        self.assertEqual(
            request().request_id,
            request().request_id,
        )

    def test_source_change_changes_identity(self):
        first = request(
            text="Requirement one."
        )

        second = request(
            text="Requirement two."
        )

        self.assertNotEqual(
            first.request_id,
            second.request_id,
        )

    def test_blank_source_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            request(
                text=" "
            )

    def test_invalid_language_is_rejected(self):
        with self.assertRaises(
            TypeError
        ):
            RequirementsRequest(
                source_text="Requirement.",
                language="en",
                project_id="P1",
                trace_id="T1",
            )

    def test_default_serialization_excludes_source_text(self):
        raw = (
            "Unique confidential "
            "requirements source text."
        )

        value = request(
            text=raw
        )

        data = value.to_dict()

        self.assertNotIn(
            raw,
            repr(data),
        )

        self.assertNotIn(
            "source_text",
            data,
        )

    def test_source_can_be_explicitly_serialized(self):
        value = request()

        data = value.to_dict(
            include_content=True
        )

        self.assertEqual(
            data[
                "source_text"
            ],
            value.source_text,
        )


class TestRequirementItem(
    unittest.TestCase
):
    def test_requirement_is_created(self):
        value = request()

        item = requirement(
            value.request_id
        )

        self.assertEqual(
            item.requirement_type,
            RequirementType.FUNCTIONAL,
        )

    def test_statement_hash_is_sha256(self):
        value = request()

        item = requirement(
            value.request_id
        )

        self.assertEqual(
            len(
                item.statement_sha256
            ),
            64,
        )

    def test_requirement_id_is_deterministic(self):
        value = request()

        first = requirement(
            value.request_id
        )

        second = requirement(
            value.request_id
        )

        self.assertEqual(
            first.requirement_id,
            second.requirement_id,
        )

    def test_statement_change_changes_identity(self):
        value = request()

        first = requirement(
            value.request_id,
            statement="System shall allow login.",
        )

        second = requirement(
            value.request_id,
            statement="System shall allow logout.",
        )

        self.assertNotEqual(
            first.requirement_id,
            second.requirement_id,
        )

    def test_negative_sequence_is_rejected(self):
        value = request()

        with self.assertRaises(
            ValueError
        ):
            requirement(
                value.request_id,
                sequence=-1,
            )

    def test_duplicate_acceptance_criteria_rejected(self):
        value = request()

        with self.assertRaises(
            ValueError
        ):
            RequirementItem(
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
                statement="System shall submit requests.",
                acceptance_criteria=(
                    "Request is saved.",
                    "Request is saved.",
                ),
            )

    def test_default_serialization_excludes_requirement_text(self):
        value = request()

        raw = (
            "Unique private requirement statement."
        )

        item = requirement(
            value.request_id,
            statement=raw,
        )

        data = item.to_dict()

        self.assertNotIn(
            raw,
            repr(data),
        )

        self.assertNotIn(
            "statement",
            data,
        )

    def test_requirement_content_can_be_explicitly_serialized(self):
        value = request()

        item = requirement(
            value.request_id
        )

        data = item.to_dict(
            include_content=True
        )

        self.assertEqual(
            data[
                "statement"
            ],
            item.statement,
        )


class TestBusinessRequirementsDocument(
    unittest.TestCase
):
    def test_brd_is_created(self):
        value = request()

        item = requirement(
            value.request_id
        )

        brd = BusinessRequirementsDocument(
            request_id=value.request_id,
            project_id=value.project_id,
            trace_id=value.trace_id,
            title="Digital Services BRD",
            requirements=(
                item,
            ),
        )

        self.assertEqual(
            brd.requirement_count,
            1,
        )

    def test_brd_id_is_deterministic(self):
        value = request()

        item = requirement(
            value.request_id
        )

        first = BusinessRequirementsDocument(
            request_id=value.request_id,
            project_id=value.project_id,
            trace_id=value.trace_id,
            title="Digital Services BRD",
            requirements=(
                item,
            ),
        )

        second = BusinessRequirementsDocument(
            request_id=value.request_id,
            project_id=value.project_id,
            trace_id=value.trace_id,
            title="Digital Services BRD",
            requirements=(
                item,
            ),
        )

        self.assertEqual(
            first.brd_id,
            second.brd_id,
        )

    def test_requirement_sequences_must_be_contiguous(self):
        value = request()

        item = requirement(
            value.request_id,
            sequence=1,
        )

        with self.assertRaises(
            ValueError
        ):
            BusinessRequirementsDocument(
                request_id=value.request_id,
                project_id=value.project_id,
                trace_id=value.trace_id,
                title="BRD",
                requirements=(
                    item,
                ),
            )

    def test_wrong_request_requirement_is_rejected(self):
        first = request(
            text="First source."
        )

        second = request(
            text="Second source."
        )

        item = requirement(
            second.request_id
        )

        with self.assertRaises(
            ValueError
        ):
            BusinessRequirementsDocument(
                request_id=first.request_id,
                project_id=first.project_id,
                trace_id=first.trace_id,
                title="BRD",
                requirements=(
                    item,
                ),
            )

    def test_must_count(self):
        value = request()

        first = requirement(
            value.request_id,
            sequence=0,
            priority=(
                RequirementPriority.MUST
            ),
        )

        second = requirement(
            value.request_id,
            sequence=1,
            statement=(
                "System should provide exports."
            ),
            priority=(
                RequirementPriority.SHOULD
            ),
        )

        brd = BusinessRequirementsDocument(
            request_id=value.request_id,
            project_id=value.project_id,
            trace_id=value.trace_id,
            title="BRD",
            requirements=(
                first,
                second,
            ),
        )

        self.assertEqual(
            brd.must_count,
            1,
        )

    def test_approved_count(self):
        value = request()

        item = requirement(
            value.request_id,
            status=(
                RequirementStatus.APPROVED
            ),
        )

        brd = BusinessRequirementsDocument(
            request_id=value.request_id,
            project_id=value.project_id,
            trace_id=value.trace_id,
            title="BRD",
            requirements=(
                item,
            ),
        )

        self.assertEqual(
            brd.approved_count,
            1,
        )

    def test_empty_brd_is_supported(self):
        value = request()

        brd = BusinessRequirementsDocument(
            request_id=value.request_id,
            project_id=value.project_id,
            trace_id=value.trace_id,
            title="Empty BRD",
            requirements=(),
        )

        self.assertEqual(
            brd.requirement_count,
            0,
        )

    def test_default_brd_serialization_excludes_raw_text(self):
        value = request()

        raw = (
            "Unique sensitive functional requirement."
        )

        item = requirement(
            value.request_id,
            statement=raw,
        )

        brd = BusinessRequirementsDocument(
            request_id=value.request_id,
            project_id=value.project_id,
            trace_id=value.trace_id,
            title="Confidential BRD",
            requirements=(
                item,
            ),
        )

        data = brd.to_dict()

        self.assertNotIn(
            raw,
            repr(data),
        )

        self.assertNotIn(
            "Confidential BRD",
            repr(data),
        )

    def test_brd_content_can_be_explicitly_serialized(self):
        value = request()

        item = requirement(
            value.request_id
        )

        brd = BusinessRequirementsDocument(
            request_id=value.request_id,
            project_id=value.project_id,
            trace_id=value.trace_id,
            title="Digital Services BRD",
            requirements=(
                item,
            ),
        )

        data = brd.to_dict(
            include_content=True
        )

        self.assertEqual(
            data[
                "title"
            ],
            "Digital Services BRD",
        )

        self.assertIn(
            "statement",
            data[
                "requirements"
            ][0],
        )


if __name__ == "__main__":
    unittest.main()
