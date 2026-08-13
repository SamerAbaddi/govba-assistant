"""Offline tests for GovBA-GAR change significance classification."""

from __future__ import annotations

import unittest
from datetime import date

from govba.change.clause_detection import (
    detect_clause_changes,
)
from govba.change.contract import (
    PolicyChangeImpact,
)
from govba.change.significance import (
    CHANGE_SIGNIFICANCE_VERSION,
    ChangeSignificancePolicy,
    ChangeSignificanceSignal,
    classify_change_significance,
    detect_significance_signals,
)
from govba.change.version_matching import (
    match_document_versions,
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
    content_hash,
):
    return AuthoritativeSource(
        document_id=document_id,
        title="Synthetic Policy",
        issuing_authority="Synthetic Ministry",
        document_type=DocumentType.POLICY,
        language=SourceLanguage.ENGLISH,
        jurisdiction="Jordan",
        effective_from=date(
            2026,
            1,
            1,
        ),
        status=SourceStatus.CURRENT,
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
        language=SourceLanguage.ENGLISH,
    )


def build_detection(
    old_text,
    new_text,
):
    baseline_source = source(
        "OLD",
        "a" * 64,
    )

    candidate_source = source(
        "NEW",
        "b" * 64,
    )

    old_chunks = (
        chunk(
            "OLD",
            0,
            old_text,
        ),
    )

    new_chunks = (
        chunk(
            "NEW",
            0,
            new_text,
        ),
    )

    match = match_document_versions(
        baseline_source,
        candidate_source,
    )

    detection = detect_clause_changes(
        match,
        baseline_chunks=old_chunks,
        candidate_chunks=new_chunks,
        trace_id="TRACE-001",
    )

    return (
        detection,
        old_chunks,
        new_chunks,
    )


class TestChangeSignificance(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            CHANGE_SIGNIFICANCE_VERSION,
            "govba-change-significance-v1",
        )

    def test_penalty_signal_is_detected(self):
        signals = detect_significance_signals(
            "A penalty shall apply."
        )

        self.assertIn(
            ChangeSignificanceSignal
            .PENALTY_SANCTION,
            signals,
        )

    def test_arabic_penalty_is_detected(self):
        signals = detect_significance_signals(
            "تطبق غرامة عند المخالفة"
        )

        self.assertIn(
            ChangeSignificanceSignal
            .PENALTY_SANCTION,
            signals,
        )

    def test_prohibition_signal_is_detected(self):
        signals = detect_significance_signals(
            "Employees must not disclose records."
        )

        self.assertIn(
            ChangeSignificanceSignal.PROHIBITION,
            signals,
        )

    def test_financial_signal_is_detected(self):
        signals = detect_significance_signals(
            "A fee of 20 JOD applies."
        )

        self.assertIn(
            ChangeSignificanceSignal.FINANCIAL,
            signals,
        )

    def test_deadline_signal_is_detected(self):
        signals = detect_significance_signals(
            "Submit within 10 days."
        )

        self.assertIn(
            ChangeSignificanceSignal.DEADLINE,
            signals,
        )

    def test_obligation_signal_is_detected(self):
        signals = detect_significance_signals(
            "The employee must submit the form."
        )

        self.assertIn(
            ChangeSignificanceSignal.OBLIGATION,
            signals,
        )

    def test_policy_id_is_deterministic(self):
        self.assertEqual(
            ChangeSignificancePolicy().policy_id,
            ChangeSignificancePolicy().policy_id,
        )

    def test_penalty_change_is_critical(self):
        (
            detection,
            old_chunks,
            new_chunks,
        ) = build_detection(
            "No penalty previously applied.",
            "A penalty of 100 JOD shall apply.",
        )

        result = classify_change_significance(
            detection,
            baseline_chunks=old_chunks,
            candidate_chunks=new_chunks,
        )

        self.assertEqual(
            result.maximum_impact,
            PolicyChangeImpact.CRITICAL,
        )

    def test_financial_change_is_high(self):
        (
            detection,
            old_chunks,
            new_chunks,
        ) = build_detection(
            "The service is available.",
            "A service fee is now payable.",
        )

        result = classify_change_significance(
            detection,
            baseline_chunks=old_chunks,
            candidate_chunks=new_chunks,
        )

        self.assertEqual(
            result.maximum_impact,
            PolicyChangeImpact.HIGH,
        )

    def test_obligation_change_is_medium(self):
        (
            detection,
            old_chunks,
            new_chunks,
        ) = build_detection(
            "Employees may submit a request.",
            "Employees must submit a request.",
        )

        result = classify_change_significance(
            detection,
            baseline_chunks=old_chunks,
            candidate_chunks=new_chunks,
        )

        self.assertEqual(
            result.maximum_impact,
            PolicyChangeImpact.MEDIUM,
        )

    def test_generic_change_defaults_low(self):
        (
            detection,
            old_chunks,
            new_chunks,
        ) = build_detection(
            "The office uses blue folders.",
            "The office uses green folders.",
        )

        result = classify_change_significance(
            detection,
            baseline_chunks=old_chunks,
            candidate_chunks=new_chunks,
        )

        self.assertEqual(
            result.maximum_impact,
            PolicyChangeImpact.LOW,
        )

    def test_highest_signal_controls_impact(self):
        (
            detection,
            old_chunks,
            new_chunks,
        ) = build_detection(
            "Employees submit requests.",
            (
                "Employees must submit requests "
                "or a penalty shall apply."
            ),
        )

        result = classify_change_significance(
            detection,
            baseline_chunks=old_chunks,
            candidate_chunks=new_chunks,
        )

        self.assertEqual(
            result.maximum_impact,
            PolicyChangeImpact.CRITICAL,
        )

    def test_custom_policy_changes_impact(self):
        (
            detection,
            old_chunks,
            new_chunks,
        ) = build_detection(
            "Employees may submit.",
            "Employees must submit.",
        )

        policy = ChangeSignificancePolicy(
            obligation_impact=(
                PolicyChangeImpact.HIGH
            )
        )

        result = classify_change_significance(
            detection,
            baseline_chunks=old_chunks,
            candidate_chunks=new_chunks,
            policy=policy,
        )

        self.assertEqual(
            result.maximum_impact,
            PolicyChangeImpact.HIGH,
        )

    def test_classification_is_deterministic(self):
        (
            detection,
            old_chunks,
            new_chunks,
        ) = build_detection(
            "Employees may submit.",
            "Employees must submit.",
        )

        first = classify_change_significance(
            detection,
            baseline_chunks=old_chunks,
            candidate_chunks=new_chunks,
        )

        second = classify_change_significance(
            detection,
            baseline_chunks=old_chunks,
            candidate_chunks=new_chunks,
        )

        self.assertEqual(
            first.classification_id,
            second.classification_id,
        )

    def test_wrong_document_chunk_is_rejected(self):
        (
            detection,
            _old_chunks,
            new_chunks,
        ) = build_detection(
            "Old clause.",
            "New clause.",
        )

        with self.assertRaises(
            ValueError
        ):
            classify_change_significance(
                detection,
                baseline_chunks=(
                    chunk(
                        "WRONG",
                        0,
                        "Old clause.",
                    ),
                ),
                candidate_chunks=new_chunks,
            )

    def test_missing_required_chunk_is_rejected(self):
        (
            detection,
            _old_chunks,
            new_chunks,
        ) = build_detection(
            "Employees may submit.",
            "Employees must submit.",
        )

        with self.assertRaises(
            ValueError
        ):
            classify_change_significance(
                detection,
                baseline_chunks=(),
                candidate_chunks=new_chunks,
            )

    def test_serialization_contains_no_clause_text(self):
        raw = (
            "A penalty of 100 JOD shall apply."
        )

        (
            detection,
            old_chunks,
            new_chunks,
        ) = build_detection(
            "No sanction applies.",
            raw,
        )

        result = classify_change_significance(
            detection,
            baseline_chunks=old_chunks,
            candidate_chunks=new_chunks,
        )

        data = result.to_dict()

        self.assertNotIn(
            raw,
            repr(data),
        )

        self.assertNotIn(
            "text",
            data,
        )


if __name__ == "__main__":
    unittest.main()
