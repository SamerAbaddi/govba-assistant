"""Offline tests for GovBA-GAR temporal policy conflicts."""

from __future__ import annotations

import unittest
from datetime import (
    date,
    datetime,
    timezone,
)

from govba.rag.models import (
    AuthoritativeSource,
    DocumentType,
    SourceLanguage,
    SourceStatus,
)
from govba.rag.supersession import (
    SupersessionGraph,
)
from govba.rag.temporal_conflicts import (
    TEMPORAL_CONFLICT_VERSION,
    TemporalConflictCode,
    TemporalConflictDecision,
    TemporalConflictDetector,
    detect_temporal_conflicts,
)


AS_OF = date(
    2026,
    8,
    13,
)


def make_source(
    document_id,
    *,
    effective_from=date(
        2026,
        1,
        1,
    ),
    effective_until=None,
    supersedes=(),
):
    return AuthoritativeSource(
        document_id=document_id,
        title=f"Policy {document_id}",
        issuing_authority=(
            "Synthetic Government Authority"
        ),
        document_type=DocumentType.POLICY,
        language=SourceLanguage.ENGLISH,
        status=SourceStatus.CURRENT,
        effective_from=effective_from,
        effective_until=effective_until,
        supersedes=tuple(
            supersedes
        ),
        official_source_url=(
            "https://example.gov.jo/"
            f"{document_id}.pdf"
        ),
    )


class TestTemporalConflicts(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            TEMPORAL_CONFLICT_VERSION,
            "govba-temporal-conflict-v1",
        )

    def test_single_current_policy_is_clear(self):
        graph = SupersessionGraph(
            (
                make_source(
                    "DOC-A"
                ),
            )
        )

        report = detect_temporal_conflicts(
            graph,
            as_of_date=AS_OF,
        )

        self.assertEqual(
            report.decision,
            TemporalConflictDecision.CLEAR,
        )

        self.assertFalse(
            report.should_abstain
        )

    def test_normal_supersession_chain_is_clear(self):
        graph = SupersessionGraph(
            (
                make_source(
                    "DOC-NEW",
                    effective_from=date(
                        2026,
                        6,
                        1,
                    ),
                    supersedes=(
                        "DOC-OLD",
                    ),
                ),
                make_source(
                    "DOC-OLD",
                    effective_from=date(
                        2025,
                        1,
                        1,
                    ),
                ),
            )
        )

        report = (
            TemporalConflictDetector(
                graph
            ).detect(
                as_of_date=AS_OF
            )
        )

        self.assertEqual(
            report.decision,
            TemporalConflictDecision.CLEAR,
        )

        self.assertEqual(
            report.conflict_count,
            0,
        )

    def test_unrelated_current_documents_do_not_conflict(self):
        graph = SupersessionGraph(
            (
                make_source(
                    "DOC-A"
                ),
                make_source(
                    "DOC-B"
                ),
            )
        )

        report = detect_temporal_conflicts(
            graph,
            as_of_date=AS_OF,
        )

        self.assertFalse(
            report.has_conflict
        )

    def test_two_active_successor_branches_conflict(self):
        graph = SupersessionGraph(
            (
                make_source(
                    "DOC-B",
                    effective_from=date(
                        2026,
                        5,
                        1,
                    ),
                    supersedes=(
                        "DOC-A",
                    ),
                ),
                make_source(
                    "DOC-C",
                    effective_from=date(
                        2026,
                        6,
                        1,
                    ),
                    supersedes=(
                        "DOC-A",
                    ),
                ),
                make_source(
                    "DOC-A",
                    effective_from=date(
                        2025,
                        1,
                        1,
                    ),
                ),
            )
        )

        report = detect_temporal_conflicts(
            graph,
            as_of_date=AS_OF,
        )

        self.assertEqual(
            report.decision,
            TemporalConflictDecision.CONFLICTING,
        )

        self.assertTrue(
            report.should_abstain
        )

        self.assertEqual(
            report.conflict_count,
            1,
        )

        conflict = report.conflicts[
            0
        ]

        self.assertEqual(
            conflict.code,
            (
                TemporalConflictCode
                .MULTIPLE_CURRENT_SUCCESSORS
            ),
        )

        self.assertEqual(
            conflict.current_document_ids,
            (
                "DOC-B",
                "DOC-C",
            ),
        )

    def test_future_successor_branch_is_not_conflict_yet(self):
        graph = SupersessionGraph(
            (
                make_source(
                    "DOC-B",
                    effective_from=date(
                        2026,
                        5,
                        1,
                    ),
                    supersedes=(
                        "DOC-A",
                    ),
                ),
                make_source(
                    "DOC-C",
                    effective_from=date(
                        2027,
                        1,
                        1,
                    ),
                    supersedes=(
                        "DOC-A",
                    ),
                ),
                make_source(
                    "DOC-A",
                    effective_from=date(
                        2025,
                        1,
                        1,
                    ),
                ),
            )
        )

        report = detect_temporal_conflicts(
            graph,
            as_of_date=AS_OF,
        )

        self.assertEqual(
            report.decision,
            TemporalConflictDecision.CLEAR,
        )

    def test_cycle_is_detected_as_conflict(self):
        graph = SupersessionGraph(
            (
                make_source(
                    "DOC-A",
                    supersedes=(
                        "DOC-B",
                    ),
                ),
                make_source(
                    "DOC-B",
                    supersedes=(
                        "DOC-A",
                    ),
                ),
            )
        )

        report = detect_temporal_conflicts(
            graph,
            as_of_date=AS_OF,
        )

        self.assertTrue(
            report.has_conflict
        )

        self.assertEqual(
            report.conflicts[
                0
            ].code,
            (
                TemporalConflictCode
                .SUPERSESSION_CYCLE
            ),
        )

    def test_missing_temporal_metadata_is_insufficient(self):
        graph = SupersessionGraph(
            (
                make_source(
                    "DOC-A",
                    effective_from=None,
                ),
            )
        )

        report = detect_temporal_conflicts(
            graph,
            as_of_date=AS_OF,
        )

        self.assertEqual(
            report.decision,
            TemporalConflictDecision.INSUFFICIENT,
        )

        self.assertEqual(
            report.insufficient_document_ids,
            (
                "DOC-A",
            ),
        )

        self.assertTrue(
            report.should_abstain
        )

    def test_conflict_takes_priority_over_insufficient(self):
        graph = SupersessionGraph(
            (
                make_source(
                    "DOC-B",
                    supersedes=(
                        "DOC-A",
                    ),
                ),
                make_source(
                    "DOC-C",
                    supersedes=(
                        "DOC-A",
                    ),
                ),
                make_source(
                    "DOC-A",
                    effective_from=date(
                        2025,
                        1,
                        1,
                    ),
                ),
                make_source(
                    "DOC-X",
                    effective_from=None,
                ),
            )
        )

        report = detect_temporal_conflicts(
            graph,
            as_of_date=AS_OF,
        )

        self.assertEqual(
            report.decision,
            TemporalConflictDecision.CONFLICTING,
        )

        self.assertIn(
            "DOC-X",
            report.insufficient_document_ids,
        )

    def test_conflict_component_is_deterministically_sorted(self):
        graph = SupersessionGraph(
            (
                make_source(
                    "DOC-C",
                    supersedes=(
                        "DOC-A",
                    ),
                ),
                make_source(
                    "DOC-B",
                    supersedes=(
                        "DOC-A",
                    ),
                ),
                make_source(
                    "DOC-A",
                    effective_from=date(
                        2025,
                        1,
                        1,
                    ),
                ),
            )
        )

        conflict = (
            detect_temporal_conflicts(
                graph,
                as_of_date=AS_OF,
            )
            .conflicts[0]
        )

        self.assertEqual(
            conflict.component_document_ids,
            (
                "DOC-A",
                "DOC-B",
                "DOC-C",
            ),
        )

    def test_conflict_id_is_sha256(self):
        graph = SupersessionGraph(
            (
                make_source(
                    "DOC-B",
                    supersedes=(
                        "DOC-A",
                    ),
                ),
                make_source(
                    "DOC-C",
                    supersedes=(
                        "DOC-A",
                    ),
                ),
                make_source(
                    "DOC-A",
                    effective_from=date(
                        2025,
                        1,
                        1,
                    ),
                ),
            )
        )

        conflict = (
            detect_temporal_conflicts(
                graph,
                as_of_date=AS_OF,
            )
            .conflicts[0]
        )

        self.assertEqual(
            len(
                conflict.conflict_id
            ),
            64,
        )

        int(
            conflict.conflict_id,
            16,
        )

    def test_conflict_id_is_deterministic(self):
        sources = (
            make_source(
                "DOC-B",
                supersedes=(
                    "DOC-A",
                ),
            ),
            make_source(
                "DOC-C",
                supersedes=(
                    "DOC-A",
                ),
            ),
            make_source(
                "DOC-A",
                effective_from=date(
                    2025,
                    1,
                    1,
                ),
            ),
        )

        first = detect_temporal_conflicts(
            SupersessionGraph(
                sources
            ),
            as_of_date=AS_OF,
        )

        second = detect_temporal_conflicts(
            SupersessionGraph(
                reversed(
                    sources
                )
            ),
            as_of_date=AS_OF,
        )

        self.assertEqual(
            first.conflicts[
                0
            ].conflict_id,
            second.conflicts[
                0
            ].conflict_id,
        )

    def test_datetime_is_rejected(self):
        detector = (
            TemporalConflictDetector(
                SupersessionGraph(
                    (
                        make_source(
                            "DOC-A"
                        ),
                    )
                )
            )
        )

        with self.assertRaises(
            TypeError
        ):
            detector.detect(
                as_of_date=datetime.now(
                    timezone.utc
                )
            )

    def test_invalid_graph_is_rejected(self):
        with self.assertRaises(
            TypeError
        ):
            TemporalConflictDetector(
                object()
            )

    def test_report_serialization_is_privacy_safe(self):
        graph = SupersessionGraph(
            (
                make_source(
                    "DOC-A"
                ),
            )
        )

        report = detect_temporal_conflicts(
            graph,
            as_of_date=AS_OF,
        )

        data = report.to_dict()

        self.assertEqual(
            data["detector_version"],
            TEMPORAL_CONFLICT_VERSION,
        )

        self.assertEqual(
            data["decision"],
            "clear",
        )

        self.assertNotIn(
            "Policy DOC-A",
            repr(
                data
            ),
        )


if __name__ == "__main__":
    unittest.main()
