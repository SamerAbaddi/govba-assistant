"""Offline tests for GovBA-GAR temporal evidence integration."""

from __future__ import annotations

import unittest
from datetime import (
    date,
    datetime,
    timezone,
)

from govba.rag.evidence import (
    EvidenceChunk,
)
from govba.rag.evidence_card import (
    EvidenceCard,
    EvidenceTemporalState,
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
from govba.rag.temporal_evidence import (
    TEMPORAL_EVIDENCE_VERSION,
    TemporalEvidenceIntegrationReport,
    TemporalEvidenceIntegrator,
    integrate_temporal_evidence,
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


def make_card(
    source,
    *,
    chunk_index=0,
):
    chunk = EvidenceChunk(
        document_id=(
            source.document_id
        ),
        chunk_index=chunk_index,
        text=(
            f"Evidence for "
            f"{source.document_id}."
        ),
        language=SourceLanguage.ENGLISH,
        page_start=1,
        page_end=1,
    )

    return EvidenceCard(
        source=source,
        chunk=chunk,
        retrieval_rank=(
            chunk_index + 1
        ),
        retrieval_score=0.9,
        retrieval_method=(
            "hybrid-rrf-v1"
        ),
    )


class TestTemporalEvidenceIntegration(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            TEMPORAL_EVIDENCE_VERSION,
            "govba-temporal-evidence-v1",
        )

    def test_insufficient_evidence_state_exists(self):
        self.assertEqual(
            EvidenceTemporalState
            .INSUFFICIENT
            .value,
            "insufficient",
        )

    def test_current_policy_marks_card_current(self):
        source = make_source(
            "DOC-A"
        )

        report = (
            integrate_temporal_evidence(
                (
                    make_card(
                        source
                    ),
                ),
                SupersessionGraph(
                    (
                        source,
                    )
                ),
                as_of_date=AS_OF,
            )
        )

        self.assertEqual(
            report.cards[
                0
            ].temporal_state,
            EvidenceTemporalState.CURRENT,
        )

        self.assertEqual(
            report.current_count,
            1,
        )

    def test_expired_policy_marks_card_superseded(self):
        source = make_source(
            "DOC-A",
            effective_from=date(
                2025,
                1,
                1,
            ),
            effective_until=date(
                2025,
                12,
                31,
            ),
        )

        report = (
            integrate_temporal_evidence(
                (
                    make_card(
                        source
                    ),
                ),
                SupersessionGraph(
                    (
                        source,
                    )
                ),
                as_of_date=AS_OF,
            )
        )

        self.assertEqual(
            report.cards[
                0
            ].temporal_state,
            EvidenceTemporalState.SUPERSEDED,
        )

        self.assertEqual(
            report.blocked_count,
            1,
        )

    def test_active_successor_marks_old_card_superseded(self):
        old = make_source(
            "DOC-OLD",
            effective_from=date(
                2025,
                1,
                1,
            ),
        )

        new = make_source(
            "DOC-NEW",
            effective_from=date(
                2026,
                6,
                1,
            ),
            supersedes=(
                "DOC-OLD",
            ),
        )

        graph = SupersessionGraph(
            (
                old,
                new,
            )
        )

        report = (
            integrate_temporal_evidence(
                (
                    make_card(
                        old
                    ),
                    make_card(
                        new,
                        chunk_index=1,
                    ),
                ),
                graph,
                as_of_date=AS_OF,
            )
        )

        states = {
            card.document_id: (
                card.temporal_state
            )
            for card
            in report.cards
        }

        self.assertEqual(
            states[
                "DOC-OLD"
            ],
            EvidenceTemporalState.SUPERSEDED,
        )

        self.assertEqual(
            states[
                "DOC-NEW"
            ],
            EvidenceTemporalState.CURRENT,
        )

    def test_missing_temporal_metadata_is_insufficient(self):
        source = make_source(
            "DOC-A",
            effective_from=None,
        )

        report = (
            integrate_temporal_evidence(
                (
                    make_card(
                        source
                    ),
                ),
                SupersessionGraph(
                    (
                        source,
                    )
                ),
                as_of_date=AS_OF,
            )
        )

        self.assertEqual(
            report.cards[
                0
            ].temporal_state,
            EvidenceTemporalState.INSUFFICIENT,
        )

        self.assertTrue(
            report.has_insufficient_evidence
        )

    def test_competing_successors_mark_component_conflicting(self):
        old = make_source(
            "DOC-A",
            effective_from=date(
                2025,
                1,
                1,
            ),
        )

        first = make_source(
            "DOC-B",
            effective_from=date(
                2026,
                5,
                1,
            ),
            supersedes=(
                "DOC-A",
            ),
        )

        second = make_source(
            "DOC-C",
            effective_from=date(
                2026,
                6,
                1,
            ),
            supersedes=(
                "DOC-A",
            ),
        )

        graph = SupersessionGraph(
            (
                old,
                first,
                second,
            )
        )

        report = (
            integrate_temporal_evidence(
                (
                    make_card(
                        first
                    ),
                    make_card(
                        second,
                        chunk_index=1,
                    ),
                ),
                graph,
                as_of_date=AS_OF,
            )
        )

        self.assertTrue(
            all(
                card.temporal_state
                is EvidenceTemporalState
                .CONFLICTING
                for card
                in report.cards
            )
        )

        self.assertTrue(
            report.has_conflicting_evidence
        )

    def test_cycle_marks_evidence_conflicting(self):
        first = make_source(
            "DOC-A",
            supersedes=(
                "DOC-B",
            ),
        )

        second = make_source(
            "DOC-B",
            supersedes=(
                "DOC-A",
            ),
        )

        graph = SupersessionGraph(
            (
                first,
                second,
            )
        )

        report = (
            integrate_temporal_evidence(
                (
                    make_card(
                        first
                    ),
                ),
                graph,
                as_of_date=AS_OF,
            )
        )

        self.assertEqual(
            report.cards[
                0
            ].temporal_state,
            EvidenceTemporalState.CONFLICTING,
        )

    def test_unrelated_conflict_does_not_change_current_card(self):
        current = make_source(
            "DOC-X"
        )

        old = make_source(
            "DOC-A",
            effective_from=date(
                2025,
                1,
                1,
            ),
        )

        first = make_source(
            "DOC-B",
            supersedes=(
                "DOC-A",
            ),
        )

        second = make_source(
            "DOC-C",
            supersedes=(
                "DOC-A",
            ),
        )

        graph = SupersessionGraph(
            (
                current,
                old,
                first,
                second,
            )
        )

        report = (
            integrate_temporal_evidence(
                (
                    make_card(
                        current
                    ),
                ),
                graph,
                as_of_date=AS_OF,
            )
        )

        self.assertEqual(
            report.cards[
                0
            ].temporal_state,
            EvidenceTemporalState.CURRENT,
        )

        self.assertFalse(
            report.has_temporal_risk
        )

    def test_card_identity_is_preserved(self):
        source = make_source(
            "DOC-A"
        )

        original = make_card(
            source
        )

        report = (
            integrate_temporal_evidence(
                (
                    original,
                ),
                SupersessionGraph(
                    (
                        source,
                    )
                ),
                as_of_date=AS_OF,
            )
        )

        updated = report.cards[
            0
        ]

        self.assertEqual(
            updated.card_id,
            original.card_id,
        )

        self.assertEqual(
            updated.chunk_id,
            original.chunk_id,
        )

    def test_input_order_is_preserved(self):
        source_a = make_source(
            "DOC-A"
        )

        source_b = make_source(
            "DOC-B"
        )

        report = (
            integrate_temporal_evidence(
                (
                    make_card(
                        source_b
                    ),
                    make_card(
                        source_a,
                        chunk_index=1,
                    ),
                ),
                SupersessionGraph(
                    (
                        source_a,
                        source_b,
                    )
                ),
                as_of_date=AS_OF,
            )
        )

        self.assertEqual(
            tuple(
                card.document_id
                for card
                in report.cards
            ),
            (
                "DOC-B",
                "DOC-A",
            ),
        )

    def test_unknown_card_source_is_rejected(self):
        registered = make_source(
            "DOC-A"
        )

        unknown = make_source(
            "DOC-X"
        )

        integrator = (
            TemporalEvidenceIntegrator(
                SupersessionGraph(
                    (
                        registered,
                    )
                )
            )
        )

        with self.assertRaises(
            KeyError
        ):
            integrator.integrate(
                (
                    make_card(
                        unknown
                    ),
                ),
                as_of_date=AS_OF,
            )

    def test_duplicate_cards_are_rejected(self):
        source = make_source(
            "DOC-A"
        )

        card = make_card(
            source
        )

        with self.assertRaises(
            ValueError
        ):
            integrate_temporal_evidence(
                (
                    card,
                    card,
                ),
                SupersessionGraph(
                    (
                        source,
                    )
                ),
                as_of_date=AS_OF,
            )

    def test_datetime_is_rejected(self):
        source = make_source(
            "DOC-A"
        )

        with self.assertRaises(
            TypeError
        ):
            integrate_temporal_evidence(
                (
                    make_card(
                        source
                    ),
                ),
                SupersessionGraph(
                    (
                        source,
                    )
                ),
                as_of_date=datetime.now(
                    timezone.utc
                ),
            )

    def test_invalid_graph_is_rejected(self):
        with self.assertRaises(
            TypeError
        ):
            TemporalEvidenceIntegrator(
                object()
            )

    def test_report_is_privacy_safe(self):
        source = make_source(
            "DOC-A"
        )

        card = make_card(
            source
        )

        report = (
            integrate_temporal_evidence(
                (
                    card,
                ),
                SupersessionGraph(
                    (
                        source,
                    )
                ),
                as_of_date=AS_OF,
            )
        )

        self.assertIsInstance(
            report,
            TemporalEvidenceIntegrationReport,
        )

        data = report.to_dict()

        self.assertEqual(
            data["integration_version"],
            TEMPORAL_EVIDENCE_VERSION,
        )

        self.assertNotIn(
            card.evidence_text,
            repr(
                data
            ),
        )


if __name__ == "__main__":
    unittest.main()
