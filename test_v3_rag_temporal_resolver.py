"""Offline tests for GovBA-GAR as-of-date temporal resolution."""

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
from govba.rag.temporal import (
    TemporalPolicyState,
    TemporalReasonCode,
)
from govba.rag.temporal_resolver import (
    TEMPORAL_RESOLVER_VERSION,
    TemporalResolution,
    TemporalResolver,
)


def make_source(
    document_id,
    *,
    effective_from=None,
    effective_until=None,
    supersedes=(),
    superseded_by=(),
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
        superseded_by=tuple(
            superseded_by
        ),
        official_source_url=(
            "https://example.gov.jo/"
            f"{document_id}.pdf"
        ),
    )


class TestTemporalResolver(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            TEMPORAL_RESOLVER_VERSION,
            "govba-temporal-resolver-v1",
        )

    def test_current_within_effective_window(self):
        graph = SupersessionGraph(
            (
                make_source(
                    "DOC-A",
                    effective_from=date(
                        2026,
                        1,
                        1,
                    ),
                    effective_until=date(
                        2026,
                        12,
                        31,
                    ),
                ),
            )
        )

        result = TemporalResolver(
            graph
        ).resolve(
            "DOC-A",
            as_of_date=date(
                2026,
                8,
                13,
            ),
        )

        self.assertEqual(
            result.state,
            TemporalPolicyState.CURRENT,
        )

        self.assertIn(
            TemporalReasonCode
            .WITHIN_EFFECTIVE_WINDOW,
            result.assessment.reason_codes,
        )

    def test_before_effective_date_is_insufficient(self):
        graph = SupersessionGraph(
            (
                make_source(
                    "DOC-A",
                    effective_from=date(
                        2026,
                        9,
                        1,
                    ),
                ),
            )
        )

        result = TemporalResolver(
            graph
        ).resolve(
            "DOC-A",
            as_of_date=date(
                2026,
                8,
                13,
            ),
        )

        self.assertEqual(
            result.state,
            TemporalPolicyState.INSUFFICIENT,
        )

        self.assertTrue(
            result.should_abstain
        )

    def test_after_effective_until_is_superseded(self):
        graph = SupersessionGraph(
            (
                make_source(
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
                ),
            )
        )

        result = TemporalResolver(
            graph
        ).resolve(
            "DOC-A",
            as_of_date=date(
                2026,
                1,
                1,
            ),
        )

        self.assertEqual(
            result.state,
            TemporalPolicyState.SUPERSEDED,
        )

        self.assertIn(
            TemporalReasonCode
            .AFTER_EFFECTIVE_DATE,
            result.assessment.reason_codes,
        )

    def test_active_superseder_marks_old_document_superseded(self):
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

        result = TemporalResolver(
            graph
        ).resolve(
            "DOC-OLD",
            as_of_date=date(
                2026,
                8,
                13,
            ),
        )

        self.assertEqual(
            result.state,
            TemporalPolicyState.SUPERSEDED,
        )

        self.assertEqual(
            result.active_superseding_document_ids,
            (
                "DOC-NEW",
            ),
        )

    def test_future_superseder_does_not_replace_policy_yet(self):
        graph = SupersessionGraph(
            (
                make_source(
                    "DOC-NEW",
                    effective_from=date(
                        2026,
                        9,
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

        result = TemporalResolver(
            graph
        ).resolve(
            "DOC-OLD",
            as_of_date=date(
                2026,
                8,
                13,
            ),
        )

        self.assertEqual(
            result.state,
            TemporalPolicyState.CURRENT,
        )

        self.assertEqual(
            result.future_superseding_document_ids,
            (
                "DOC-NEW",
            ),
        )

    def test_unknown_superseder_date_is_insufficient(self):
        graph = SupersessionGraph(
            (
                make_source(
                    "DOC-NEW",
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

        result = TemporalResolver(
            graph
        ).resolve(
            "DOC-OLD",
            as_of_date=date(
                2026,
                8,
                13,
            ),
        )

        self.assertEqual(
            result.state,
            TemporalPolicyState.INSUFFICIENT,
        )

        self.assertEqual(
            result.unresolved_superseding_document_ids,
            (
                "DOC-NEW",
            ),
        )

    def test_unresolved_superseded_by_reference_is_insufficient(self):
        graph = SupersessionGraph(
            (
                make_source(
                    "DOC-OLD",
                    effective_from=date(
                        2025,
                        1,
                        1,
                    ),
                    superseded_by=(
                        "DOC-MISSING",
                    ),
                ),
            )
        )

        result = TemporalResolver(
            graph
        ).resolve(
            "DOC-OLD",
            as_of_date=date(
                2026,
                8,
                13,
            ),
        )

        self.assertEqual(
            result.state,
            TemporalPolicyState.INSUFFICIENT,
        )

        self.assertEqual(
            result.unresolved_superseding_document_ids,
            (
                "DOC-MISSING",
            ),
        )

    def test_transitive_superseder_is_detected(self):
        graph = SupersessionGraph(
            (
                make_source(
                    "DOC-C",
                    effective_from=date(
                        2026,
                        7,
                        1,
                    ),
                    supersedes=(
                        "DOC-B",
                    ),
                ),
                make_source(
                    "DOC-B",
                    effective_from=date(
                        2026,
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

        result = TemporalResolver(
            graph
        ).resolve(
            "DOC-A",
            as_of_date=date(
                2026,
                8,
                13,
            ),
        )

        self.assertEqual(
            result.state,
            TemporalPolicyState.SUPERSEDED,
        )

        self.assertEqual(
            result.active_superseding_document_ids,
            (
                "DOC-B",
                "DOC-C",
            ),
        )

    def test_cycle_is_conflicting(self):
        graph = SupersessionGraph(
            (
                make_source(
                    "DOC-A",
                    effective_from=date(
                        2026,
                        1,
                        1,
                    ),
                    supersedes=(
                        "DOC-B",
                    ),
                ),
                make_source(
                    "DOC-B",
                    effective_from=date(
                        2026,
                        1,
                        1,
                    ),
                    supersedes=(
                        "DOC-A",
                    ),
                ),
            )
        )

        result = TemporalResolver(
            graph
        ).resolve(
            "DOC-A",
            as_of_date=date(
                2026,
                8,
                13,
            ),
        )

        self.assertEqual(
            result.state,
            TemporalPolicyState.CONFLICTING,
        )

        self.assertTrue(
            result.should_abstain
        )

    def test_missing_effective_start_is_insufficient(self):
        graph = SupersessionGraph(
            (
                make_source(
                    "DOC-A"
                ),
            )
        )

        result = TemporalResolver(
            graph
        ).resolve(
            "DOC-A",
            as_of_date=date(
                2026,
                8,
                13,
            ),
        )

        self.assertEqual(
            result.state,
            TemporalPolicyState.INSUFFICIENT,
        )

        self.assertIn(
            TemporalReasonCode
            .MISSING_TEMPORAL_METADATA,
            result.assessment.reason_codes,
        )

    def test_effective_start_boundary_is_current(self):
        start = date(
            2026,
            8,
            13,
        )

        graph = SupersessionGraph(
            (
                make_source(
                    "DOC-A",
                    effective_from=start,
                ),
            )
        )

        result = TemporalResolver(
            graph
        ).resolve(
            "DOC-A",
            as_of_date=start,
        )

        self.assertEqual(
            result.state,
            TemporalPolicyState.CURRENT,
        )

    def test_effective_end_boundary_is_current(self):
        end = date(
            2026,
            8,
            13,
        )

        graph = SupersessionGraph(
            (
                make_source(
                    "DOC-A",
                    effective_from=date(
                        2026,
                        1,
                        1,
                    ),
                    effective_until=end,
                ),
            )
        )

        result = TemporalResolver(
            graph
        ).resolve(
            "DOC-A",
            as_of_date=end,
        )

        self.assertEqual(
            result.state,
            TemporalPolicyState.CURRENT,
        )

    def test_datetime_is_rejected(self):
        graph = SupersessionGraph(
            (
                make_source(
                    "DOC-A",
                    effective_from=date(
                        2026,
                        1,
                        1,
                    ),
                ),
            )
        )

        with self.assertRaises(
            TypeError
        ):
            TemporalResolver(
                graph
            ).resolve(
                "DOC-A",
                as_of_date=datetime.now(
                    timezone.utc
                ),
            )

    def test_unknown_document_raises_key_error(self):
        resolver = TemporalResolver(
            SupersessionGraph(())
        )

        with self.assertRaises(
            KeyError
        ):
            resolver.resolve(
                "DOC-MISSING",
                as_of_date=date(
                    2026,
                    8,
                    13,
                ),
            )

    def test_resolve_all_is_deterministically_ordered(self):
        graph = SupersessionGraph(
            (
                make_source(
                    "DOC-B",
                    effective_from=date(
                        2026,
                        1,
                        1,
                    ),
                ),
                make_source(
                    "DOC-A",
                    effective_from=date(
                        2026,
                        1,
                        1,
                    ),
                ),
            )
        )

        results = TemporalResolver(
            graph
        ).resolve_all(
            as_of_date=date(
                2026,
                8,
                13,
            )
        )

        self.assertEqual(
            tuple(
                result.document_id
                for result in results
            ),
            (
                "DOC-A",
                "DOC-B",
            ),
        )

    def test_resolution_serializes_without_source_text(self):
        graph = SupersessionGraph(
            (
                make_source(
                    "DOC-A",
                    effective_from=date(
                        2026,
                        1,
                        1,
                    ),
                ),
            )
        )

        result = TemporalResolver(
            graph
        ).resolve(
            "DOC-A",
            as_of_date=date(
                2026,
                8,
                13,
            ),
        )

        data = result.to_dict()

        self.assertEqual(
            data["resolver_version"],
            TEMPORAL_RESOLVER_VERSION,
        )

        self.assertEqual(
            data["assessment"]["state"],
            "current",
        )


if __name__ == "__main__":
    unittest.main()
