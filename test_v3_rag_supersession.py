"""Offline tests for GovBA-GAR policy supersession graph."""

from __future__ import annotations

import unittest

from govba.rag.models import (
    AuthoritativeSource,
    DocumentType,
    SourceLanguage,
    SourceStatus,
)
from govba.rag.supersession import (
    SUPERSESSION_GRAPH_VERSION,
    SupersessionDeclaration,
    SupersessionGraph,
)


def make_source(
    document_id,
    *,
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


class TestSupersessionGraph(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            SUPERSESSION_GRAPH_VERSION,
            "govba-supersession-graph-v1",
        )

    def test_empty_relationship_graph(self):
        graph = SupersessionGraph(
            (
                make_source(
                    "DOC-A"
                ),
            )
        )

        self.assertEqual(
            graph.source_count,
            1,
        )
        self.assertEqual(
            graph.edge_count,
            0,
        )
        self.assertFalse(
            graph.has_cycle
        )

    def test_supersedes_field_creates_edge(self):
        graph = SupersessionGraph(
            (
                make_source(
                    "DOC-NEW",
                    supersedes=(
                        "DOC-OLD",
                    ),
                ),
                make_source(
                    "DOC-OLD"
                ),
            )
        )

        self.assertTrue(
            graph.has_direct_edge(
                "DOC-NEW",
                "DOC-OLD",
            )
        )

    def test_superseded_by_field_creates_edge(self):
        graph = SupersessionGraph(
            (
                make_source(
                    "DOC-OLD",
                    superseded_by=(
                        "DOC-NEW",
                    ),
                ),
                make_source(
                    "DOC-NEW"
                ),
            )
        )

        self.assertTrue(
            graph.has_direct_edge(
                "DOC-NEW",
                "DOC-OLD",
            )
        )

    def test_reciprocal_declarations_create_one_edge(self):
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
                    superseded_by=(
                        "DOC-NEW",
                    ),
                ),
            )
        )

        self.assertEqual(
            graph.edge_count,
            1,
        )

        self.assertEqual(
            set(
                graph.edges[
                    0
                ].declarations
            ),
            {
                SupersessionDeclaration
                .SUPERSEDES,
                SupersessionDeclaration
                .SUPERSEDED_BY,
            },
        )

    def test_direct_supersedes_query(self):
        graph = SupersessionGraph(
            (
                make_source(
                    "DOC-C",
                    supersedes=(
                        "DOC-B",
                    ),
                ),
                make_source(
                    "DOC-B"
                ),
            )
        )

        self.assertEqual(
            graph.supersedes(
                "DOC-C"
            ),
            (
                "DOC-B",
            ),
        )

    def test_direct_superseded_by_query(self):
        graph = SupersessionGraph(
            (
                make_source(
                    "DOC-C",
                    supersedes=(
                        "DOC-B",
                    ),
                ),
                make_source(
                    "DOC-B"
                ),
            )
        )

        self.assertEqual(
            graph.superseded_by(
                "DOC-B"
            ),
            (
                "DOC-C",
            ),
        )

    def test_transitive_superseded_documents(self):
        graph = SupersessionGraph(
            (
                make_source(
                    "DOC-C",
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
                make_source(
                    "DOC-A"
                ),
            )
        )

        self.assertEqual(
            graph.all_superseded_documents(
                "DOC-C"
            ),
            (
                "DOC-A",
                "DOC-B",
            ),
        )

    def test_transitive_superseding_documents(self):
        graph = SupersessionGraph(
            (
                make_source(
                    "DOC-C",
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
                make_source(
                    "DOC-A"
                ),
            )
        )

        self.assertEqual(
            graph.all_superseding_documents(
                "DOC-A"
            ),
            (
                "DOC-B",
                "DOC-C",
            ),
        )

    def test_unknown_reference_is_tracked(self):
        graph = SupersessionGraph(
            (
                make_source(
                    "DOC-A",
                    supersedes=(
                        "DOC-MISSING",
                    ),
                ),
            )
        )

        self.assertEqual(
            graph.unresolved_count,
            1,
        )

        reference = (
            graph.unresolved_references[
                0
            ]
        )

        self.assertEqual(
            reference.related_document_id,
            "DOC-MISSING",
        )

    def test_cycle_is_detected(self):
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

        self.assertTrue(
            graph.has_cycle
        )

        self.assertEqual(
            graph.cycle_document_ids,
            (
                "DOC-A",
                "DOC-B",
            ),
        )

    def test_acyclic_chain_has_no_cycle(self):
        graph = SupersessionGraph(
            (
                make_source(
                    "DOC-C",
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
                make_source(
                    "DOC-A"
                ),
            )
        )

        self.assertFalse(
            graph.has_cycle
        )

    def test_duplicate_source_ids_are_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            SupersessionGraph(
                (
                    make_source(
                        "DOC-A"
                    ),
                    make_source(
                        "DOC-A"
                    ),
                )
            )

    def test_invalid_source_is_rejected(self):
        with self.assertRaises(
            TypeError
        ):
            SupersessionGraph(
                (
                    object(),
                )
            )

    def test_unknown_document_query_raises(self):
        graph = SupersessionGraph(
            (
                make_source(
                    "DOC-A"
                ),
            )
        )

        with self.assertRaises(
            KeyError
        ):
            graph.supersedes(
                "DOC-MISSING"
            )

    def test_edge_id_is_deterministic(self):
        sources = (
            make_source(
                "DOC-B",
                supersedes=(
                    "DOC-A",
                ),
            ),
            make_source(
                "DOC-A"
            ),
        )

        first = SupersessionGraph(
            sources
        )

        second = SupersessionGraph(
            reversed(
                sources
            )
        )

        self.assertEqual(
            first.edges[
                0
            ].edge_id,
            second.edges[
                0
            ].edge_id,
        )

    def test_graph_serialization_contains_integrity_metadata(self):
        graph = SupersessionGraph(
            (
                make_source(
                    "DOC-B",
                    supersedes=(
                        "DOC-A",
                    ),
                ),
                make_source(
                    "DOC-A"
                ),
            )
        )

        data = graph.to_dict()

        self.assertEqual(
            data["graph_version"],
            SUPERSESSION_GRAPH_VERSION,
        )

        self.assertEqual(
            data["source_count"],
            2,
        )

        self.assertEqual(
            data["edge_count"],
            1,
        )

        self.assertFalse(
            data["has_cycle"]
        )


if __name__ == "__main__":
    unittest.main()
