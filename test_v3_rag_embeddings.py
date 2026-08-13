"""Offline tests for GovBA-GAR embedding infrastructure."""

from __future__ import annotations

import math
import unittest

from govba.rag import (
    EmbeddingProvider,
    cosine_similarity,
    cosine_to_unit_interval,
    embedding_norm,
    normalize_embedding_vector,
)


class TestEmbeddingVectorValidation(unittest.TestCase):
    """Tests for provider-independent vector normalization."""

    def test_integer_and_float_values_normalize_to_floats(self):
        vector = normalize_embedding_vector(
            (1, 2.5, -3)
        )

        self.assertEqual(
            vector,
            (
                1.0,
                2.5,
                -3.0,
            ),
        )

    def test_empty_vector_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_embedding_vector(
                ()
            )

    def test_boolean_value_is_rejected(self):
        with self.assertRaises(TypeError):
            normalize_embedding_vector(
                (
                    1.0,
                    True,
                )
            )

    def test_non_numeric_value_is_rejected(self):
        with self.assertRaises(TypeError):
            normalize_embedding_vector(
                (
                    1.0,
                    "invalid",
                )
            )

    def test_non_finite_values_are_rejected(self):
        for value in (
            math.nan,
            math.inf,
            -math.inf,
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_embedding_vector(
                        (
                            1.0,
                            value,
                        )
                    )

    def test_embedding_norm_is_correct(self):
        self.assertEqual(
            embedding_norm(
                (
                    3.0,
                    4.0,
                )
            ),
            5.0,
        )


class TestCosineSimilarity(unittest.TestCase):
    """Tests for deterministic semantic vector mathematics."""

    def test_identical_vectors_have_similarity_one(self):
        self.assertAlmostEqual(
            cosine_similarity(
                (
                    1.0,
                    2.0,
                ),
                (
                    1.0,
                    2.0,
                ),
            ),
            1.0,
            places=12,
        )

    def test_orthogonal_vectors_have_similarity_zero(self):
        self.assertEqual(
            cosine_similarity(
                (
                    1.0,
                    0.0,
                ),
                (
                    0.0,
                    1.0,
                ),
            ),
            0.0,
        )

    def test_opposite_vectors_have_similarity_negative_one(self):
        self.assertEqual(
            cosine_similarity(
                (
                    1.0,
                    0.0,
                ),
                (
                    -1.0,
                    0.0,
                ),
            ),
            -1.0,
        )

    def test_dimension_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            cosine_similarity(
                (
                    1.0,
                    2.0,
                ),
                (
                    1.0,
                    2.0,
                    3.0,
                ),
            )

    def test_zero_vector_is_rejected(self):
        with self.assertRaises(ValueError):
            cosine_similarity(
                (
                    0.0,
                    0.0,
                ),
                (
                    1.0,
                    0.0,
                ),
            )

    def test_cosine_similarity_maps_to_unit_interval(self):
        self.assertEqual(
            cosine_to_unit_interval(
                -1.0
            ),
            0.0,
        )

        self.assertEqual(
            cosine_to_unit_interval(
                0.0
            ),
            0.5,
        )

        self.assertEqual(
            cosine_to_unit_interval(
                1.0
            ),
            1.0,
        )


class TestEmbeddingProviderProtocol(unittest.TestCase):
    """Tests for provider abstraction compatibility."""

    def test_structural_embedding_provider_contract(self):
        class FakeEmbeddingProvider:
            def embed_documents(
                self,
                texts,
            ):
                return [
                    (
                        float(index + 1),
                        0.0,
                    )
                    for index, _ in enumerate(
                        texts
                    )
                ]

            def embed_query(
                self,
                text,
            ):
                return (
                    1.0,
                    0.0,
                )

        provider = FakeEmbeddingProvider()

        self.assertIsInstance(
            provider,
            EmbeddingProvider,
        )

        document_vectors = (
            provider.embed_documents(
                (
                    "document one",
                    "document two",
                )
            )
        )

        self.assertEqual(
            len(document_vectors),
            2,
        )

        self.assertEqual(
            provider.embed_query(
                "query"
            ),
            (
                1.0,
                0.0,
            ),
        )


if __name__ == "__main__":
    unittest.main()
