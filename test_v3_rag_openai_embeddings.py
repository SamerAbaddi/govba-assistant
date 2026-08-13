"""Offline mocked tests for the GovBA-GAR OpenAI embedding adapter.

No external OpenAI API request is made by this test module.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from govba.rag import (
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    OpenAIEmbeddingProviderError,
)


def make_response(
    vectors,
):
    """Create an OpenAI-like embedding response."""

    return SimpleNamespace(
        data=[
            SimpleNamespace(
                index=index,
                embedding=vector,
            )
            for index, vector
            in enumerate(vectors)
        ],
        model="test-embedding-model",
        usage=SimpleNamespace(
            prompt_tokens=10,
            total_tokens=10,
        ),
    )


class TestOpenAIEmbeddingConfiguration(
    unittest.TestCase
):

    def test_provider_satisfies_embedding_contract(self):
        client = MagicMock()

        provider = OpenAIEmbeddingProvider(
            model="test-model",
            client=client,
        )

        self.assertIsInstance(
            provider,
            EmbeddingProvider,
        )

    def test_configuration_properties_are_available(self):
        provider = OpenAIEmbeddingProvider(
            model=" test-model ",
            dimensions=256,
            batch_size=32,
            client=MagicMock(),
        )

        self.assertEqual(
            provider.model,
            "test-model",
        )

        self.assertEqual(
            provider.dimensions,
            256,
        )

        self.assertEqual(
            provider.batch_size,
            32,
        )

    def test_invalid_configuration_is_rejected(self):
        invalid_cases = (
            {
                "model": " ",
            },
            {
                "model": "test",
                "dimensions": 0,
            },
            {
                "model": "test",
                "batch_size": 0,
            },
            {
                "model": "test",
                "batch_size": 257,
            },
            {
                "model": "test",
                "timeout": 0,
            },
        )

        for kwargs in invalid_cases:
            with self.subTest(
                kwargs=kwargs
            ):
                with self.assertRaises(
                    ValueError
                ):
                    OpenAIEmbeddingProvider(
                        client=MagicMock(),
                        **kwargs,
                    )


class TestOpenAIEmbeddingRequests(
    unittest.TestCase
):

    def test_documents_are_embedded_in_order(self):
        client = MagicMock()

        response = make_response(
            (
                (
                    1.0,
                    0.0,
                ),
                (
                    0.0,
                    1.0,
                ),
            )
        )

        response.data = (
            response.data[1],
            response.data[0],
        )

        client.embeddings.create.return_value = (
            response
        )

        provider = OpenAIEmbeddingProvider(
            model="test-model",
            client=client,
        )

        vectors = provider.embed_documents(
            (
                "Document A",
                "Document B",
            )
        )

        self.assertEqual(
            vectors,
            (
                (
                    1.0,
                    0.0,
                ),
                (
                    0.0,
                    1.0,
                ),
            ),
        )

    def test_query_uses_embedding_endpoint(self):
        client = MagicMock()

        client.embeddings.create.return_value = (
            make_response(
                (
                    (
                        0.25,
                        0.75,
                    ),
                )
            )
        )

        provider = OpenAIEmbeddingProvider(
            model="test-model",
            client=client,
        )

        vector = provider.embed_query(
            " service requirements "
        )

        self.assertEqual(
            vector,
            (
                0.25,
                0.75,
            ),
        )

        client.embeddings.create.assert_called_once_with(
            model="test-model",
            input=[
                "service requirements"
            ],
            encoding_format="float",
        )

    def test_dimensions_are_sent_when_configured(self):
        client = MagicMock()

        client.embeddings.create.return_value = (
            make_response(
                (
                    (
                        1.0,
                        0.0,
                    ),
                )
            )
        )

        provider = OpenAIEmbeddingProvider(
            model="test-model",
            dimensions=2,
            client=client,
        )

        provider.embed_query(
            "query"
        )

        request = (
            client
            .embeddings
            .create
            .call_args
            .kwargs
        )

        self.assertEqual(
            request["dimensions"],
            2,
        )

    def test_dimensions_are_omitted_by_default(self):
        client = MagicMock()

        client.embeddings.create.return_value = (
            make_response(
                (
                    (
                        1.0,
                        0.0,
                    ),
                )
            )
        )

        provider = OpenAIEmbeddingProvider(
            model="test-model",
            client=client,
        )

        provider.embed_query(
            "query"
        )

        request = (
            client
            .embeddings
            .create
            .call_args
            .kwargs
        )

        self.assertNotIn(
            "dimensions",
            request,
        )

    def test_document_batching_is_deterministic(self):
        client = MagicMock()

        client.embeddings.create.side_effect = (
            make_response(
                (
                    (
                        1.0,
                        0.0,
                    ),
                    (
                        2.0,
                        0.0,
                    ),
                )
            ),
            make_response(
                (
                    (
                        3.0,
                        0.0,
                    ),
                )
            ),
        )

        provider = OpenAIEmbeddingProvider(
            model="test-model",
            batch_size=2,
            client=client,
        )

        vectors = provider.embed_documents(
            (
                "One",
                "Two",
                "Three",
            )
        )

        self.assertEqual(
            len(vectors),
            3,
        )

        self.assertEqual(
            client.embeddings.create.call_count,
            2,
        )

    def test_empty_document_sequence_needs_no_request(self):
        client = MagicMock()

        provider = OpenAIEmbeddingProvider(
            model="test-model",
            client=client,
        )

        self.assertEqual(
            provider.embed_documents(
                ()
            ),
            (),
        )

        client.embeddings.create.assert_not_called()

    def test_blank_query_is_rejected_before_request(self):
        client = MagicMock()

        provider = OpenAIEmbeddingProvider(
            model="test-model",
            client=client,
        )

        with self.assertRaises(ValueError):
            provider.embed_query(
                "   "
            )

        client.embeddings.create.assert_not_called()

    def test_blank_document_is_rejected_before_request(self):
        client = MagicMock()

        provider = OpenAIEmbeddingProvider(
            model="test-model",
            client=client,
        )

        with self.assertRaises(ValueError):
            provider.embed_documents(
                (
                    "Valid",
                    " ",
                )
            )

        client.embeddings.create.assert_not_called()

    def test_string_is_not_accepted_as_document_sequence(self):
        provider = OpenAIEmbeddingProvider(
            model="test-model",
            client=MagicMock(),
        )

        with self.assertRaises(TypeError):
            provider.embed_documents(
                "not a sequence of documents"
            )

    def test_response_count_mismatch_is_rejected(self):
        client = MagicMock()

        client.embeddings.create.return_value = (
            make_response(
                (
                    (
                        1.0,
                        0.0,
                    ),
                )
            )
        )

        provider = OpenAIEmbeddingProvider(
            model="test-model",
            client=client,
        )

        with self.assertRaises(
            OpenAIEmbeddingProviderError
        ):
            provider.embed_documents(
                (
                    "One",
                    "Two",
                )
            )

    def test_duplicate_response_index_is_rejected(self):
        client = MagicMock()

        response = make_response(
            (
                (
                    1.0,
                    0.0,
                ),
                (
                    0.0,
                    1.0,
                ),
            )
        )

        response.data[1].index = 0

        client.embeddings.create.return_value = (
            response
        )

        provider = OpenAIEmbeddingProvider(
            model="test-model",
            client=client,
        )

        with self.assertRaises(
            OpenAIEmbeddingProviderError
        ):
            provider.embed_documents(
                (
                    "One",
                    "Two",
                )
            )

    def test_invalid_vector_is_rejected(self):
        client = MagicMock()

        client.embeddings.create.return_value = (
            make_response(
                (
                    (),
                )
            )
        )

        provider = OpenAIEmbeddingProvider(
            model="test-model",
            client=client,
        )

        with self.assertRaises(
            OpenAIEmbeddingProviderError
        ):
            provider.embed_query(
                "query"
            )

    def test_provider_exception_is_safely_wrapped(self):
        client = MagicMock()

        client.embeddings.create.side_effect = (
            RuntimeError(
                "secret provider diagnostic"
            )
        )

        provider = OpenAIEmbeddingProvider(
            model="test-model",
            client=client,
        )

        with self.assertRaises(
            OpenAIEmbeddingProviderError
        ) as context:
            provider.embed_query(
                "query"
            )

        self.assertEqual(
            str(context.exception),
            "OpenAI embedding request failed.",
        )

        self.assertNotIn(
            "secret provider diagnostic",
            str(context.exception),
        )


if __name__ == "__main__":
    unittest.main()
