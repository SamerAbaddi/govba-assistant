"""Offline tests for GovBA-GAR embedding telemetry.

All OpenAI behavior is mocked. No external API request is made.
"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from govba.rag import (
    OpenAIEmbeddingProvider,
    OpenAIEmbeddingProviderError,
)
from govba.telemetry.events import TelemetryEvent
from govba.telemetry.usage import extract_usage


def make_response(
    vectors,
    *,
    prompt_tokens=5,
    include_usage=True,
):
    """Return an OpenAI-like mocked embedding response."""

    response = SimpleNamespace(
        data=[
            SimpleNamespace(
                index=index,
                embedding=vector,
            )
            for index, vector
            in enumerate(vectors)
        ],
    )

    if include_usage:
        response.usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            total_tokens=prompt_tokens,
        )

    return response


class TestEmbeddingUsageCompatibility(unittest.TestCase):

    def test_prompt_tokens_map_to_input_tokens(self):
        response = make_response(
            (
                (
                    1.0,
                    0.0,
                ),
            ),
            prompt_tokens=9,
        )

        usage = extract_usage(
            response
        )

        self.assertEqual(
            usage.input_tokens,
            9,
        )

        self.assertEqual(
            usage.output_tokens,
            0,
        )

        self.assertEqual(
            usage.total_tokens,
            9,
        )


class TestEmbeddingTelemetry(unittest.TestCase):

    def test_successful_query_records_telemetry(self):
        client = MagicMock()

        client.embeddings.create.return_value = (
            make_response(
                (
                    (
                        0.25,
                        0.75,
                    ),
                ),
                prompt_tokens=7,
            )
        )

        provider = OpenAIEmbeddingProvider(
            model="test-embedding-model",
            client=client,
        )

        with patch(
            "govba.rag.openai_embeddings.record_event"
        ) as recorder:
            vector = provider.embed_query(
                "service requirements"
            )

        self.assertEqual(
            vector,
            (
                0.25,
                0.75,
            ),
        )

        recorder.assert_called_once()

        event = recorder.call_args.args[0]

        self.assertIsInstance(
            event,
            TelemetryEvent,
        )

        self.assertEqual(
            event.task_type,
            "embedding_query",
        )

        self.assertEqual(
            event.mode,
            "embedding",
        )

        self.assertEqual(
            event.model,
            "test-embedding-model",
        )

        self.assertEqual(
            event.input_tokens,
            7,
        )

        self.assertEqual(
            event.output_tokens,
            0,
        )

        self.assertEqual(
            event.total_tokens,
            7,
        )

        self.assertTrue(
            event.success
        )

        self.assertEqual(
            event.error_type,
            "",
        )

        self.assertGreaterEqual(
            event.latency_ms,
            0.0,
        )

    def test_missing_usage_is_safe(self):
        client = MagicMock()

        client.embeddings.create.return_value = (
            make_response(
                (
                    (
                        1.0,
                        0.0,
                    ),
                ),
                include_usage=False,
            )
        )

        provider = OpenAIEmbeddingProvider(
            model="test-model",
            client=client,
        )

        with patch(
            "govba.rag.openai_embeddings.record_event"
        ) as recorder:
            provider.embed_query(
                "query"
            )

        event = recorder.call_args.args[0]

        self.assertEqual(
            event.input_tokens,
            0,
        )

        self.assertEqual(
            event.total_tokens,
            0,
        )

        self.assertTrue(
            event.success
        )

    def test_raw_query_is_not_written_to_event(self):
        sensitive_text = (
            "PRIVATE-QUERY-CONTENT-938271"
        )

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

        with patch(
            "govba.rag.openai_embeddings.record_event"
        ) as recorder:
            provider.embed_query(
                sensitive_text
            )

        event = recorder.call_args.args[0]

        serialized = json.dumps(
            event.to_dict(),
            ensure_ascii=False,
        )

        self.assertNotIn(
            sensitive_text,
            serialized,
        )

    def test_telemetry_failure_cannot_break_success(self):
        client = MagicMock()

        client.embeddings.create.return_value = (
            make_response(
                (
                    (
                        0.2,
                        0.8,
                    ),
                )
            )
        )

        provider = OpenAIEmbeddingProvider(
            model="test-model",
            client=client,
        )

        with patch(
            "govba.rag.openai_embeddings.record_event",
            side_effect=RuntimeError(
                "telemetry unavailable"
            ),
        ):
            vector = provider.embed_query(
                "safe query"
            )

        self.assertEqual(
            vector,
            (
                0.2,
                0.8,
            ),
        )

    def test_provider_failure_records_safe_error_type(self):
        client = MagicMock()

        client.embeddings.create.side_effect = (
            RuntimeError(
                "SECRET PROVIDER DIAGNOSTIC"
            )
        )

        provider = OpenAIEmbeddingProvider(
            model="test-model",
            client=client,
        )

        with patch(
            "govba.rag.openai_embeddings.record_event"
        ) as recorder:
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

        event = recorder.call_args.args[0]

        self.assertFalse(
            event.success
        )

        self.assertEqual(
            event.error_type,
            "RuntimeError",
        )

        serialized = json.dumps(
            event.to_dict(),
            ensure_ascii=False,
        )

        self.assertNotIn(
            "SECRET PROVIDER DIAGNOSTIC",
            serialized,
        )

    def test_document_batches_record_individual_events(self):
        client = MagicMock()

        client.embeddings.create.side_effect = (
            make_response(
                (
                    (
                        1.0,
                        0.0,
                    ),
                    (
                        0.9,
                        0.1,
                    ),
                ),
                prompt_tokens=10,
            ),
            make_response(
                (
                    (
                        0.0,
                        1.0,
                    ),
                ),
                prompt_tokens=4,
            ),
        )

        provider = OpenAIEmbeddingProvider(
            model="test-model",
            batch_size=2,
            client=client,
        )

        with patch(
            "govba.rag.openai_embeddings.record_event"
        ) as recorder:
            vectors = provider.embed_documents(
                (
                    "Document one",
                    "Document two",
                    "Document three",
                )
            )

        self.assertEqual(
            len(vectors),
            3,
        )

        self.assertEqual(
            recorder.call_count,
            2,
        )

        events = [
            call.args[0]
            for call in recorder.call_args_list
        ]

        self.assertTrue(
            all(
                event.task_type
                == "embedding_documents_batch"
                for event in events
            )
        )

        self.assertEqual(
            [
                event.input_tokens
                for event in events
            ],
            [
                10,
                4,
            ],
        )

    def test_document_text_is_not_written_to_event(self):
        sensitive_document = (
            "PRIVATE-DOCUMENT-CONTENT-483920"
        )

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

        with patch(
            "govba.rag.openai_embeddings.record_event"
        ) as recorder:
            provider.embed_documents(
                (
                    sensitive_document,
                )
            )

        event = recorder.call_args.args[0]

        serialized = json.dumps(
            event.to_dict(),
            ensure_ascii=False,
        )

        self.assertNotIn(
            sensitive_document,
            serialized,
        )


if __name__ == "__main__":
    unittest.main()
