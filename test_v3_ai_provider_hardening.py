"""Offline regression tests for GovBA-GAR structured AI responses.

All OpenAI behavior is mocked. No external API request is made.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import ai_provider
from govba.telemetry import TelemetryEvent


READY_STATUS = {
    "provider": "OpenAI",
    "configured": True,
    "enabled": True,
    "sdk_available": True,
    "ready": True,
    "model": "test-model",
    "mode": "AI mode",
    "message": "AI mode is enabled and ready.",
}


def make_response(output_text: str):
    """Return an OpenAI-like response for offline testing."""

    return SimpleNamespace(
        status="completed",
        id="resp_hardening_test",
        incomplete_details=None,
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            input_tokens_details=SimpleNamespace(
                cached_tokens=0,
            ),
            output_tokens_details=SimpleNamespace(
                reasoning_tokens=0,
            ),
        ),
        output_text=output_text,
        output=[],
    )


def test_schema():
    """Return the structured-output schema used by the tests."""

    return {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
            }
        },
        "required": ["summary"],
        "additionalProperties": False,
    }


class TestStructuredResponseHardening(unittest.TestCase):
    """Regression tests for structured provider output."""

    def test_malformed_json_requires_fallback(self):
        response = make_response(
            '{"summary": "broken"'
        )

        with patch(
            "ai_provider.get_ai_provider_status",
            return_value=READY_STATUS,
        ), patch(
            "ai_provider._read_setting",
            return_value="fake-test-key",
        ), patch(
            "openai.OpenAI",
        ) as openai_class, patch(
            "ai_provider.record_event",
        ) as recorder:

            client = openai_class.return_value
            client.responses.create.return_value = response

            result = ai_provider.request_ai_json(
                "Return structured JSON.",
                "Test input.",
                test_schema(),
                max_output_tokens=100,
            )

        self.assertFalse(result["success"])
        self.assertTrue(result["fallback_required"])
        self.assertIsNone(result["data"])

        self.assertEqual(
            result["error"],
            "The AI provider returned invalid structured JSON.",
        )

        recorder.assert_called_once()

        event = recorder.call_args.args[0]

        self.assertIsInstance(event, TelemetryEvent)
        self.assertFalse(event.success)
        self.assertTrue(event.fallback_used)
        self.assertEqual(
            event.error_type,
            "JSONDecodeError",
        )

    def test_valid_json_remains_successful(self):
        response = make_response(
            '{"summary":"valid response"}'
        )

        with patch(
            "ai_provider.get_ai_provider_status",
            return_value=READY_STATUS,
        ), patch(
            "ai_provider._read_setting",
            return_value="fake-test-key",
        ), patch(
            "openai.OpenAI",
        ) as openai_class, patch(
            "ai_provider.record_event",
        ) as recorder:

            client = openai_class.return_value
            client.responses.create.return_value = response

            result = ai_provider.request_ai_json(
                "Return structured JSON.",
                "Test input.",
                test_schema(),
                max_output_tokens=100,
            )

        self.assertTrue(result["success"])
        self.assertFalse(result["fallback_required"])

        self.assertEqual(
            result["data"],
            {
                "summary": "valid response",
            },
        )

        self.assertIsNone(result["error"])

        event = recorder.call_args.args[0]

        self.assertTrue(event.success)
        self.assertFalse(event.fallback_used)
        self.assertEqual(event.error_type, "")


if __name__ == "__main__":
    unittest.main()
