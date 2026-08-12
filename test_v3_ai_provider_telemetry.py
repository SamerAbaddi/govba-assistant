"""Offline integration tests for GovBA-GAR AI-provider telemetry.

These tests mock the OpenAI client. They make no network request and
require no real API key.
"""

from __future__ import annotations

import json
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

NOT_READY_STATUS = {
    "provider": "OpenAI",
    "configured": False,
    "enabled": False,
    "sdk_available": True,
    "ready": False,
    "model": "test-model",
    "mode": "Rule-based fallback",
    "message": "AI mode is intentionally disabled.",
}


def make_response(
    *,
    output_text: str = "Mock AI response.",
    status: str = "completed",
    response_id: str = "resp_test_123",
    input_tokens: int = 120,
    output_tokens: int = 30,
    total_tokens: int = 150,
    cached_tokens: int = 20,
    reasoning_tokens: int = 10,
    incomplete_reason: str | None = None,
    include_usage: bool = True,
):
    """Build an OpenAI-like response object for offline testing."""

    usage = None

    if include_usage:
        usage = SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            input_tokens_details=SimpleNamespace(
                cached_tokens=cached_tokens,
            ),
            output_tokens_details=SimpleNamespace(
                reasoning_tokens=reasoning_tokens,
            ),
        )

    incomplete_details = None

    if incomplete_reason is not None:
        incomplete_details = SimpleNamespace(
            reason=incomplete_reason,
        )

    return SimpleNamespace(
        status=status,
        id=response_id,
        incomplete_details=incomplete_details,
        usage=usage,
        output_text=output_text,
        output=[],
    )


class TestAIProviderTelemetry(unittest.TestCase):
    """Contract and observability tests for the AI gateway."""

    def test_provider_not_ready_preserves_fallback_contract(self):
        with patch(
            "ai_provider.get_ai_provider_status",
            return_value=NOT_READY_STATUS,
        ), patch(
            "ai_provider.record_event",
        ) as recorder:

            result = ai_provider.request_ai_text(
                "Test instructions.",
                "Test input.",
                max_output_tokens=100,
            )

        self.assertFalse(result["success"])
        self.assertTrue(result["fallback_required"])
        self.assertEqual(
            result["mode"],
            "Rule-based fallback",
        )

        recorder.assert_called_once()

        event = recorder.call_args.args[0]

        self.assertIsInstance(event, TelemetryEvent)
        self.assertFalse(event.success)
        self.assertTrue(event.fallback_used)
        self.assertEqual(
            event.error_type,
            "ProviderNotReady",
        )

    def test_success_preserves_result_and_records_usage(self):
        response = make_response()
        sensitive_input = "SENSITIVE-EMPLOYEE-TEXT"

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

            result = ai_provider.request_ai_text(
                "Test instructions.",
                sensitive_input,
                max_output_tokens=100,
            )

        self.assertTrue(result["success"])
        self.assertFalse(result["fallback_required"])
        self.assertEqual(result["mode"], "AI mode")
        self.assertEqual(
            result["text"],
            "Mock AI response.",
        )

        self.assertEqual(
            result["usage"]["input_tokens"],
            120,
        )
        self.assertEqual(
            result["usage"]["output_tokens"],
            30,
        )
        self.assertEqual(
            result["usage"]["reasoning_tokens"],
            10,
        )
        self.assertEqual(
            result["usage"]["total_tokens"],
            150,
        )

        recorder.assert_called_once()

        event = recorder.call_args.args[0]

        self.assertTrue(event.success)
        self.assertFalse(event.fallback_used)
        self.assertEqual(event.input_tokens, 120)
        self.assertEqual(event.output_tokens, 30)
        self.assertEqual(event.total_tokens, 150)
        self.assertEqual(
            event.cached_input_tokens,
            20,
        )
        self.assertEqual(
            event.reasoning_output_tokens,
            10,
        )
        self.assertGreaterEqual(event.latency_ms, 0)

        telemetry_payload = json.dumps(
            event.to_dict()
        )

        self.assertNotIn(
            sensitive_input,
            telemetry_payload,
        )

        request_kwargs = (
            client.responses.create.call_args.kwargs
        )

        self.assertFalse(request_kwargs["store"])

    def test_missing_usage_is_safe(self):
        response = make_response(
            include_usage=False,
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

            result = ai_provider.request_ai_text(
                "Test instructions.",
                "Test input.",
                max_output_tokens=100,
            )

        self.assertTrue(result["success"])
        self.assertIsNone(result["usage"])

        event = recorder.call_args.args[0]

        self.assertEqual(event.input_tokens, 0)
        self.assertEqual(event.output_tokens, 0)
        self.assertEqual(event.total_tokens, 0)

    def test_openai_exception_preserves_failure_contract(self):
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
            client.responses.create.side_effect = RuntimeError(
                "mock provider failure"
            )

            result = ai_provider.request_ai_text(
                "Test instructions.",
                "Test input.",
                max_output_tokens=100,
            )

        self.assertFalse(result["success"])
        self.assertTrue(result["fallback_required"])
        self.assertIn(
            "RuntimeError",
            result["error"],
        )

        event = recorder.call_args.args[0]

        self.assertFalse(event.success)
        self.assertTrue(event.fallback_used)
        self.assertEqual(
            event.error_type,
            "RuntimeError",
        )

    def test_empty_response_is_recorded(self):
        response = make_response(
            output_text="",
            status="completed",
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

            result = ai_provider.request_ai_text(
                "Test instructions.",
                "Test input.",
                max_output_tokens=100,
            )

        self.assertFalse(result["success"])
        self.assertTrue(result["fallback_required"])
        self.assertIn(
            "no visible text",
            result["error"],
        )

        event = recorder.call_args.args[0]

        self.assertEqual(
            event.error_type,
            "EmptyResponse",
        )
        self.assertFalse(event.success)

    def test_incomplete_response_is_recorded(self):
        response = make_response(
            output_text="",
            status="incomplete",
            incomplete_reason="max_output_tokens",
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

            result = ai_provider.request_ai_text(
                "Test instructions.",
                "Test input.",
                max_output_tokens=100,
            )

        self.assertFalse(result["success"])
        self.assertTrue(result["fallback_required"])
        self.assertEqual(
            result["response_status"],
            "incomplete",
        )

        event = recorder.call_args.args[0]

        self.assertEqual(
            event.error_type,
            "IncompleteResponse",
        )

    def test_structured_json_contract_is_preserved(self):
        response = make_response(
            output_text='{"summary":"mock summary"}'
        )

        schema = {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                }
            },
            "required": ["summary"],
            "additionalProperties": False,
        }

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
                "Return JSON.",
                "Test input.",
                schema,
                max_output_tokens=100,
            )

        self.assertTrue(result["success"])
        self.assertFalse(result["fallback_required"])

        self.assertEqual(
            result["data"],
            {
                "summary": "mock summary",
            },
        )

        event = recorder.call_args.args[0]

        self.assertTrue(event.success)

        request_kwargs = (
            client.responses.create.call_args.kwargs
        )

        self.assertEqual(
            request_kwargs["text"]["format"]["type"],
            "json_schema",
        )
        self.assertTrue(
            request_kwargs["text"]["format"]["strict"]
        )

    def test_telemetry_failure_cannot_break_ai_success(self):
        response = make_response()

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
            side_effect=OSError(
                "simulated telemetry failure"
            ),
        ):

            client = openai_class.return_value
            client.responses.create.return_value = response

            result = ai_provider.request_ai_text(
                "Test instructions.",
                "Test input.",
                max_output_tokens=100,
            )

        self.assertTrue(result["success"])
        self.assertFalse(result["fallback_required"])
        self.assertEqual(
            result["text"],
            "Mock AI response.",
        )

    def test_validation_contract_is_unchanged(self):
        with patch(
            "ai_provider.record_event",
        ) as recorder:

            with self.assertRaises(ValueError):
                ai_provider.request_ai_text(
                    "",
                    "Test input.",
                    max_output_tokens=100,
                )

        recorder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
