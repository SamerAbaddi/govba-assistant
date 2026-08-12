"""Offline tests for GovBA-GAR telemetry.

These tests make no network requests and require no OpenAI API key.
"""

from __future__ import annotations

import json
import time
import unittest
from types import SimpleNamespace

from govba.telemetry import (
    TelemetryEvent,
    TelemetryTimer,
    UsageMetrics,
    extract_usage,
)


class TestTelemetryEvent(unittest.TestCase):
    """Tests for the privacy-safe telemetry event schema."""

    def test_event_has_unique_run_id(self):
        first = TelemetryEvent()
        second = TelemetryEvent()

        self.assertTrue(first.run_id)
        self.assertTrue(second.run_id)
        self.assertNotEqual(first.run_id, second.run_id)

    def test_event_serializes_to_json(self):
        event = TelemetryEvent(
            task_type="email_summary",
            language="en",
            mode="offline",
            model="test-model",
            success=True,
        )

        data = event.to_dict()

        serialized = json.dumps(data)

        self.assertTrue(serialized)
        self.assertEqual(data["task_type"], "email_summary")
        self.assertTrue(data["success"])

    def test_schema_excludes_sensitive_content_fields(self):
        data = TelemetryEvent().to_dict()

        prohibited_fields = {
            "prompt",
            "raw_prompt",
            "email_body",
            "document_text",
            "api_key",
            "access_code",
            "national_id",
            "employee_name",
            "citizen_name",
        }

        self.assertTrue(
            prohibited_fields.isdisjoint(data.keys())
        )


class TestTelemetryTimer(unittest.TestCase):
    """Tests for monotonic execution timing."""

    def test_timer_measures_positive_duration(self):
        timer = TelemetryTimer()

        timer.start()
        time.sleep(0.005)
        elapsed = timer.stop()

        self.assertGreater(elapsed, 0)
        self.assertEqual(elapsed, timer.elapsed_ms)

    def test_context_manager_records_duration(self):
        with TelemetryTimer() as timer:
            time.sleep(0.005)

        self.assertGreater(timer.elapsed_ms, 0)

    def test_stop_before_start_raises_error(self):
        timer = TelemetryTimer()

        with self.assertRaises(RuntimeError):
            timer.stop()


class TestUsageExtraction(unittest.TestCase):
    """Tests for provider-independent usage normalization."""

    def test_dictionary_usage(self):
        response = {
            "usage": {
                "input_tokens": 120,
                "output_tokens": 30,
                "total_tokens": 150,
                "input_tokens_details": {
                    "cached_tokens": 20,
                },
                "output_tokens_details": {
                    "reasoning_tokens": 10,
                },
            }
        }

        usage = extract_usage(response)

        self.assertEqual(usage.input_tokens, 120)
        self.assertEqual(usage.output_tokens, 30)
        self.assertEqual(usage.total_tokens, 150)
        self.assertEqual(usage.cached_input_tokens, 20)
        self.assertEqual(usage.reasoning_output_tokens, 10)

    def test_object_usage(self):
        response = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=200,
                output_tokens=50,
                total_tokens=250,
                input_tokens_details=SimpleNamespace(
                    cached_tokens=25
                ),
                output_tokens_details=SimpleNamespace(
                    reasoning_tokens=12
                ),
            )
        )

        usage = extract_usage(response)

        self.assertEqual(usage.input_tokens, 200)
        self.assertEqual(usage.output_tokens, 50)
        self.assertEqual(usage.total_tokens, 250)
        self.assertEqual(usage.cached_input_tokens, 25)
        self.assertEqual(usage.reasoning_output_tokens, 12)

    def test_missing_usage_returns_zeroes(self):
        usage = extract_usage({})

        self.assertEqual(usage, UsageMetrics())

    def test_total_can_be_derived(self):
        response = {
            "usage": {
                "input_tokens": 80,
                "output_tokens": 20,
            }
        }

        usage = extract_usage(response)

        self.assertEqual(usage.total_tokens, 100)

    def test_negative_values_are_not_accepted(self):
        response = {
            "usage": {
                "input_tokens": -10,
                "output_tokens": -5,
                "total_tokens": -15,
            }
        }

        usage = extract_usage(response)

        self.assertEqual(usage.input_tokens, 0)
        self.assertEqual(usage.output_tokens, 0)
        self.assertEqual(usage.total_tokens, 0)


if __name__ == "__main__":
    unittest.main()
