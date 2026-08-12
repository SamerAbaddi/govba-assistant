"""Offline tests for the GovBA-GAR telemetry recorder.

No OpenAI API call, network request, or API key is required.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from govba.telemetry import (
    TelemetryEvent,
    get_telemetry_path,
    record_event,
    telemetry_enabled,
)


class TestTelemetryRecorder(unittest.TestCase):
    """Tests for fail-safe privacy-safe telemetry recording."""

    def test_disabled_by_default(self):
        with patch.dict(
            os.environ,
            {},
            clear=True,
        ):
            self.assertFalse(telemetry_enabled())

    def test_disabled_recorder_writes_nothing(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "events.jsonl"

            written = record_event(
                TelemetryEvent(success=True),
                path=path,
                enabled=False,
            )

            self.assertFalse(written)
            self.assertFalse(path.exists())

    def test_enabled_recorder_writes_jsonl(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "events.jsonl"

            event = TelemetryEvent(
                task_type="email_summary",
                language="en",
                mode="ai",
                model="test-model",
                success=True,
            )

            written = record_event(
                event,
                path=path,
                enabled=True,
            )

            self.assertTrue(written)
            self.assertTrue(path.exists())

            lines = path.read_text(
                encoding="utf-8"
            ).splitlines()

            self.assertEqual(len(lines), 1)

            stored = json.loads(lines[0])

            self.assertEqual(
                stored["run_id"],
                event.run_id,
            )
            self.assertEqual(
                stored["task_type"],
                "email_summary",
            )

    def test_recorder_appends_multiple_events(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "events.jsonl"

            first = TelemetryEvent(
                task_type="first",
                success=True,
            )
            second = TelemetryEvent(
                task_type="second",
                success=True,
            )

            self.assertTrue(
                record_event(
                    first,
                    path=path,
                    enabled=True,
                )
            )

            self.assertTrue(
                record_event(
                    second,
                    path=path,
                    enabled=True,
                )
            )

            lines = path.read_text(
                encoding="utf-8"
            ).splitlines()

            self.assertEqual(len(lines), 2)

            first_data = json.loads(lines[0])
            second_data = json.loads(lines[1])

            self.assertEqual(
                first_data["task_type"],
                "first",
            )
            self.assertEqual(
                second_data["task_type"],
                "second",
            )

    def test_environment_path_is_supported(self):
        with TemporaryDirectory() as temp_dir:
            expected = (
                Path(temp_dir)
                / "research"
                / "events.jsonl"
            )

            with patch.dict(
                os.environ,
                {
                    "GOVBA_TELEMETRY_PATH": str(
                        expected
                    ),
                },
            ):
                self.assertEqual(
                    get_telemetry_path(),
                    expected,
                )

    def test_storage_failure_is_fail_safe(self):
        with TemporaryDirectory() as temp_dir:
            invalid_destination = Path(temp_dir)
            event = TelemetryEvent(success=True)

            written = record_event(
                event,
                path=invalid_destination,
                enabled=True,
            )

            self.assertFalse(written)

    def test_non_event_object_is_rejected(self):
        written = record_event(
            {"unsafe": "object"},  # type: ignore[arg-type]
            enabled=True,
        )

        self.assertFalse(written)


if __name__ == "__main__":
    unittest.main()
