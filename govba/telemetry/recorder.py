"""Fail-safe telemetry recording for GovBA-GAR.

Telemetry is disabled by default. When enabled, privacy-safe
TelemetryEvent records are appended as JSON Lines (JSONL).

Telemetry failures must never interrupt normal GovBA operation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock

from govba.telemetry.events import TelemetryEvent


_DEFAULT_TELEMETRY_PATH = Path(
    ".govba_runtime/telemetry/events.jsonl"
)

_TRUE_VALUES = {
    "1",
    "true",
    "yes",
    "on",
}

_WRITE_LOCK = Lock()


def telemetry_enabled() -> bool:
    """Return whether telemetry recording is explicitly enabled."""
    value = os.getenv(
        "GOVBA_TELEMETRY_ENABLED",
        "false",
    )

    return value.strip().lower() in _TRUE_VALUES


def get_telemetry_path() -> Path:
    """Return the configured telemetry JSONL destination."""
    configured = os.getenv(
        "GOVBA_TELEMETRY_PATH",
        "",
    ).strip()

    if configured:
        return Path(configured).expanduser()

    return _DEFAULT_TELEMETRY_PATH


def record_event(
    event: TelemetryEvent,
    *,
    path: str | Path | None = None,
    enabled: bool | None = None,
) -> bool:
    """Append one privacy-safe telemetry event.

    Returns True when the event was written.

    Returns False when telemetry is disabled, the supplied object is
    invalid, or storage fails. Recording failures are intentionally
    fail-safe and must not interrupt GovBA.
    """

    if not isinstance(event, TelemetryEvent):
        return False

    should_record = (
        telemetry_enabled()
        if enabled is None
        else bool(enabled)
    )

    if not should_record:
        return False

    destination = (
        Path(path)
        if path is not None
        else get_telemetry_path()
    )

    try:
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = json.dumps(
            event.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

        with _WRITE_LOCK:
            with destination.open(
                "a",
                encoding="utf-8",
            ) as file:
                file.write(payload)
                file.write("\n")
                file.flush()

        return True

    except (OSError, TypeError, ValueError):
        # Telemetry must never break normal GovBA operation.
        return False
