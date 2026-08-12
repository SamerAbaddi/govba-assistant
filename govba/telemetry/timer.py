"""High-resolution timing utilities for GovBA-GAR."""

from __future__ import annotations

from time import perf_counter


class TelemetryTimer:
    """Measure elapsed execution time using a monotonic clock."""

    def __init__(self) -> None:
        self._started_at: float | None = None
        self._elapsed_ms: float | None = None

    def start(self) -> "TelemetryTimer":
        """Start or restart the timer."""
        self._started_at = perf_counter()
        self._elapsed_ms = None
        return self

    def stop(self) -> float:
        """Stop the timer and return elapsed milliseconds."""
        if self._started_at is None:
            raise RuntimeError("TelemetryTimer has not been started.")

        elapsed_seconds = perf_counter() - self._started_at
        self._elapsed_ms = elapsed_seconds * 1000.0
        self._started_at = None

        return self._elapsed_ms

    @property
    def elapsed_ms(self) -> float:
        """Return the latest completed measurement."""
        if self._elapsed_ms is None:
            raise RuntimeError(
                "No completed telemetry measurement is available."
            )

        return self._elapsed_ms

    def __enter__(self) -> "TelemetryTimer":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()
