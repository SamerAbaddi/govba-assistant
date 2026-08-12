"""Privacy-safe telemetry event model for GovBA-GAR."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from govba.version import ARCHITECTURE_VERSION


def _new_run_id() -> str:
    """Return a unique identifier without embedding user information."""
    return uuid4().hex


def _utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class TelemetryEvent:
    """Privacy-safe measurement record for one GovBA operation."""

    run_id: str = field(default_factory=_new_run_id)
    timestamp_utc: str = field(default_factory=_utc_timestamp)

    architecture_version: str = ARCHITECTURE_VERSION

    task_type: str = "unknown"
    language: str = "unknown"
    mode: str = "unknown"
    model: str = ""

    latency_ms: float = 0.0

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_output_tokens: int = 0

    success: bool = False
    fallback_used: bool = False

    retrieval_used: bool = False
    source_count: int = 0

    verification_status: str = "not_run"
    abstained: bool = False

    error_type: str = ""

    def to_dict(self) -> dict:
        """Return the event as a JSON-serializable dictionary."""
        return asdict(self)
