"""Token-usage normalization for GovBA-GAR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class UsageMetrics:
    """Normalized AI token-usage measurements."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_output_tokens: int = 0


def _read_value(source: Any, name: str, default: Any = None) -> Any:
    """Read a field from either a dictionary or an object."""
    if source is None:
        return default

    if isinstance(source, dict):
        return source.get(name, default)

    return getattr(source, name, default)


def _safe_int(value: Any) -> int:
    """Convert an arbitrary usage value into a non-negative integer."""
    try:
        number = int(value or 0)
    except (TypeError, ValueError):
        return 0

    return max(number, 0)


def extract_usage(response: Any) -> UsageMetrics:
    """Extract normalized token usage without making any API call."""

    usage = _read_value(response, "usage")

    if usage is None:
        return UsageMetrics()

    input_tokens = _safe_int(
        _read_value(usage, "input_tokens", 0)
    )
    output_tokens = _safe_int(
        _read_value(usage, "output_tokens", 0)
    )
    total_tokens = _safe_int(
        _read_value(usage, "total_tokens", 0)
    )

    input_details = _read_value(
        usage,
        "input_tokens_details",
    )
    output_details = _read_value(
        usage,
        "output_tokens_details",
    )

    cached_input_tokens = _safe_int(
        _read_value(input_details, "cached_tokens", 0)
    )

    reasoning_output_tokens = _safe_int(
        _read_value(output_details, "reasoning_tokens", 0)
    )

    if total_tokens == 0 and (input_tokens or output_tokens):
        total_tokens = input_tokens + output_tokens

    return UsageMetrics(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached_input_tokens,
        reasoning_output_tokens=reasoning_output_tokens,
    )
