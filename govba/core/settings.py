"""Shared privacy-safe configuration access for GovBA-GAR.

Configuration precedence is intentionally preserved:

1. Non-blank environment variable.
2. Non-blank Streamlit secret.
3. Caller-supplied default.

This module never logs, serializes, or exposes secret values.
"""

from __future__ import annotations

import os


_TRUE_VALUES = frozenset(
    {
        "1",
        "true",
        "yes",
        "on",
        "enabled",
    }
)


def _read_streamlit_secret(
    name: str,
) -> str | None:
    """Read one Streamlit secret safely when Streamlit is available."""

    try:
        import streamlit as st
    except ImportError:
        return None

    try:
        value = st.secrets.get(name)
    except Exception:
        return None

    if value is None:
        return None

    cleaned = str(value).strip()

    return cleaned or None


def read_setting(
    name: str,
    default: str | None = None,
) -> str | None:
    """Read a setting using GovBA's established precedence."""

    environment_value = os.getenv(
        name
    )

    if environment_value is not None:
        cleaned = (
            environment_value.strip()
        )

        if cleaned:
            return cleaned

    streamlit_value = (
        _read_streamlit_secret(
            name
        )
    )

    if streamlit_value is not None:
        return streamlit_value

    return default


def read_boolean_setting(
    name: str,
    default: bool = False,
) -> bool:
    """Read a boolean setting using the established GovBA convention."""

    raw_value = read_setting(
        name
    )

    if raw_value is None:
        return default

    return (
        raw_value.lower()
        in _TRUE_VALUES
    )
