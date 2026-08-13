"""Safe embedding-provider configuration for GovBA-GAR.

Embedding retrieval is independently configurable from generative AI.
No secret values are included in the public provider status.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.util import find_spec

from govba.core.settings import (
    read_boolean_setting,
    read_setting,
)


DEFAULT_OPENAI_EMBEDDING_MODEL = (
    "text-embedding-3-small"
)


@dataclass(frozen=True)
class EmbeddingProviderStatus:
    """Privacy-safe status of the configured embedding provider."""

    provider: str
    configured: bool
    enabled: bool
    sdk_available: bool
    ready: bool
    model: str
    mode: str
    message: str


def _sdk_available() -> bool:
    """Return whether the OpenAI Python SDK can be located."""

    try:
        return find_spec("openai") is not None
    except (
        ImportError,
        ValueError,
    ):
        return False


def get_embedding_provider_status() -> dict[str, object]:
    """Return privacy-safe embedding configuration status."""

    api_key = read_setting(
        "OPENAI_API_KEY"
    )

    enabled = read_boolean_setting(
        "EMBEDDINGS_ENABLED",
        default=False,
    )

    model = (
        read_setting(
            "OPENAI_EMBEDDING_MODEL",
            DEFAULT_OPENAI_EMBEDDING_MODEL,
        )
        or DEFAULT_OPENAI_EMBEDDING_MODEL
    )

    sdk_available = (
        _sdk_available()
    )

    configured = bool(
        api_key
    )

    ready = (
        configured
        and enabled
        and sdk_available
    )

    if ready:
        mode = "Semantic embeddings"

        message = (
            "Embedding retrieval is enabled and ready."
        )

    elif not enabled:
        mode = "Retrieval fallback"

        message = (
            "Embedding retrieval is intentionally disabled. "
            "Lexical retrieval remains available."
        )

    elif not configured:
        mode = "Retrieval fallback"

        message = (
            "Embedding retrieval is enabled, but no "
            "OPENAI_API_KEY is configured."
        )

    else:
        mode = "Retrieval fallback"

        message = (
            "The OpenAI Python package is unavailable."
        )

    return asdict(
        EmbeddingProviderStatus(
            provider="OpenAI",
            configured=configured,
            enabled=enabled,
            sdk_available=sdk_available,
            ready=ready,
            model=model,
            mode=mode,
            message=message,
        )
    )
