"""CONTROLLED LIVE OpenAI embedding connectivity test.

IMPORTANT:
- This file makes a REAL, cost-bearing API request when embeddings are enabled.
- Do not include it in routine regression testing.
- It uses synthetic text only.
- It never prints the API key or embedding-vector values.
"""

from __future__ import annotations

import math

from govba.core.settings import read_setting
from govba.rag import (
    OpenAIEmbeddingProvider,
    OpenAIEmbeddingProviderError,
    get_embedding_provider_status,
)


TEST_TEXT = (
    "GovBA controlled embedding connectivity test."
)


def main() -> int:
    status = get_embedding_provider_status()

    print("GovBA controlled LIVE embedding test")
    print("-" * 48)

    for key in (
        "provider",
        "configured",
        "enabled",
        "sdk_available",
        "ready",
        "model",
        "mode",
    ):
        print(
            f"{key}: {status[key]}"
        )

    if not status["ready"]:
        print()
        print(
            "LIVE request blocked safely: "
            "embedding provider is not ready."
        )
        print(
            "No external embedding API request was made."
        )
        return 2

    api_key = read_setting(
        "OPENAI_API_KEY"
    )

    if not api_key:
        print(
            "LIVE request blocked safely: "
            "API key unavailable."
        )
        return 2

    provider = OpenAIEmbeddingProvider(
        api_key=api_key,
        model=str(
            status["model"]
        ),
        batch_size=1,
    )

    try:
        vector = provider.embed_query(
            TEST_TEXT
        )

    except OpenAIEmbeddingProviderError as exc:
        print()
        print(
            "LIVE embedding request failed safely."
        )
        print(
            "Error type:",
            type(exc).__name__,
        )
        print(
            "Safe message:",
            str(exc),
        )

        return 1

    if not vector:
        print(
            "LIVE embedding test failed: "
            "empty vector."
        )
        return 1

    vector_is_finite = all(
        isinstance(value, float)
        and math.isfinite(value)
        for value in vector
    )

    if not vector_is_finite:
        print(
            "LIVE embedding test failed: "
            "vector contained invalid values."
        )
        return 1

    vector_norm = math.sqrt(
        sum(
            value * value
            for value in vector
        )
    )

    print()
    print(
        "CONTROLLED LIVE EMBEDDING TEST PASSED."
    )
    print(
        "model:",
        status["model"],
    )
    print(
        "dimensions:",
        len(vector),
    )
    print(
        "finite_vector:",
        vector_is_finite,
    )
    print(
        "vector_norm:",
        round(
            vector_norm,
            6,
        ),
    )
    print(
        "vector_values_printed: False"
    )
    print(
        "synthetic_input_only: True"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
