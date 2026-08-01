"""
Safe provider test.

This file forces AI_ENABLED=false, so no paid API call can occur.
"""

import os

os.environ["AI_ENABLED"] = "false"

from ai_provider import get_ai_provider_status, request_ai_text


status = get_ai_provider_status()

assert status["ready"] is False
assert status["mode"] == "Rule-based fallback"

result = request_ai_text(
    "Return a one-sentence test response.",
    "GovBA Assistant provider test.",
    max_output_tokens=100,
)

assert result["success"] is False
assert result["fallback_required"] is True

print("GovBA AI provider safety test passed.")
print(f"Provider: {status['provider']}")
print(f"Model setting: {status['model']}")
print(f"Mode: {status['mode']}")
print("No external API request was made.")