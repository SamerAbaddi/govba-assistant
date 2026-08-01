import json

from ai_provider import get_ai_provider_status, request_ai_text

status = get_ai_provider_status()

print("Provider readiness:")
print(json.dumps({
    "provider": status["provider"],
    "configured": status["configured"],
    "enabled": status["enabled"],
    "sdk_available": status["sdk_available"],
    "ready": status["ready"],
    "model": status["model"],
    "mode": status["mode"],
}, indent=2))

if not status["ready"]:
    raise SystemExit(
        "AI provider is not ready. Check AI_ENABLED and local secrets."
    )

result = request_ai_text(
    (
        "Return exactly one short sentence confirming that the secure "
        "API connection works. Do not add anything else."
    ),
    "GovBA Assistant secure connection test.",
    max_output_tokens=300,
    reasoning_effort="minimal",
    text_verbosity="low",
)

safe_result = {
    "success": result["success"],
    "fallback_required": result["fallback_required"],
    "provider": result["provider"],
    "model": result["model"],
    "mode": result["mode"],
    "response_status": result["response_status"],
    "incomplete_reason": result["incomplete_reason"],
    "usage": result["usage"],
    "text": result["text"],
    "error": result["error"],
}

print("\nConnection result:")
print(json.dumps(safe_result, indent=2, ensure_ascii=False))

if not result["success"]:
    raise SystemExit(1)

print("\nControlled AI connection test passed.")