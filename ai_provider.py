"""
Central AI provider for GovBA Assistant Phase 2.

AI is disabled by default. Existing rule-based engines remain the fallback.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any

from govba.telemetry import (
    TelemetryEvent,
    TelemetryTimer,
    extract_usage,
    record_event,
)


DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_MAX_OUTPUT_TOKENS = 900
DEFAULT_REASONING_EFFORT = "minimal"
DEFAULT_TEXT_VERBOSITY = "low"
MAX_AI_INPUT_CHARACTERS = 50_000


@dataclass(frozen=True)
class AIProviderStatus:
    provider: str
    configured: bool
    enabled: bool
    sdk_available: bool
    ready: bool
    model: str
    mode: str
    message: str


def _read_streamlit_secret(name: str) -> str | None:
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


def _read_setting(
    name: str,
    default: str | None = None,
) -> str | None:
    environment_value = os.getenv(name)

    if environment_value is not None:
        cleaned = environment_value.strip()

        if cleaned:
            return cleaned

    streamlit_value = _read_streamlit_secret(name)

    if streamlit_value is not None:
        return streamlit_value

    return default


def _read_boolean_setting(
    name: str,
    default: bool = False,
) -> bool:
    raw_value = _read_setting(name)

    if raw_value is None:
        return default

    return raw_value.lower() in {
        "1",
        "true",
        "yes",
        "on",
        "enabled",
    }


def _sdk_available() -> bool:
    try:
        import openai  # noqa: F401
    except ImportError:
        return False

    return True


def get_ai_provider_status() -> dict[str, Any]:
    api_key = _read_setting("OPENAI_API_KEY")
    enabled = _read_boolean_setting(
        "AI_ENABLED",
        default=False,
    )
    sdk_available = _sdk_available()
    model = _read_setting(
        "OPENAI_MODEL",
        DEFAULT_MODEL,
    ) or DEFAULT_MODEL

    configured = bool(api_key)
    ready = configured and enabled and sdk_available

    if ready:
        mode = "AI mode"
        message = (
            "AI mode is enabled and ready."
        )
    elif not enabled:
        mode = "Rule-based fallback"
        message = (
            "AI mode is intentionally disabled. Existing rule-based "
            "engines should continue to be used."
        )
    elif not configured:
        mode = "Rule-based fallback"
        message = (
            "AI mode is enabled, but no OPENAI_API_KEY is configured."
        )
    else:
        mode = "Rule-based fallback"
        message = (
            "The OpenAI Python package is unavailable."
        )

    return asdict(
        AIProviderStatus(
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


def _validate_request(
    instructions: str,
    user_input: str,
    max_output_tokens: int,
    reasoning_effort: str,
    text_verbosity: str,
) -> None:
    if not instructions or not instructions.strip():
        raise ValueError("AI instructions are required.")

    if not user_input or not user_input.strip():
        raise ValueError("AI input text is required.")

    if len(user_input) > MAX_AI_INPUT_CHARACTERS:
        raise ValueError(
            "AI input exceeds the prototype limit of "
            f"{MAX_AI_INPUT_CHARACTERS:,} characters."
        )

    if not 100 <= max_output_tokens <= 4_000:
        raise ValueError(
            "max_output_tokens must be between 100 and 4,000."
        )

    if reasoning_effort not in {
        "minimal",
        "low",
        "medium",
        "high",
    }:
        raise ValueError(
            "reasoning_effort must be minimal, low, medium, or high."
        )

    if text_verbosity not in {
        "low",
        "medium",
        "high",
    }:
        raise ValueError(
            "text_verbosity must be low, medium, or high."
        )


def _extract_response_text(response: Any) -> str:
    output_text = str(
        getattr(response, "output_text", "") or ""
    ).strip()

    if output_text:
        return output_text

    collected: list[str] = []

    for output_item in getattr(response, "output", []) or []:
        for content_item in getattr(output_item, "content", []) or []:
            content_type = getattr(content_item, "type", "")

            if content_type == "output_text":
                text = str(
                    getattr(content_item, "text", "") or ""
                ).strip()

                if text:
                    collected.append(text)

            elif content_type == "refusal":
                refusal = str(
                    getattr(content_item, "refusal", "") or ""
                ).strip()

                if refusal:
                    collected.append(refusal)

    return "\n".join(collected).strip()


def _usage_to_dict(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)

    if usage is None:
        return None

    output_details = getattr(
        usage,
        "output_tokens_details",
        None,
    )

    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "reasoning_tokens": (
            getattr(output_details, "reasoning_tokens", None)
            if output_details is not None
            else None
        ),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _record_ai_telemetry_safely(
    *,
    timer: TelemetryTimer,
    status: dict[str, Any],
    result: dict[str, Any],
    response: Any = None,
    error_type: str = "",
) -> None:
    """Record provider telemetry without affecting AI behavior."""

    try:
        latency_ms = timer.stop()
        usage = extract_usage(response)

        event = TelemetryEvent(
            task_type="ai_provider_request",
            language="unknown",
            mode=str(
                result.get("mode")
                or status.get("mode")
                or "unknown"
            ),
            model=str(status.get("model") or ""),
            latency_ms=latency_ms,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            cached_input_tokens=usage.cached_input_tokens,
            reasoning_output_tokens=usage.reasoning_output_tokens,
            success=(
                bool(result.get("success"))
                and not bool(error_type)
            ),
            fallback_used=bool(
                result.get("fallback_required")
            ),
            error_type=error_type,
        )

        record_event(event)

    except Exception:
        # Observability must never interrupt GovBA.
        return


def request_ai_response(
    instructions: str,
    user_input: str,
    *,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    text_verbosity: str = DEFAULT_TEXT_VERBOSITY,
    json_schema: dict[str, Any] | None = None,
    schema_name: str = "govba_response",
) -> dict[str, Any]:
    _validate_request(
        instructions,
        user_input,
        max_output_tokens,
        reasoning_effort,
        text_verbosity,
    )

    status = get_ai_provider_status()

    result = {
        "success": False,
        "fallback_required": True,
        "provider": status["provider"],
        "model": status["model"],
        "mode": status["mode"],
        "response_status": None,
        "response_id": None,
        "incomplete_reason": None,
        "usage": None,
        "text": "",
        "data": None,
        "error": None,
    }

    timer = TelemetryTimer().start()

    if not status["ready"]:
        result["error"] = status["message"]

        _record_ai_telemetry_safely(
            timer=timer,
            status=status,
            result=result,
            error_type="ProviderNotReady",
        )

        return result

    api_key = _read_setting("OPENAI_API_KEY")

    response: Any = None
    telemetry_error_type = ""

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            timeout=45.0,
            max_retries=1,
        )

        text_settings: dict[str, Any] = {
            "verbosity": text_verbosity,
        }

        if json_schema is not None:
            text_settings["format"] = {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": json_schema,
            }

        response = client.responses.create(
            model=status["model"],
            instructions=instructions.strip(),
            input=user_input.strip(),
            max_output_tokens=max_output_tokens,
            reasoning={"effort": reasoning_effort},
            text=text_settings,
            store=False,
        )

        response_status = getattr(
            response,
            "status",
            None,
        )

        incomplete_details = getattr(
            response,
            "incomplete_details",
            None,
        )

        incomplete_reason = (
            getattr(
                incomplete_details,
                "reason",
                None,
            )
            if incomplete_details is not None
            else None
        )

        result.update(
            {
                "response_status": response_status,
                "response_id": getattr(
                    response,
                    "id",
                    None,
                ),
                "incomplete_reason": incomplete_reason,
                "usage": _usage_to_dict(response),
            }
        )

        output_text = _extract_response_text(response)

        if not output_text:
            if response_status == "incomplete":
                result["error"] = (
                    "The AI response was incomplete"
                    + (
                        f" because of: {incomplete_reason}."
                        if incomplete_reason
                        else "."
                    )
                    + " Increase max_output_tokens or reduce reasoning effort."
                )
                telemetry_error_type = "IncompleteResponse"

            else:
                result["error"] = (
                    "The AI provider returned no visible text. "
                    f"Response status: {response_status or 'unknown'}."
                )
                telemetry_error_type = "EmptyResponse"

            return result

        parsed_data = None

        if json_schema is not None:
            try:
                parsed_data = json.loads(output_text)
            except json.JSONDecodeError:
                telemetry_error_type = "JSONDecodeError"
                result["error"] = (
                    "The AI provider returned invalid structured JSON."
                )
                return result

        result.update(
            {
                "success": True,
                "fallback_required": False,
                "mode": "AI mode",
                "text": output_text,
                "error": None,
            }
        )

        if json_schema is not None:
            result["data"] = parsed_data

        return result

    except Exception as error:
        telemetry_error_type = type(error).__name__

        result["error"] = (
            "The AI request could not be completed. "
            f"{type(error).__name__}: {error}"
        )

        return result

    finally:
        _record_ai_telemetry_safely(
            timer=timer,
            status=status,
            result=result,
            response=response,
            error_type=telemetry_error_type,
        )


def request_ai_text(
    instructions: str,
    user_input: str,
    *,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    text_verbosity: str = DEFAULT_TEXT_VERBOSITY,
) -> dict[str, Any]:
    return request_ai_response(
        instructions,
        user_input,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        text_verbosity=text_verbosity,
    )


def request_ai_json(
    instructions: str,
    user_input: str,
    json_schema: dict[str, Any],
    *,
    schema_name: str = "govba_response",
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    text_verbosity: str = DEFAULT_TEXT_VERBOSITY,
) -> dict[str, Any]:
    return request_ai_response(
        instructions,
        user_input,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        text_verbosity=text_verbosity,
        json_schema=json_schema,
        schema_name=schema_name,
    )