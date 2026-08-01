"""
AI-enhanced employee email summarization for GovBA Assistant.

The engine uses OpenAI only when explicitly requested and available.
Otherwise, it returns the existing rule-based summary.
"""

from __future__ import annotations

from typing import Any

from ai_provider import (
    get_ai_provider_status,
    request_ai_json,
)
from email_summary_engine import summarize_employee_email


EMAIL_SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sender": {"type": "string"},
        "recipient": {"type": "string"},
        "subject": {"type": "string"},
        "priority": {
            "type": "string",
            "enum": [
                "High",
                "Medium",
                "Normal",
                "Requires confirmation",
            ],
        },
        "summary_bullets": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 5,
        },
        "action_items": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 6,
        },
        "deadlines": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 6,
        },
        "decisions": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 6,
        },
    },
    "required": [
        "sender",
        "recipient",
        "subject",
        "priority",
        "summary_bullets",
        "action_items",
        "deadlines",
        "decisions",
    ],
    "additionalProperties": False,
}


EMAIL_SUMMARY_INSTRUCTIONS = """
You are the email-summary component of GovBA Assistant.

Analyze only the supplied email text. Do not browse, contact anyone,
send messages, or perform actions.

Requirements:
- Preserve names, dates, amounts, decisions, and obligations exactly.
- Do not invent missing facts.
- Use "Requires confirmation" when sender, recipient, or subject is absent.
- Classify priority as High only when urgency is explicit.
- Produce 2 to 5 short, clear summary bullets when possible.
- Extract only genuine requested actions.
- Extract explicit deadlines; otherwise return
  "No clear deadline detected."
- Extract explicit decisions; otherwise return
  "No clear decision detected."
- Keep every item concise and suitable for an employee dashboard.
- Return only the structured output requested by the schema.
""".strip()


def _ensure_string(value: Any, default: str) -> str:
    cleaned = str(value or "").strip()
    return cleaned or default


def _ensure_list(
    value: Any,
    default: str,
    maximum: int,
) -> list[str]:
    if not isinstance(value, list):
        return [default]

    cleaned_items = []

    for item in value:
        cleaned = str(item or "").strip()

        if cleaned and cleaned not in cleaned_items:
            cleaned_items.append(cleaned)

        if len(cleaned_items) == maximum:
            break

    return cleaned_items or [default]


def _build_rule_based_result(
    email_text: str,
    fallback_reason: str | None = None,
    ai_attempted: bool = False,
) -> dict[str, Any]:
    result = summarize_employee_email(email_text)

    result.update(
        {
            "processing_mode": "Rule-based fallback",
            "ai_attempted": ai_attempted,
            "fallback_used": True,
            "fallback_reason": fallback_reason,
            "ai_metadata": None,
        }
    )

    return result


def summarize_employee_email_safely(
    email_text: str,
    *,
    use_ai: bool = True,
) -> dict[str, Any]:
    """
    Summarize pasted or uploaded email text.

    This function never connects to Gmail, Outlook, or a Ministry
    mailbox. It only processes the supplied text.
    """

    if not email_text or not email_text.strip():
        raise ValueError(
            "Email text is required before creating a summary."
        )

    if not use_ai:
        return _build_rule_based_result(
            email_text,
            fallback_reason=(
                "Rule-based processing was selected by the user."
            ),
            ai_attempted=False,
        )

    provider_status = get_ai_provider_status()

    if not provider_status["ready"]:
        return _build_rule_based_result(
            email_text,
            fallback_reason=provider_status["message"],
            ai_attempted=False,
        )

    ai_result = request_ai_json(
        EMAIL_SUMMARY_INSTRUCTIONS,
        email_text,
        EMAIL_SUMMARY_SCHEMA,
        schema_name="govba_email_summary",
        max_output_tokens=1_200,
    )

    if (
        not ai_result["success"]
        or not isinstance(ai_result.get("data"), dict)
    ):
        return _build_rule_based_result(
            email_text,
            fallback_reason=(
                ai_result.get("error")
                or "The AI response was unavailable."
            ),
            ai_attempted=True,
        )

    data = ai_result["data"]

    return {
        "sender": _ensure_string(
            data.get("sender"),
            "Requires confirmation",
        ),
        "recipient": _ensure_string(
            data.get("recipient"),
            "Requires confirmation",
        ),
        "subject": _ensure_string(
            data.get("subject"),
            "Requires confirmation",
        ),
        "priority": _ensure_string(
            data.get("priority"),
            "Requires confirmation",
        ),
        "summary_bullets": _ensure_list(
            data.get("summary_bullets"),
            "Requires confirmation",
            5,
        ),
        "action_items": _ensure_list(
            data.get("action_items"),
            "No clear action item detected.",
            6,
        ),
        "deadlines": _ensure_list(
            data.get("deadlines"),
            "No clear deadline detected.",
            6,
        ),
        "decisions": _ensure_list(
            data.get("decisions"),
            "No clear decision detected.",
            6,
        ),
        "summary_mode": "OpenAI structured summary",
        "human_review_notes": [
            "Review the original email before acting.",
            "The AI may omit context or interpret wording incorrectly.",
            "No Gmail, Outlook, or Ministry email connection is used.",
        ],
        "processing_mode": "AI-enhanced",
        "ai_attempted": True,
        "fallback_used": False,
        "fallback_reason": None,
        "ai_metadata": {
            "provider": ai_result.get("provider"),
            "model": ai_result.get("model"),
            "response_status": ai_result.get("response_status"),
            "usage": ai_result.get("usage"),
        },
    }