"""Small deterministic capability router for GovBA-GAR."""

from __future__ import annotations

from govba.router.contract import (
    Capability,
    RouteReason,
    RoutingDecision,
    RoutingRequest,
)


REQUIREMENTS_SIGNALS = (
    "requirement",
    "requirements",
    "brd",
    "business requirements",
    "متطلب",
    "متطلبات",
)

CHANGE_SIGNALS = (
    "what changed",
    "policy change",
    "circular change",
    "compare policies",
    "compare versions",
    "superseded",
    "ما الذي تغير",
    "ما الذي تغيّر",
    "تغيير السياسة",
    "مقارنة الإصدارات",
    "مقارنة الاصدارات",
)

CORRESPONDENCE_SIGNALS = (
    "email",
    "letter",
    "correspondence",
    "memorandum",
    "memo",
    "reply to",
    "بريد",
    "رسالة",
    "مراسلة",
    "كتاب رسمي",
    "مذكرة",
)

WEB_SIGNALS = (
    "latest",
    "current official",
    "official website",
    "search online",
    "search the web",
    "أحدث",
    "احدث",
    "الموقع الرسمي",
    "ابحث على الإنترنت",
    "ابحث على الانترنت",
)

RAG_SIGNALS = (
    "policy",
    "regulation",
    "law",
    "procedure",
    "circular",
    "document",
    "according to",
    "سياسة",
    "قانون",
    "لائحة",
    "إجراء",
    "اجراء",
    "تعميم",
    "وثيقة",
)


def _contains(
    text: str,
    signals: tuple[str, ...],
) -> bool:
    return any(
        signal in text
        for signal in signals
    )


def route_capability(
    request: RoutingRequest,
) -> RoutingDecision:
    if not isinstance(
        request,
        RoutingRequest,
    ):
        raise TypeError(
            "request must be a RoutingRequest."
        )

    if request.capability_hint is not None:
        return RoutingDecision(
            request_id=request.request_id,
            capability=request.capability_hint,
            reason=RouteReason.EXPLICIT_HINT,
            confidence=1.0,
        )

    text = request.text.casefold()

    checks = (
        (
            REQUIREMENTS_SIGNALS,
            Capability.REQUIREMENTS,
            RouteReason.REQUIREMENTS_SIGNAL,
            0.95,
        ),
        (
            CHANGE_SIGNALS,
            Capability.CHANGE,
            RouteReason.CHANGE_SIGNAL,
            0.95,
        ),
        (
            CORRESPONDENCE_SIGNALS,
            Capability.CORRESPONDENCE,
            RouteReason.CORRESPONDENCE_SIGNAL,
            0.90,
        ),
        (
            WEB_SIGNALS,
            Capability.WEB,
            RouteReason.WEB_SIGNAL,
            0.90,
        ),
        (
            RAG_SIGNALS,
            Capability.RAG,
            RouteReason.RAG_SIGNAL,
            0.85,
        ),
    )

    for (
        signals,
        capability,
        reason,
        confidence,
    ) in checks:
        if _contains(
            text,
            signals,
        ):
            return RoutingDecision(
                request_id=request.request_id,
                capability=capability,
                reason=reason,
                confidence=confidence,
            )

    return RoutingDecision(
        request_id=request.request_id,
        capability=Capability.DIRECT,
        reason=RouteReason.DEFAULT_DIRECT,
        confidence=0.60,
    )
