"""Minimal routing contract for GovBA-GAR."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum

from govba.rag.models import SourceLanguage


ROUTER_CONTRACT_VERSION = "govba-router-contract-v1"


class Capability(str, Enum):
    DIRECT = "direct"
    RAG = "rag"
    WEB = "web"
    CHANGE = "change"
    CORRESPONDENCE = "correspondence"
    REQUIREMENTS = "requirements"


class RouteReason(str, Enum):
    EXPLICIT_HINT = "explicit_hint"
    REQUIREMENTS_SIGNAL = "requirements_signal"
    CHANGE_SIGNAL = "change_signal"
    CORRESPONDENCE_SIGNAL = "correspondence_signal"
    WEB_SIGNAL = "web_signal"
    RAG_SIGNAL = "rag_signal"
    DEFAULT_DIRECT = "default_direct"


def _text(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")

    value = " ".join(value.split()).strip()

    if not value:
        raise ValueError(f"{name} must not be blank.")

    return value


def _sha256(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class RoutingRequest:
    text: str
    language: SourceLanguage
    trace_id: str
    capability_hint: Capability | None = None
    version: str = ROUTER_CONTRACT_VERSION
    text_sha256: str = field(init=False)
    request_id: str = field(init=False)

    def __post_init__(self) -> None:
        text = _text(self.text, "text")
        trace_id = _text(
            self.trace_id,
            "trace_id",
        )

        if not isinstance(
            self.language,
            SourceLanguage,
        ):
            raise TypeError(
                "language must be a SourceLanguage."
            )

        if (
            self.capability_hint is not None
            and not isinstance(
                self.capability_hint,
                Capability,
            )
        ):
            raise TypeError(
                "capability_hint must be "
                "a Capability or None."
            )

        digest = _sha256(text)

        identity = "\x1f".join(
            (
                self.version,
                digest,
                self.language.value,
                trace_id,
                (
                    self.capability_hint.value
                    if self.capability_hint
                    else ""
                ),
            )
        )

        object.__setattr__(
            self,
            "text",
            text,
        )
        object.__setattr__(
            self,
            "trace_id",
            trace_id,
        )
        object.__setattr__(
            self,
            "text_sha256",
            digest,
        )
        object.__setattr__(
            self,
            "request_id",
            _sha256(identity),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "language": self.language.value,
            "text_sha256": self.text_sha256,
            "capability_hint": (
                self.capability_hint.value
                if self.capability_hint
                else None
            ),
        }


@dataclass(frozen=True)
class RoutingDecision:
    request_id: str
    capability: Capability
    reason: RouteReason
    confidence: float
    version: str = ROUTER_CONTRACT_VERSION
    route_id: str = field(init=False)

    def __post_init__(self) -> None:
        request_id = _text(
            self.request_id,
            "request_id",
        )

        if not isinstance(
            self.capability,
            Capability,
        ):
            raise TypeError(
                "capability must be a Capability."
            )

        if not isinstance(
            self.reason,
            RouteReason,
        ):
            raise TypeError(
                "reason must be a RouteReason."
            )

        if (
            isinstance(self.confidence, bool)
            or not isinstance(
                self.confidence,
                (int, float),
            )
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError(
                "confidence must be between 0 and 1."
            )

        identity = "\x1f".join(
            (
                self.version,
                request_id,
                self.capability.value,
                self.reason.value,
                f"{float(self.confidence):.6f}",
            )
        )

        object.__setattr__(
            self,
            "request_id",
            request_id,
        )
        object.__setattr__(
            self,
            "confidence",
            float(self.confidence),
        )
        object.__setattr__(
            self,
            "route_id",
            _sha256(identity),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "route_id": self.route_id,
            "request_id": self.request_id,
            "capability": self.capability.value,
            "reason": self.reason.value,
            "confidence": self.confidence,
        }
