"""Deterministic prompt-injection detection for GovBA-GAR."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import Enum

from govba.governance.security import (
    SecurityAssessment,
    SecurityDecision,
    SecurityFinding,
    SecurityIssueCode,
    SecuritySeverity,
    SecuritySurface,
    assess_security,
)


PROMPT_INJECTION_VERSION = (
    "govba-prompt-injection-v1"
)


class PromptInjectionSignalType(
    str,
    Enum,
):
    """Auditable prompt-injection signal taxonomy."""

    INSTRUCTION_OVERRIDE = (
        "instruction_override"
    )

    SYSTEM_PROMPT_DISCLOSURE = (
        "system_prompt_disclosure"
    )

    POLICY_BYPASS = (
        "policy_bypass"
    )

    ROLE_OVERRIDE = (
        "role_override"
    )

    TOOL_MANIPULATION = (
        "tool_manipulation"
    )

    DATA_EXFILTRATION = (
        "data_exfiltration"
    )


@dataclass(frozen=True)
class _PatternDefinition:
    signal_type: (
        PromptInjectionSignalType
    )

    pattern: re.Pattern[str]

    severity: SecuritySeverity


_PATTERN_DEFINITIONS = (
    _PatternDefinition(
        signal_type=(
            PromptInjectionSignalType
            .INSTRUCTION_OVERRIDE
        ),
        pattern=re.compile(
            r"\bignore\s+"
            r"(?:all\s+|any\s+|the\s+)?"
            r"(?:previous|prior|above)"
            r"\s+instructions?\b",
            flags=re.IGNORECASE,
        ),
        severity=SecuritySeverity.HIGH,
    ),
    _PatternDefinition(
        signal_type=(
            PromptInjectionSignalType
            .INSTRUCTION_OVERRIDE
        ),
        pattern=re.compile(
            r"(?:تجاهل|اهمل)"
            r"\s+(?:كل\s+)?"
            r"(?:التعليمات|الاوامر|الأوامر)"
            r"\s+(?:السابقة|المذكورة|اعلاه|أعلاه)",
            flags=re.IGNORECASE,
        ),
        severity=SecuritySeverity.HIGH,
    ),
    _PatternDefinition(
        signal_type=(
            PromptInjectionSignalType
            .SYSTEM_PROMPT_DISCLOSURE
        ),
        pattern=re.compile(
            r"\b(?:show|reveal|print|display|expose)"
            r"\s+(?:me\s+)?"
            r"(?:the\s+)?"
            r"(?:system\s+prompt|system\s+instructions"
            r"|hidden\s+instructions)\b",
            flags=re.IGNORECASE,
        ),
        severity=SecuritySeverity.CRITICAL,
    ),
    _PatternDefinition(
        signal_type=(
            PromptInjectionSignalType
            .SYSTEM_PROMPT_DISCLOSURE
        ),
        pattern=re.compile(
            r"(?:اكشف|اعرض|اظهر|أظهر)"
            r".{0,20}"
            r"(?:تعليمات\s+النظام"
            r"|موجه\s+النظام"
            r"|التعليمات\s+المخفية)",
            flags=re.IGNORECASE,
        ),
        severity=SecuritySeverity.CRITICAL,
    ),
    _PatternDefinition(
        signal_type=(
            PromptInjectionSignalType
            .POLICY_BYPASS
        ),
        pattern=re.compile(
            r"\b(?:bypass|override|circumvent|disable)"
            r"\s+(?:the\s+)?"
            r"(?:safety|security|policy|policies"
            r"|guardrails?|restrictions?)\b",
            flags=re.IGNORECASE,
        ),
        severity=SecuritySeverity.CRITICAL,
    ),
    _PatternDefinition(
        signal_type=(
            PromptInjectionSignalType
            .POLICY_BYPASS
        ),
        pattern=re.compile(
            r"(?:تجاوز|عطل|عطّل)"
            r".{0,20}"
            r"(?:سياسات|قيود|حماية|الحماية|الامان|الأمان)",
            flags=re.IGNORECASE,
        ),
        severity=SecuritySeverity.CRITICAL,
    ),
    _PatternDefinition(
        signal_type=(
            PromptInjectionSignalType
            .ROLE_OVERRIDE
        ),
        pattern=re.compile(
            r"\b(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be)"
            r".{0,40}"
            r"(?:system|administrator|developer|root)\b",
            flags=re.IGNORECASE,
        ),
        severity=SecuritySeverity.HIGH,
    ),
    _PatternDefinition(
        signal_type=(
            PromptInjectionSignalType
            .TOOL_MANIPULATION
        ),
        pattern=re.compile(
            r"\b(?:call|invoke|execute|run)"
            r"\s+(?:the\s+)?"
            r"(?:tool|function|command)"
            r".{0,40}"
            r"(?:without|ignore|bypass)\b",
            flags=re.IGNORECASE,
        ),
        severity=SecuritySeverity.CRITICAL,
    ),
    _PatternDefinition(
        signal_type=(
            PromptInjectionSignalType
            .DATA_EXFILTRATION
        ),
        pattern=re.compile(
            r"\b(?:send|upload|transmit|exfiltrate)"
            r".{0,40}"
            r"(?:secret|credential|password|api\s*key"
            r"|private\s+data)\b",
            flags=re.IGNORECASE,
        ),
        severity=SecuritySeverity.CRITICAL,
    ),
)


def _required_text(
    value: str,
    field_name: str,
) -> str:
    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            f"{field_name} must be a string."
        )

    value = value.strip()

    if not value:
        raise ValueError(
            f"{field_name} must not be blank."
        )

    return value


def _sha256_text(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode(
            "utf-8"
        )
    ).hexdigest()


@dataclass(frozen=True)
class PromptInjectionSignal:
    """Injection metadata without storing matched text."""

    signal_type: (
        PromptInjectionSignalType
    )

    severity: SecuritySeverity

    start: int
    end: int

    def __post_init__(self) -> None:
        if not isinstance(
            self.signal_type,
            PromptInjectionSignalType,
        ):
            raise TypeError(
                "signal_type must be a "
                "PromptInjectionSignalType."
            )

        if not isinstance(
            self.severity,
            SecuritySeverity,
        ):
            raise TypeError(
                "severity must be a "
                "SecuritySeverity."
            )

        if (
            isinstance(self.start, bool)
            or not isinstance(
                self.start,
                int,
            )
            or self.start < 0
        ):
            raise ValueError(
                "start must be a "
                "non-negative integer."
            )

        if (
            isinstance(self.end, bool)
            or not isinstance(
                self.end,
                int,
            )
            or self.end <= self.start
        ):
            raise ValueError(
                "end must be greater than start."
            )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "signal_type": (
                self.signal_type.value
            ),
            "severity": (
                self.severity.value
            ),
            "start": (
                self.start
            ),
            "end": (
                self.end
            ),
        }


@dataclass(frozen=True)
class PromptInjectionAssessment:
    """Privacy-safe prompt-injection assessment."""

    trace_id: str

    input_sha256: str

    signals: tuple[
        PromptInjectionSignal,
        ...
    ]

    security_assessment: (
        SecurityAssessment
    )

    version: str = (
        PROMPT_INJECTION_VERSION
    )

    def __post_init__(self) -> None:
        trace_id = _required_text(
            self.trace_id,
            "trace_id",
        )

        input_hash = _required_text(
            self.input_sha256,
            "input_sha256",
        ).lower()

        if (
            len(input_hash) != 64
            or any(
                character
                not in "0123456789abcdef"
                for character in input_hash
            )
        ):
            raise ValueError(
                "input_sha256 must be "
                "a SHA-256 digest."
            )

        try:
            signals = tuple(
                self.signals
            )
        except TypeError as exc:
            raise TypeError(
                "signals must be iterable."
            ) from exc

        for signal in signals:
            if not isinstance(
                signal,
                PromptInjectionSignal,
            ):
                raise TypeError(
                    "signals must contain only "
                    "PromptInjectionSignal values."
                )

        if not isinstance(
            self.security_assessment,
            SecurityAssessment,
        ):
            raise TypeError(
                "security_assessment must be "
                "a SecurityAssessment."
            )

        if (
            self.security_assessment.trace_id
            != trace_id
        ):
            raise ValueError(
                "Security assessment trace ID "
                "must match."
            )

        object.__setattr__(
            self,
            "trace_id",
            trace_id,
        )

        object.__setattr__(
            self,
            "input_sha256",
            input_hash,
        )

        object.__setattr__(
            self,
            "signals",
            signals,
        )

    @property
    def injection_detected(
        self,
    ) -> bool:
        return bool(
            self.signals
        )

    @property
    def signal_count(
        self,
    ) -> int:
        return len(
            self.signals
        )

    @property
    def should_block(
        self,
    ) -> bool:
        return (
            self.security_assessment
            .should_block
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        """Serialize without storing inspected content."""

        return {
            "version": (
                self.version
            ),
            "trace_id": (
                self.trace_id
            ),
            "input_sha256": (
                self.input_sha256
            ),
            "injection_detected": (
                self.injection_detected
            ),
            "signal_count": (
                self.signal_count
            ),
            "signals": [
                signal.to_dict()
                for signal
                in self.signals
            ],
            "security_assessment": (
                self.security_assessment
                .to_dict()
            ),
        }


def detect_prompt_injection(
    text: str,
) -> tuple[
    PromptInjectionSignal,
    ...
]:
    """Detect deterministic prompt-injection signals."""

    value = _required_text(
        text,
        "text",
    )

    signals = []

    for definition in (
        _PATTERN_DEFINITIONS
    ):
        for match in (
            definition.pattern.finditer(
                value
            )
        ):
            signals.append(
                PromptInjectionSignal(
                    signal_type=(
                        definition.signal_type
                    ),
                    severity=(
                        definition.severity
                    ),
                    start=match.start(),
                    end=match.end(),
                )
            )

    return tuple(
        sorted(
            signals,
            key=lambda signal: (
                signal.start,
                signal.end,
                signal.signal_type.value,
            ),
        )
    )


def assess_prompt_injection(
    text: str,
    *,
    trace_id: str,
    surface: SecuritySurface,
) -> PromptInjectionAssessment:
    """Assess input and fail closed when injection is detected."""

    value = _required_text(
        text,
        "text",
    )

    trace = _required_text(
        trace_id,
        "trace_id",
    )

    if not isinstance(
        surface,
        SecuritySurface,
    ):
        raise TypeError(
            "surface must be a "
            "SecuritySurface."
        )

    input_hash = _sha256_text(
        value
    )

    signals = detect_prompt_injection(
        value
    )

    if signals:
        severity = (
            SecuritySeverity.CRITICAL
            if any(
                signal.severity
                is SecuritySeverity.CRITICAL
                for signal in signals
            )
            else SecuritySeverity.HIGH
        )

        finding = SecurityFinding(
            code=(
                SecurityIssueCode
                .PROMPT_INJECTION_DETECTED
            ),
            surface=surface,
            severity=severity,
            decision=(
                SecurityDecision.BLOCK
            ),
            occurrence_count=len(
                signals
            ),
            reference_id=(
                f"{input_hash[:16]}:"
                "prompt-injection"
            ),
        )

        findings = (
            finding,
        )

    else:
        findings = ()

    security = assess_security(
        trace_id=trace,
        findings=findings,
    )

    return PromptInjectionAssessment(
        trace_id=trace,
        input_sha256=input_hash,
        signals=signals,
        security_assessment=security,
    )
