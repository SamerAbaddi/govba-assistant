"""Government-source allowlisting and trust validation for GovBA-GAR."""

from __future__ import annotations

import hashlib
import ipaddress
import re
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlsplit

from govba.governance.security import (
    SecurityAssessment,
    SecurityDecision,
    SecurityFinding,
    SecurityIssueCode,
    SecuritySeverity,
    SecuritySurface,
    assess_security,
)
from govba.rag.models import (
    AuthoritativeSource,
)


SOURCE_ALLOWLIST_VERSION = (
    "govba-source-allowlist-v1"
)


class SourceAllowlistReasonCode(
    str,
    Enum,
):
    """Stable source-trust reason taxonomy."""

    ALLOWED = "allowed"

    MISSING_OFFICIAL_URL = (
        "missing_official_url"
    )

    INVALID_URL = (
        "invalid_url"
    )

    NON_HTTPS_URL = (
        "non_https_url"
    )

    EMBEDDED_CREDENTIALS = (
        "embedded_credentials"
    )

    IP_LITERAL_HOST = (
        "ip_literal_host"
    )

    DISALLOWED_PORT = (
        "disallowed_port"
    )

    DISALLOWED_DOMAIN = (
        "disallowed_domain"
    )


_DOMAIN_LABEL_RE = re.compile(
    r"^[a-z0-9]"
    r"(?:[a-z0-9-]{0,61}[a-z0-9])?$"
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


def _optional_text(
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

    return value.strip()


def _normalize_domain(
    value: str,
) -> str:
    domain = _required_text(
        value,
        "domain",
    ).lower()

    domain = domain.rstrip(
        "."
    )

    if any(
        character in domain
        for character in (
            "/",
            "\\",
            ":",
            "@",
            "*",
            "?",
            "#",
        )
    ):
        raise ValueError(
            "Allowed domains must contain "
            "hostnames only."
        )

    try:
        domain = (
            domain
            .encode("idna")
            .decode("ascii")
            .lower()
        )
    except UnicodeError as exc:
        raise ValueError(
            "Domain is not a valid IDNA hostname."
        ) from exc

    if len(domain) > 253:
        raise ValueError(
            "Domain is too long."
        )

    labels = domain.split(
        "."
    )

    if (
        len(labels) < 2
        or any(
            not label
            for label in labels
        )
        or any(
            not _DOMAIN_LABEL_RE.fullmatch(
                label
            )
            for label in labels
        )
    ):
        raise ValueError(
            "Domain is not a valid hostname."
        )

    try:
        ipaddress.ip_address(
            domain
        )
    except ValueError:
        pass
    else:
        raise ValueError(
            "IP addresses cannot be "
            "allowlisted as government domains."
        )

    return domain


def _policy_id(
    *,
    allowed_domains: tuple[
        str,
        ...
    ],
    allow_subdomains: bool,
    require_https: bool,
) -> str:
    payload = "\x1f".join(
        (
            SOURCE_ALLOWLIST_VERSION,
            ",".join(
                allowed_domains
            ),
            str(
                allow_subdomains
            ).lower(),
            str(
                require_https
            ).lower(),
        )
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        payload
    ).hexdigest()


@dataclass(frozen=True)
class SourceAllowlistPolicy:
    """Explicit deployment allowlist for authoritative web sources."""

    allowed_domains: tuple[
        str,
        ...
    ]

    allow_subdomains: bool = True

    require_https: bool = True

    version: str = (
        SOURCE_ALLOWLIST_VERSION
    )

    policy_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        try:
            domains = tuple(
                self.allowed_domains
            )
        except TypeError as exc:
            raise TypeError(
                "allowed_domains must be iterable."
            ) from exc

        if not domains:
            raise ValueError(
                "At least one allowed domain "
                "is required."
            )

        normalized = tuple(
            sorted(
                {
                    _normalize_domain(
                        domain
                    )
                    for domain
                    in domains
                }
            )
        )

        if not isinstance(
            self.allow_subdomains,
            bool,
        ):
            raise TypeError(
                "allow_subdomains must be boolean."
            )

        if not isinstance(
            self.require_https,
            bool,
        ):
            raise TypeError(
                "require_https must be boolean."
            )

        object.__setattr__(
            self,
            "allowed_domains",
            normalized,
        )

        object.__setattr__(
            self,
            "policy_id",
            _policy_id(
                allowed_domains=(
                    normalized
                ),
                allow_subdomains=(
                    self.allow_subdomains
                ),
                require_https=(
                    self.require_https
                ),
            ),
        )

    def allows_domain(
        self,
        domain: str,
    ) -> bool:
        normalized = _normalize_domain(
            domain
        )

        for allowed in (
            self.allowed_domains
        ):
            if normalized == allowed:
                return True

            if (
                self.allow_subdomains
                and normalized.endswith(
                    "." + allowed
                )
            ):
                return True

        return False

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "version": (
                self.version
            ),
            "policy_id": (
                self.policy_id
            ),
            "allowed_domains": list(
                self.allowed_domains
            ),
            "allow_subdomains": (
                self.allow_subdomains
            ),
            "require_https": (
                self.require_https
            ),
        }


@dataclass(frozen=True)
class SourceAllowlistResult:
    """Privacy-safe result of one source trust assessment."""

    trace_id: str

    document_id: str

    domain: str

    reason: (
        SourceAllowlistReasonCode
    )

    security_assessment: (
        SecurityAssessment
    )

    policy_id: str

    version: str = (
        SOURCE_ALLOWLIST_VERSION
    )

    def __post_init__(self) -> None:
        trace_id = _required_text(
            self.trace_id,
            "trace_id",
        )

        document_id = _required_text(
            self.document_id,
            "document_id",
        )

        if not isinstance(
            self.domain,
            str,
        ):
            raise TypeError(
                "domain must be a string."
            )

        domain = self.domain.strip().lower()

        if not isinstance(
            self.reason,
            SourceAllowlistReasonCode,
        ):
            raise TypeError(
                "reason must be a "
                "SourceAllowlistReasonCode."
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
                "must match source assessment."
            )

        policy_id = _required_text(
            self.policy_id,
            "policy_id",
        )

        if (
            len(policy_id) != 64
            or any(
                character
                not in "0123456789abcdef"
                for character
                in policy_id.lower()
            )
        ):
            raise ValueError(
                "policy_id must be a SHA-256 "
                "hexadecimal digest."
            )

        allowed = (
            self.reason
            is SourceAllowlistReasonCode.ALLOWED
        )

        if (
            allowed
            and not self.security_assessment.allowed
        ):
            raise ValueError(
                "Allowed source cannot have "
                "a blocking security assessment."
            )

        if (
            not allowed
            and not self.security_assessment.should_block
        ):
            raise ValueError(
                "Disallowed source must fail closed."
            )

        object.__setattr__(
            self,
            "trace_id",
            trace_id,
        )

        object.__setattr__(
            self,
            "document_id",
            document_id,
        )

        object.__setattr__(
            self,
            "domain",
            domain,
        )

        object.__setattr__(
            self,
            "policy_id",
            policy_id.lower(),
        )

    @property
    def allowed(
        self,
    ) -> bool:
        return (
            self.reason
            is SourceAllowlistReasonCode.ALLOWED
        )

    @property
    def should_block(
        self,
    ) -> bool:
        return not self.allowed

    def to_dict(
        self,
    ) -> dict[str, object]:
        """Serialize without exposing the complete source URL."""

        return {
            "version": (
                self.version
            ),
            "trace_id": (
                self.trace_id
            ),
            "document_id": (
                self.document_id
            ),
            "domain": (
                self.domain
                or None
            ),
            "reason": (
                self.reason.value
            ),
            "allowed": (
                self.allowed
            ),
            "should_block": (
                self.should_block
            ),
            "policy_id": (
                self.policy_id
            ),
            "security_assessment": (
                self.security_assessment
                .to_dict()
            ),
        }


def _extract_domain_and_reason(
    source_url: str,
    policy: SourceAllowlistPolicy,
) -> tuple[
    str,
    SourceAllowlistReasonCode,
]:
    value = _optional_text(
        source_url,
        "source_url",
    )

    if not value:
        return (
            "",
            SourceAllowlistReasonCode
            .MISSING_OFFICIAL_URL,
        )

    try:
        parsed = urlsplit(
            value
        )
    except ValueError:
        return (
            "",
            SourceAllowlistReasonCode.INVALID_URL,
        )

    if (
        parsed.username is not None
        or parsed.password is not None
    ):
        return (
            "",
            SourceAllowlistReasonCode
            .EMBEDDED_CREDENTIALS,
        )

    scheme = (
        parsed.scheme
        .strip()
        .lower()
    )

    if policy.require_https:
        if scheme != "https":
            return (
                "",
                SourceAllowlistReasonCode
                .NON_HTTPS_URL,
            )
    elif scheme not in {
        "http",
        "https",
    }:
        return (
            "",
            SourceAllowlistReasonCode.INVALID_URL,
        )

    hostname = parsed.hostname

    if not hostname:
        return (
            "",
            SourceAllowlistReasonCode.INVALID_URL,
        )

    try:
        domain = (
            hostname
            .rstrip(".")
            .encode("idna")
            .decode("ascii")
            .lower()
        )
    except UnicodeError:
        return (
            "",
            SourceAllowlistReasonCode.INVALID_URL,
        )

    try:
        ipaddress.ip_address(
            domain
        )
    except ValueError:
        pass
    else:
        return (
            domain,
            SourceAllowlistReasonCode
            .IP_LITERAL_HOST,
        )

    try:
        port = parsed.port
    except ValueError:
        return (
            domain,
            SourceAllowlistReasonCode.INVALID_URL,
        )

    if port is not None:
        expected_port = (
            443
            if scheme == "https"
            else 80
        )

        if port != expected_port:
            return (
                domain,
                SourceAllowlistReasonCode
                .DISALLOWED_PORT,
            )

    try:
        domain_allowed = (
            policy.allows_domain(
                domain
            )
        )
    except ValueError:
        return (
            domain,
            SourceAllowlistReasonCode.INVALID_URL,
        )

    if not domain_allowed:
        return (
            domain,
            SourceAllowlistReasonCode
            .DISALLOWED_DOMAIN,
        )

    return (
        domain,
        SourceAllowlistReasonCode.ALLOWED,
    )


def _security_issue_code(
    reason: SourceAllowlistReasonCode,
) -> SecurityIssueCode:
    if (
        reason
        is SourceAllowlistReasonCode
        .DISALLOWED_DOMAIN
    ):
        return (
            SecurityIssueCode
            .DISALLOWED_DOMAIN
        )

    return (
        SecurityIssueCode
        .UNTRUSTED_SOURCE
    )


def assess_source_url(
    source_url: str,
    *,
    document_id: str,
    trace_id: str,
    policy: SourceAllowlistPolicy,
) -> SourceAllowlistResult:
    """Assess one source URL against the explicit allowlist."""

    document = _required_text(
        document_id,
        "document_id",
    )

    trace = _required_text(
        trace_id,
        "trace_id",
    )

    if not isinstance(
        policy,
        SourceAllowlistPolicy,
    ):
        raise TypeError(
            "policy must be a "
            "SourceAllowlistPolicy."
        )

    domain, reason = (
        _extract_domain_and_reason(
            source_url,
            policy,
        )
    )

    if (
        reason
        is SourceAllowlistReasonCode.ALLOWED
    ):
        findings = ()

    else:
        reference_payload = "\x1f".join(
            (
                document,
                domain,
                reason.value,
            )
        )

        reference_hash = hashlib.sha256(
            reference_payload.encode(
                "utf-8"
            )
        ).hexdigest()

        findings = (
            SecurityFinding(
                code=(
                    _security_issue_code(
                        reason
                    )
                ),
                surface=(
                    SecuritySurface.SOURCE
                ),
                severity=(
                    SecuritySeverity.HIGH
                ),
                decision=(
                    SecurityDecision.BLOCK
                ),
                occurrence_count=1,
                reference_id=(
                    f"source:"
                    f"{reference_hash[:16]}"
                ),
            ),
        )

    security = assess_security(
        trace_id=trace,
        findings=findings,
    )

    return SourceAllowlistResult(
        trace_id=trace,
        document_id=document,
        domain=domain,
        reason=reason,
        security_assessment=security,
        policy_id=policy.policy_id,
    )


def assess_authoritative_source(
    source: AuthoritativeSource,
    *,
    trace_id: str,
    policy: SourceAllowlistPolicy,
) -> SourceAllowlistResult:
    """Validate the official URL attached to an authoritative source."""

    if not isinstance(
        source,
        AuthoritativeSource,
    ):
        raise TypeError(
            "source must be an "
            "AuthoritativeSource."
        )

    return assess_source_url(
        source.official_source_url,
        document_id=(
            source.document_id
        ),
        trace_id=trace_id,
        policy=policy,
    )
