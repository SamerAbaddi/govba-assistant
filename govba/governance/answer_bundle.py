"""Governed answer packaging for GovBA-GAR."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from govba.governance.eligibility import (
    EvidenceEligibilityReport,
)
from govba.governance.evaluator import (
    evaluate_governance,
)
from govba.governance.verification import (
    GovernanceDecision,
    GovernanceVerificationResult,
)
from govba.rag.evidence_card import (
    EvidenceCard,
)
from govba.rag.grounding import (
    GroundingValidationReport,
)


GOVERNED_ANSWER_BUNDLE_VERSION = (
    "govba-governed-answer-bundle-v1"
)

DEFAULT_ABSTENTION_MESSAGE = (
    "I cannot provide a verified answer from "
    "the available authoritative evidence."
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


def _render_allowed_response(
    grounding_report: GroundingValidationReport,
) -> str:
    """Render validated claims with visible citation markers."""

    citation_map = {
        citation.citation_id: citation
        for citation
        in grounding_report.citations
    }

    rendered_claims = []

    for claim in grounding_report.claims:
        text = claim.text.strip()

        inline_ids = set(
            claim.inline_citation_ids
        )

        missing_markers = []

        for citation_id in (
            claim.citation_ids
        ):
            if citation_id in inline_ids:
                continue

            citation = citation_map.get(
                citation_id
            )

            if citation is None:
                raise ValueError(
                    "Cannot render an allowed answer "
                    "with an unknown citation."
                )

            missing_markers.append(
                citation.marker
            )

        if missing_markers:
            text = (
                f"{text} "
                f"{' '.join(missing_markers)}"
            )

        rendered_claims.append(
            text
        )

    response_text = "\n\n".join(
        rendered_claims
    ).strip()

    if not response_text:
        raise ValueError(
            "Allowed answer must contain "
            "at least one rendered claim."
        )

    return response_text


def _cited_card_ids(
    grounding_report: GroundingValidationReport,
) -> tuple[str, ...]:
    citation_map = {
        citation.citation_id: citation
        for citation
        in grounding_report.citations
    }

    card_ids = []

    for claim in grounding_report.claims:
        for citation_id in (
            claim.evidence_references
        ):
            citation = citation_map.get(
                citation_id
            )

            if citation is None:
                continue

            if (
                citation.card_id
                not in card_ids
            ):
                card_ids.append(
                    citation.card_id
                )

    return tuple(
        card_ids
    )


def _bundle_id(
    *,
    trace_id: str,
    verification_id: str,
    response_sha256: str,
    cited_card_ids: tuple[
        str,
        ...
    ],
) -> str:
    payload = "\x1f".join(
        (
            GOVERNED_ANSWER_BUNDLE_VERSION,
            trace_id,
            verification_id,
            response_sha256,
            ",".join(
                cited_card_ids
            ),
        )
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        payload
    ).hexdigest()


@dataclass(frozen=True)
class GovernedAnswerBundle:
    """Final governed response package for UI or API delivery."""

    trace_id: str

    response_text: str

    grounding_report: (
        GroundingValidationReport
    )

    eligibility_report: (
        EvidenceEligibilityReport
    )

    verification: (
        GovernanceVerificationResult
    )

    version: str = (
        GOVERNED_ANSWER_BUNDLE_VERSION
    )

    response_sha256: str = field(
        init=False
    )

    bundle_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        trace_id = _required_text(
            self.trace_id,
            "trace_id",
        )

        response_text = _required_text(
            self.response_text,
            "response_text",
        )

        if not isinstance(
            self.grounding_report,
            GroundingValidationReport,
        ):
            raise TypeError(
                "grounding_report must be a "
                "GroundingValidationReport."
            )

        if not isinstance(
            self.eligibility_report,
            EvidenceEligibilityReport,
        ):
            raise TypeError(
                "eligibility_report must be an "
                "EvidenceEligibilityReport."
            )

        if not isinstance(
            self.verification,
            GovernanceVerificationResult,
        ):
            raise TypeError(
                "verification must be a "
                "GovernanceVerificationResult."
            )

        if (
            self.verification.trace_id
            != trace_id
        ):
            raise ValueError(
                "Verification trace ID does not "
                "match answer bundle trace ID."
            )

        expected_verification = (
            evaluate_governance(
                self.grounding_report,
                self.eligibility_report,
                trace_id=trace_id,
            )
        )

        if (
            self.verification
            != expected_verification
        ):
            raise ValueError(
                "Verification result does not match "
                "the supplied governance reports."
            )

        if (
            self.verification.decision
            is GovernanceDecision.ALLOW
        ):
            expected_response = (
                _render_allowed_response(
                    self.grounding_report
                )
            )

            if (
                response_text
                != expected_response
            ):
                raise ValueError(
                    "Allowed response text must be "
                    "rendered from validated claims."
                )

        else:
            if (
                response_text
                != DEFAULT_ABSTENTION_MESSAGE
            ):
                raise ValueError(
                    "Abstained responses must use "
                    "the controlled abstention message."
                )

        cited_ids = _cited_card_ids(
            self.grounding_report
        )

        response_hash = _sha256_text(
            response_text
        )

        object.__setattr__(
            self,
            "trace_id",
            trace_id,
        )

        object.__setattr__(
            self,
            "response_text",
            response_text,
        )

        object.__setattr__(
            self,
            "response_sha256",
            response_hash,
        )

        object.__setattr__(
            self,
            "bundle_id",
            _bundle_id(
                trace_id=trace_id,
                verification_id=(
                    self.verification
                    .decision_id
                ),
                response_sha256=(
                    response_hash
                ),
                cited_card_ids=(
                    cited_ids
                ),
            ),
        )

    @property
    def allowed(
        self,
    ) -> bool:
        return (
            self.verification.allowed
        )

    @property
    def should_abstain(
        self,
    ) -> bool:
        return (
            self.verification
            .should_abstain
        )

    @property
    def claims(
        self,
    ):
        return (
            self.grounding_report.claims
        )

    @property
    def citations(
        self,
    ):
        return (
            self.grounding_report.citations
        )

    @property
    def evidence_cards(
        self,
    ):
        return (
            self.grounding_report.cards
        )

    @property
    def cited_card_ids(
        self,
    ) -> tuple[str, ...]:
        return _cited_card_ids(
            self.grounding_report
        )

    @property
    def cited_evidence_cards(
        self,
    ) -> tuple[
        EvidenceCard,
        ...
    ]:
        cited_ids = set(
            self.cited_card_ids
        )

        return tuple(
            card
            for card
            in self.evidence_cards
            if card.card_id
            in cited_ids
        )

    def to_dict(
        self,
        *,
        include_response_text: bool = False,
    ) -> dict[str, object]:
        """Serialize governed-answer metadata safely by default."""

        if not isinstance(
            include_response_text,
            bool,
        ):
            raise TypeError(
                "include_response_text "
                "must be boolean."
            )

        data: dict[
            str,
            object,
        ] = {
            "version": (
                self.version
            ),
            "bundle_id": (
                self.bundle_id
            ),
            "trace_id": (
                self.trace_id
            ),
            "decision": (
                self.verification
                .decision.value
            ),
            "allowed": (
                self.allowed
            ),
            "should_abstain": (
                self.should_abstain
            ),
            "response_sha256": (
                self.response_sha256
            ),
            "claim_count": len(
                self.claims
            ),
            "citation_count": len(
                self.citations
            ),
            "evidence_count": len(
                self.evidence_cards
            ),
            "cited_card_ids": list(
                self.cited_card_ids
            ),
            "verification": (
                self.verification
                .to_dict()
            ),
        }

        if include_response_text:
            data[
                "response_text"
            ] = self.response_text

        return data


def build_governed_answer_bundle(
    grounding_report: (
        GroundingValidationReport
    ),
    eligibility_report: (
        EvidenceEligibilityReport
    ),
    *,
    trace_id: str,
) -> GovernedAnswerBundle:
    """Build a final answer or controlled abstention deterministically."""

    if not isinstance(
        grounding_report,
        GroundingValidationReport,
    ):
        raise TypeError(
            "grounding_report must be a "
            "GroundingValidationReport."
        )

    if not isinstance(
        eligibility_report,
        EvidenceEligibilityReport,
    ):
        raise TypeError(
            "eligibility_report must be an "
            "EvidenceEligibilityReport."
        )

    verification = evaluate_governance(
        grounding_report,
        eligibility_report,
        trace_id=trace_id,
    )

    if verification.allowed:
        response_text = (
            _render_allowed_response(
                grounding_report
            )
        )
    else:
        response_text = (
            DEFAULT_ABSTENTION_MESSAGE
        )

    return GovernedAnswerBundle(
        trace_id=(
            verification.trace_id
        ),
        response_text=(
            response_text
        ),
        grounding_report=(
            grounding_report
        ),
        eligibility_report=(
            eligibility_report
        ),
        verification=verification,
    )
