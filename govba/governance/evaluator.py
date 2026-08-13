"""Unified governance abstention evaluation for GovBA-GAR."""

from __future__ import annotations

from govba.governance.eligibility import (
    EvidenceEligibilityReason,
    EvidenceEligibilityReport,
)
from govba.governance.verification import (
    GovernanceDecision,
    GovernanceReasonCode,
    GovernanceVerificationResult,
)
from govba.rag.grounding import (
    GroundingIssueCode,
    GroundingValidationReport,
)


GOVERNANCE_EVALUATOR_VERSION = (
    "govba-governance-evaluator-v1"
)


_ELIGIBILITY_REASON_MAP = {
    EvidenceEligibilityReason.REJECTED_EVIDENCE: (
        GovernanceReasonCode.REJECTED_EVIDENCE
    ),
    EvidenceEligibilityReason.SUPERSEDED_EVIDENCE: (
        GovernanceReasonCode.SUPERSEDED_EVIDENCE
    ),
    EvidenceEligibilityReason.TEMPORAL_CONFLICT: (
        GovernanceReasonCode.TEMPORAL_CONFLICT
    ),
    EvidenceEligibilityReason.TEMPORAL_INSUFFICIENT: (
        GovernanceReasonCode.TEMPORAL_INSUFFICIENT
    ),
    EvidenceEligibilityReason.TEMPORAL_UNASSESSED: (
        GovernanceReasonCode.TEMPORAL_INSUFFICIENT
    ),
}


class GovernanceAbstentionEvaluator:
    """Combine grounding and eligibility into a fail-safe decision."""

    def evaluate(
        self,
        grounding_report: GroundingValidationReport,
        eligibility_report: EvidenceEligibilityReport,
        *,
        trace_id: str,
    ) -> GovernanceVerificationResult:
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

        grounding_card_ids = tuple(
            card.card_id
            for card in grounding_report.cards
        )

        eligibility_card_ids = tuple(
            card.card_id
            for card in eligibility_report.cards
        )

        if grounding_card_ids != eligibility_card_ids:
            raise ValueError(
                "Grounding and eligibility reports "
                "must refer to the same evidence "
                "cards in the same order."
            )

        reasons = set()

        if not grounding_report.is_grounded:
            reasons.add(
                GovernanceReasonCode
                .GROUNDING_FAILURE
            )

        grounding_issue_codes = {
            issue.code
            for issue in grounding_report.issues
        }

        if (
            GroundingIssueCode.UNKNOWN_CITATION
            in grounding_issue_codes
        ):
            reasons.add(
                GovernanceReasonCode
                .UNKNOWN_CITATION
            )

        if (
            GroundingIssueCode.REJECTED_EVIDENCE
            in grounding_issue_codes
        ):
            reasons.add(
                GovernanceReasonCode
                .REJECTED_EVIDENCE
            )

        eligibility_map = {
            result.card_id: result
            for result
            in eligibility_report.results
        }

        cited_card_ids = tuple(
            dict.fromkeys(
                citation.card_id
                for citation
                in grounding_report.citations
                if citation.card_id
                in eligibility_map
            )
        )

        eligible_cited_card_ids = []

        for card_id in cited_card_ids:
            result = eligibility_map[
                card_id
            ]

            if result.eligible:
                eligible_cited_card_ids.append(
                    card_id
                )

                continue

            for reason in result.reason_codes:
                governance_reason = (
                    _ELIGIBILITY_REASON_MAP
                    .get(
                        reason
                    )
                )

                if governance_reason is not None:
                    reasons.add(
                        governance_reason
                    )

        if not eligible_cited_card_ids:
            reasons.add(
                GovernanceReasonCode
                .NO_ELIGIBLE_EVIDENCE
            )

        if reasons:
            return GovernanceVerificationResult(
                trace_id=trace_id,
                decision=(
                    GovernanceDecision.ABSTAIN
                ),
                reason_codes=tuple(
                    reasons
                ),
                evidence_count=(
                    len(
                        grounding_report.cards
                    )
                ),
                eligible_evidence_count=(
                    len(
                        eligible_cited_card_ids
                    )
                ),
            )

        return GovernanceVerificationResult(
            trace_id=trace_id,
            decision=(
                GovernanceDecision.ALLOW
            ),
            reason_codes=(
                GovernanceReasonCode.VERIFIED,
            ),
            evidence_count=(
                len(
                    grounding_report.cards
                )
            ),
            eligible_evidence_count=(
                len(
                    eligible_cited_card_ids
                )
            ),
        )


def evaluate_governance(
    grounding_report: GroundingValidationReport,
    eligibility_report: EvidenceEligibilityReport,
    *,
    trace_id: str,
) -> GovernanceVerificationResult:
    """Convenience function for unified governance evaluation."""

    return GovernanceAbstentionEvaluator().evaluate(
        grounding_report,
        eligibility_report,
        trace_id=trace_id,
    )
