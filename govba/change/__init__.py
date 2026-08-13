"""Policy and circular change intelligence for GovBA-GAR."""

from govba.change.version_matching import (
    DOCUMENT_VERSION_MATCHING_VERSION,
    DocumentVersionMatchDecision,
    DocumentVersionMatchReason,
    DocumentVersionMatchResult,
    build_change_request_from_match,
    match_document_versions,
)
from govba.change.contract import (
    POLICY_CHANGE_CONTRACT_VERSION,
    PolicyChangeFinding,
    PolicyChangeImpact,
    PolicyChangeReport,
    PolicyChangeRequest,
    PolicyChangeType,
    build_policy_change_report,
)


__all__ = [
    "DOCUMENT_VERSION_MATCHING_VERSION",

    "DocumentVersionMatchDecision",

    "DocumentVersionMatchReason",

    "DocumentVersionMatchResult",

    "build_change_request_from_match",

    "match_document_versions",

    "POLICY_CHANGE_CONTRACT_VERSION",
    "PolicyChangeFinding",
    "PolicyChangeImpact",
    "PolicyChangeReport",
    "PolicyChangeRequest",
    "PolicyChangeType",
    "build_policy_change_report",
]
