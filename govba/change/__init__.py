"""Policy and circular change intelligence for GovBA-GAR."""

from govba.change.version_matching import (
    DOCUMENT_VERSION_MATCHING_VERSION,
    DocumentVersionMatchDecision,
    DocumentVersionMatchReason,
    DocumentVersionMatchResult,
    build_change_request_from_match,
    match_document_versions,
)
from govba.change.briefing import (
    CHANGE_BRIEFING_VERSION,
    ChangeBriefingDecision,
    ChangeBriefingItem,
    ChangeBriefingReason,
    ChangeEvidenceReference,
    GovernedChangeBriefing,
    build_governed_change_briefing,
)
from govba.change.clause_detection import (
    CLAUSE_CHANGE_DETECTION_ALGORITHM,
    CLAUSE_CHANGE_DETECTION_VERSION,
    DEFAULT_MODIFICATION_SIMILARITY_THRESHOLD,
    ClauseChangeDetectionResult,
    ClauseChangeMatch,
    ClauseChangeReason,
    clause_similarity,
    detect_clause_changes,
)
from govba.change.significance import (
    CHANGE_SIGNIFICANCE_ALGORITHM,
    CHANGE_SIGNIFICANCE_VERSION,
    ChangeSignificanceAssessment,
    ChangeSignificancePolicy,
    ChangeSignificanceResult,
    ChangeSignificanceSignal,
    classify_change_significance,
    detect_significance_signals,
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
    "CHANGE_BRIEFING_VERSION",

    "ChangeBriefingDecision",

    "ChangeBriefingItem",

    "ChangeBriefingReason",

    "ChangeEvidenceReference",

    "GovernedChangeBriefing",

    "build_governed_change_briefing",

    "CHANGE_SIGNIFICANCE_ALGORITHM",

    "CHANGE_SIGNIFICANCE_VERSION",

    "ChangeSignificanceAssessment",

    "ChangeSignificancePolicy",

    "ChangeSignificanceResult",

    "ChangeSignificanceSignal",

    "classify_change_significance",

    "detect_significance_signals",

    "CLAUSE_CHANGE_DETECTION_ALGORITHM",

    "CLAUSE_CHANGE_DETECTION_VERSION",

    "DEFAULT_MODIFICATION_SIMILARITY_THRESHOLD",

    "ClauseChangeDetectionResult",

    "ClauseChangeMatch",

    "ClauseChangeReason",

    "clause_similarity",

    "detect_clause_changes",

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
