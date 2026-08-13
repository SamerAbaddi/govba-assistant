"""Correspondence intelligence for GovBA-GAR."""

from govba.correspondence.actions import (
    CORRESPONDENCE_ACTION_ALGORITHM,
    CORRESPONDENCE_ACTION_VERSION,
    CorrespondenceAction,
    CorrespondenceActionExtractionResult,
    CorrespondenceActionKind,
    CorrespondenceActionNature,
    CorrespondenceActionOwner,
    extract_correspondence_actions,
)
from govba.correspondence.briefing import (
    CORRESPONDENCE_BRIEFING_VERSION,
    CorrespondenceBriefingAction,
    CorrespondenceBriefingDeadline,
    CorrespondenceBriefingDecision,
    CorrespondenceBriefingPolicy,
    CorrespondenceBriefingReason,
    GovernedCorrespondenceBriefing,
    build_governed_correspondence_briefing,
)
from govba.correspondence.classification import (
    CORRESPONDENCE_CLASSIFICATION_ALGORITHM,
    CORRESPONDENCE_CLASSIFICATION_VERSION,
    CorrespondenceIntentSignal,
    ParsedCorrespondence,
    analyze_correspondence_intent,
    classify_correspondence_intent,
    detect_intent_signals,
    parse_correspondence,
)
from govba.correspondence.priority import (
    CORRESPONDENCE_PRIORITY_ALGORITHM,
    CORRESPONDENCE_PRIORITY_VERSION,
    CorrespondenceDeadline,
    CorrespondenceDeadlineKind,
    CorrespondenceDeadlineResult,
    CorrespondencePriorityAssessment,
    CorrespondencePriorityReason,
    assess_correspondence_priority,
    extract_correspondence_deadlines,
)
from govba.correspondence.contract import (
    CORRESPONDENCE_CONTRACT_VERSION,
    CorrespondenceChannel,
    CorrespondenceClassification,
    CorrespondenceDirection,
    CorrespondenceIntelligenceResult,
    CorrespondenceIntent,
    CorrespondencePriority,
    CorrespondenceRequest,
)


__all__ = [
    "CORRESPONDENCE_BRIEFING_VERSION",

    "CorrespondenceBriefingAction",

    "CorrespondenceBriefingDeadline",

    "CorrespondenceBriefingDecision",

    "CorrespondenceBriefingPolicy",

    "CorrespondenceBriefingReason",

    "GovernedCorrespondenceBriefing",

    "build_governed_correspondence_briefing",

    "CORRESPONDENCE_PRIORITY_ALGORITHM",

    "CORRESPONDENCE_PRIORITY_VERSION",

    "CorrespondenceDeadline",

    "CorrespondenceDeadlineKind",

    "CorrespondenceDeadlineResult",

    "CorrespondencePriorityAssessment",

    "CorrespondencePriorityReason",

    "assess_correspondence_priority",

    "extract_correspondence_deadlines",

    "CORRESPONDENCE_ACTION_ALGORITHM",

    "CORRESPONDENCE_ACTION_VERSION",

    "CorrespondenceAction",

    "CorrespondenceActionExtractionResult",

    "CorrespondenceActionKind",

    "CorrespondenceActionNature",

    "CorrespondenceActionOwner",

    "extract_correspondence_actions",

    "CORRESPONDENCE_CLASSIFICATION_ALGORITHM",

    "CORRESPONDENCE_CLASSIFICATION_VERSION",

    "CorrespondenceIntentSignal",

    "ParsedCorrespondence",

    "analyze_correspondence_intent",

    "classify_correspondence_intent",

    "detect_intent_signals",

    "parse_correspondence",

    "CORRESPONDENCE_CONTRACT_VERSION",
    "CorrespondenceChannel",
    "CorrespondenceClassification",
    "CorrespondenceDirection",
    "CorrespondenceIntelligenceResult",
    "CorrespondenceIntent",
    "CorrespondencePriority",
    "CorrespondenceRequest",
]
