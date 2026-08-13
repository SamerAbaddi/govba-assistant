"""Correspondence intelligence for GovBA-GAR."""

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
