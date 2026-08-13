"""Controlled official-web intelligence for GovBA-GAR."""

from govba.web.adapter import (
    DEFAULT_WEB_CANDIDATE_POOL,
    GOVERNMENT_WEB_ADAPTER_VERSION,
    GovernmentWebAdapter,
    RawWebSearchResult,
    WebSearchBackend,
    is_official_web_retriever,
)
from govba.web.evaluation import (
    CONTROLLED_WEB_EVALUATION_VERSION,
    ControlledWebBenchmarkResult,
    ControlledWebCaseResult,
    ControlledWebGoldCase,
    ControlledWebMetrics,
    evaluate_controlled_web,
)
from govba.web.freshness import (
    WEB_FRESHNESS_VERSION,
    WebEvidenceControlAssessment,
    WebEvidenceControlDecision,
    WebEvidenceControlIssue,
    WebEvidenceControlIssueCode,
    WebEvidenceControlPolicy,
    WebFreshnessState,
    WebProvenanceState,
    assess_web_evidence_controls,
)
from govba.web.evidence_bridge import (
    WEB_EVIDENCE_BRIDGE_VERSION,
    WebEvidenceBridgeResult,
    WebEvidenceMetadata,
    bridge_official_web_result,
    build_web_document_id,
)
from govba.web.contract import (
    OFFICIAL_WEB_RETRIEVAL_VERSION,
    OfficialWebRequest,
    OfficialWebResult,
    OfficialWebRetriever,
)


__all__ = [
    "CONTROLLED_WEB_EVALUATION_VERSION",

    "ControlledWebBenchmarkResult",

    "ControlledWebCaseResult",

    "ControlledWebGoldCase",

    "ControlledWebMetrics",

    "evaluate_controlled_web",

    "WEB_FRESHNESS_VERSION",

    "WebEvidenceControlAssessment",

    "WebEvidenceControlDecision",

    "WebEvidenceControlIssue",

    "WebEvidenceControlIssueCode",

    "WebEvidenceControlPolicy",

    "WebFreshnessState",

    "WebProvenanceState",

    "assess_web_evidence_controls",

    "WEB_EVIDENCE_BRIDGE_VERSION",

    "WebEvidenceBridgeResult",

    "WebEvidenceMetadata",

    "bridge_official_web_result",

    "build_web_document_id",

    "DEFAULT_WEB_CANDIDATE_POOL",

    "GOVERNMENT_WEB_ADAPTER_VERSION",

    "GovernmentWebAdapter",

    "RawWebSearchResult",

    "WebSearchBackend",

    "is_official_web_retriever",

    "OFFICIAL_WEB_RETRIEVAL_VERSION",
    "OfficialWebRequest",
    "OfficialWebResult",
    "OfficialWebRetriever",
]
