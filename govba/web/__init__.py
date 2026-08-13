"""Controlled official-web intelligence for GovBA-GAR."""

from govba.web.adapter import (
    DEFAULT_WEB_CANDIDATE_POOL,
    GOVERNMENT_WEB_ADAPTER_VERSION,
    GovernmentWebAdapter,
    RawWebSearchResult,
    WebSearchBackend,
    is_official_web_retriever,
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
