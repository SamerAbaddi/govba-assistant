"""Controlled official-web intelligence for GovBA-GAR."""

from govba.web.adapter import (
    DEFAULT_WEB_CANDIDATE_POOL,
    GOVERNMENT_WEB_ADAPTER_VERSION,
    GovernmentWebAdapter,
    RawWebSearchResult,
    WebSearchBackend,
    is_official_web_retriever,
)
from govba.web.contract import (
    OFFICIAL_WEB_RETRIEVAL_VERSION,
    OfficialWebRequest,
    OfficialWebResult,
    OfficialWebRetriever,
)


__all__ = [
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
