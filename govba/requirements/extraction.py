"""Deterministic bilingual requirement extraction for GovBA-GAR."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum

from govba.rag.chunking import normalize_ingested_text
from govba.requirements.contract import (
    BusinessRequirementsDocument,
    RequirementItem,
    RequirementPriority,
    RequirementStatus,
    RequirementType,
    RequirementsRequest,
)


REQUIREMENT_EXTRACTION_VERSION = (
    "govba-requirement-extraction-v1"
)

REQUIREMENT_EXTRACTION_ALGORITHM = (
    "deterministic-bilingual-requirement-extraction-v1"
)


class RequirementDetectionSignal(
    str,
    Enum,
):
    MUST = "must_signal"
    SHOULD = "should_signal"
    COULD = "could_signal"
    WONT = "wont_signal"


_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?؟؛;])\s+|\n+"
)

_LEADING_LIST_MARKER_RE = re.compile(
    r"^\s*(?:"
    r"[-*•–—]+"
    r"|\d+[.)]"
    r"|[A-Za-z][.)]"
    r")\s*"
)


_PRIORITY_PATTERNS = {
    RequirementPriority.WONT: (
        re.compile(
            r"\bout\s+of\s+scope\b"
            r"|\bnot\s+in\s+scope\b"
            r"|\bexcluded\s+from\s+scope\b"
            r"|\bwill\s+not\s+be\s+included\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"خارج\s+النطاق"
            r"|ليس\s+ضمن\s+النطاق"
            r"|مستبعد\s+من\s+النطاق"
        ),
    ),

    RequirementPriority.MUST: (
        re.compile(
            r"\bmust(?:\s+not)?\b"
            r"|\bshall(?:\s+not)?\b"
            r"|\bis\s+required\s+to\b"
            r"|\bare\s+required\s+to\b"
            r"|\brequired\s+to\b"
            r"|\bmandatory\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"يجب"
            r"|يتعين"
            r"|يلزم"
            r"|مطلوب\s+من"
            r"|إلزامي"
            r"|الزامي"
        ),
    ),

    RequirementPriority.SHOULD: (
        re.compile(
            r"\bshould(?:\s+not)?\b"
            r"|\brecommended\s+to\b"
            r"|\bpreferably\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"ينبغي"
            r"|يفضل"
            r"|يُفضل"
            r"|يوصى"
            r"|يُوصى"
        ),
    ),

    RequirementPriority.COULD: (
        re.compile(
            r"\bcould\b"
            r"|\bmay\b"
            r"|\boptionally\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"يمكن"
            r"|اختياري"
            r"|اختيارياً"
            r"|اختياريًا"
        ),
    ),
}


_TYPE_PATTERNS = {
    RequirementType.COMPLIANCE: (
        re.compile(
            r"\bcompl(?:y|ies|iance)\b"
            r"|\blaw\b"
            r"|\bregulation\b"
            r"|\blegislation\b"
            r"|\bpolicy\b"
            r"|\bstandard\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"امتثال"
            r"|قانون"
            r"|تشريع"
            r"|لائحة"
            r"|نظام\s+(?:قانوني|تشريعي|تنظيمي)"
            r"|الأنظمة\s+(?:القانونية|التشريعية|التنظيمية)"
            r"|سياسة"
            r"|معيار"
        ),
    ),

    RequirementType.SECURITY: (
        re.compile(
            r"\bsecurity\b"
            r"|\bsecure\b"
            r"|\bauthentication\b"
            r"|\bauthori[sz]ation\b"
            r"|\bencryption\b"
            r"|\baccess\s+control\b"
            r"|\bprivacy\b"
            r"|\bpassword\b"
            r"|\bmulti[- ]factor\b"
            r"|\bmfa\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"أمن"
            r"|امن"
            r"|مصادقة"
            r"|توثيق"
            r"|تشفير"
            r"|صلاحيات"
            r"|التحكم\s+بالوصول"
            r"|خصوصية"
            r"|كلمة\s+المرور"
        ),
    ),

    RequirementType.INTEGRATION: (
        re.compile(
            r"\bintegrat(?:e|es|ed|ion)\b"
            r"|\bapi\b"
            r"|\bexternal\s+interface\b"
            r"|\bintegration\s+interface\b"
            r"|\binterface\s+with\b"
            r"|\bwebhook\b"
            r"|\bexternal\s+system\b"
            r"|\bthird[- ]party\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"تكامل"
            r"|ربط"
            r"|واجهة\s+برمجة"
            r"|نظام\s+خارجي"
            r"|أنظمة\s+خارجية"
            r"|انظمة\s+خارجية"
        ),
    ),

    RequirementType.REPORTING: (
        re.compile(
            r"\breport(?:s|ing)?\b"
            r"|\bdashboard\b"
            r"|\banalytics\b"
            r"|\bstatistics\b"
            r"|\bkpi\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"تقرير"
            r"|تقارير"
            r"|لوحة\s+معلومات"
            r"|لوحة\s+قيادة"
            r"|تحليلات"
            r"|إحصائيات"
            r"|احصائيات"
        ),
    ),

    RequirementType.DATA: (
        re.compile(
            r"\bdata\b"
            r"|\bdatabase\b"
            r"|\brecords?\b"
            r"|\bdata\s+retention\b"
            r"|\bdata\s+quality\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"بيانات"
            r"|قاعدة\s+بيانات"
            r"|سجل"
            r"|سجلات"
            r"|جودة\s+البيانات"
            r"|الاحتفاظ\s+بالبيانات"
        ),
    ),

    RequirementType.NON_FUNCTIONAL: (
        re.compile(
            r"\bperformance\b"
            r"|\bavailability\b"
            r"|\bscalab(?:le|ility)\b"
            r"|\breliability\b"
            r"|\blatency\b"
            r"|\bresponse\s+time\b"
            r"|\bthroughput\b"
            r"|\buptime\b"
            r"|\bconcurrent\s+users?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"أداء"
            r"|اداء"
            r"|توافر"
            r"|قابلية\s+التوسع"
            r"|موثوقية"
            r"|زمن\s+الاستجابة"
            r"|زمن\s+الاستجابه"
            r"|مستخدمين\s+متزامنين"
        ),
    ),

    RequirementType.BUSINESS: (
        re.compile(
            r"\bbusiness\s+requirement\b"
            r"|\bbusiness\s+objective\b"
            r"|\bbusiness\s+goal\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"متطلب\s+أعمال"
            r"|متطلبات\s+الأعمال"
            r"|هدف\s+الأعمال"
            r"|هدف\s+العمل"
        ),
    ),
}


_TYPE_PRECEDENCE = (
    RequirementType.COMPLIANCE,
    RequirementType.SECURITY,
    RequirementType.INTEGRATION,
    RequirementType.REPORTING,
    RequirementType.DATA,
    RequirementType.NON_FUNCTIONAL,
    RequirementType.BUSINESS,
)


def _required_text(
    value: str,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} must be a string."
        )

    value = value.strip()

    if not value:
        raise ValueError(
            f"{field_name} must not be blank."
        )

    return value


def _sha256(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def _matches_any(
    text: str,
    patterns: tuple[
        re.Pattern[str],
        ...
    ],
) -> bool:
    return any(
        pattern.search(text)
        for pattern in patterns
    )


def _clean_unit(
    value: str,
) -> str:
    value = _LEADING_LIST_MARKER_RE.sub(
        "",
        value,
    )

    return " ".join(
        value.split()
    ).strip()


@dataclass(frozen=True)
class ParsedRequirementsSource:
    """Normalized source units used for deterministic extraction."""

    request_id: str

    normalized_text: str

    units: tuple[
        str,
        ...
    ]

    version: str = (
        REQUIREMENT_EXTRACTION_VERSION
    )

    parser_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        request_id = _required_text(
            self.request_id,
            "request_id",
        )

        normalized_text = _required_text(
            self.normalized_text,
            "normalized_text",
        )

        try:
            units = tuple(
                _required_text(
                    value,
                    "unit",
                )
                for value
                in self.units
            )
        except TypeError as exc:
            raise TypeError(
                "units must be iterable."
            ) from exc

        if not units:
            raise ValueError(
                "At least one source unit "
                "is required."
            )

        payload = "\x1f".join(
            (
                self.version,
                request_id,
                _sha256(
                    normalized_text
                ),
                ",".join(
                    _sha256(
                        unit
                    )
                    for unit
                    in units
                ),
            )
        )

        object.__setattr__(
            self,
            "request_id",
            request_id,
        )

        object.__setattr__(
            self,
            "normalized_text",
            normalized_text,
        )

        object.__setattr__(
            self,
            "units",
            units,
        )

        object.__setattr__(
            self,
            "parser_id",
            _sha256(payload),
        )

    def to_dict(
        self,
        *,
        include_text: bool = False,
    ) -> dict[str, object]:
        if not isinstance(
            include_text,
            bool,
        ):
            raise TypeError(
                "include_text must be boolean."
            )

        data: dict[
            str,
            object,
        ] = {
            "version": self.version,
            "parser_id": self.parser_id,
            "request_id": self.request_id,
            "normalized_text_sha256": (
                _sha256(
                    self.normalized_text
                )
            ),
            "unit_count": len(
                self.units
            ),
            "unit_hashes": [
                _sha256(
                    unit
                )
                for unit
                in self.units
            ],
        }

        if include_text:
            data[
                "normalized_text"
            ] = self.normalized_text

            data[
                "units"
            ] = list(
                self.units
            )

        return data


@dataclass(frozen=True)
class RequirementExtractionResult:
    """Privacy-safe deterministic requirement extraction result."""

    request_id: str

    project_id: str

    trace_id: str

    parser_id: str

    requirements: tuple[
        RequirementItem,
        ...
    ]

    algorithm: str = (
        REQUIREMENT_EXTRACTION_ALGORITHM
    )

    version: str = (
        REQUIREMENT_EXTRACTION_VERSION
    )

    extraction_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        request_id = _required_text(
            self.request_id,
            "request_id",
        )

        project_id = _required_text(
            self.project_id,
            "project_id",
        )

        trace_id = _required_text(
            self.trace_id,
            "trace_id",
        )

        parser_id = _required_text(
            self.parser_id,
            "parser_id",
        )

        algorithm = _required_text(
            self.algorithm,
            "algorithm",
        )

        try:
            requirements = tuple(
                self.requirements
            )
        except TypeError as exc:
            raise TypeError(
                "requirements must be iterable."
            ) from exc

        for requirement in requirements:
            if not isinstance(
                requirement,
                RequirementItem,
            ):
                raise TypeError(
                    "requirements must contain only "
                    "RequirementItem values."
                )

            if (
                requirement.request_id
                != request_id
            ):
                raise ValueError(
                    "Requirement request ID mismatch."
                )

        sequences = tuple(
            requirement.sequence
            for requirement
            in requirements
        )

        if sequences != tuple(
            range(
                len(requirements)
            )
        ):
            raise ValueError(
                "Requirement sequences must be "
                "contiguous from zero."
            )

        requirement_ids = tuple(
            requirement.requirement_id
            for requirement
            in requirements
        )

        if (
            len(
                set(
                    requirement_ids
                )
            )
            != len(
                requirement_ids
            )
        ):
            raise ValueError(
                "Requirements must be unique."
            )

        payload = "\x1f".join(
            (
                self.version,
                request_id,
                project_id,
                trace_id,
                parser_id,
                algorithm,
                ",".join(
                    requirement_ids
                ),
            )
        )

        object.__setattr__(
            self,
            "request_id",
            request_id,
        )

        object.__setattr__(
            self,
            "project_id",
            project_id,
        )

        object.__setattr__(
            self,
            "trace_id",
            trace_id,
        )

        object.__setattr__(
            self,
            "parser_id",
            parser_id,
        )

        object.__setattr__(
            self,
            "requirements",
            requirements,
        )

        object.__setattr__(
            self,
            "algorithm",
            algorithm,
        )

        object.__setattr__(
            self,
            "extraction_id",
            _sha256(payload),
        )

    @property
    def requirement_count(
        self,
    ) -> int:
        return len(
            self.requirements
        )

    @property
    def must_count(
        self,
    ) -> int:
        return sum(
            requirement.priority
            is RequirementPriority.MUST
            for requirement
            in self.requirements
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "version": self.version,
            "extraction_id": (
                self.extraction_id
            ),
            "request_id": (
                self.request_id
            ),
            "project_id": (
                self.project_id
            ),
            "trace_id": (
                self.trace_id
            ),
            "parser_id": (
                self.parser_id
            ),
            "algorithm": (
                self.algorithm
            ),
            "requirement_count": (
                self.requirement_count
            ),
            "must_count": (
                self.must_count
            ),
            "requirements": [
                requirement.to_dict()
                for requirement
                in self.requirements
            ],
        }


def parse_requirements_source(
    request: RequirementsRequest,
) -> ParsedRequirementsSource:
    """Normalize a requirements source into deterministic extraction units."""

    if not isinstance(
        request,
        RequirementsRequest,
    ):
        raise TypeError(
            "request must be a "
            "RequirementsRequest."
        )

    normalized = normalize_ingested_text(
        request.source_text
    )

    if not normalized:
        raise ValueError(
            "Requirements source produced "
            "no normalized text."
        )

    raw_units = _SENTENCE_SPLIT_RE.split(
        normalized
    )

    units = tuple(
        cleaned
        for value in raw_units
        if (
            cleaned := _clean_unit(
                value
            )
        )
    )

    if not units:
        units = (
            _clean_unit(
                normalized
            ),
        )

    return ParsedRequirementsSource(
        request_id=(
            request.request_id
        ),
        normalized_text=normalized,
        units=units,
    )


def detect_requirement_priority(
    text: str,
) -> (
    RequirementPriority
    | None
):
    """Return MoSCoW priority when a deterministic requirement cue exists."""

    text = _required_text(
        text,
        "text",
    )

    for priority in (
        RequirementPriority.WONT,
        RequirementPriority.MUST,
        RequirementPriority.SHOULD,
        RequirementPriority.COULD,
    ):
        if _matches_any(
            text,
            _PRIORITY_PATTERNS[
                priority
            ],
        ):
            return priority

    return None


def classify_requirement_type(
    text: str,
) -> RequirementType:
    """Classify one detected requirement using deterministic precedence."""

    text = _required_text(
        text,
        "text",
    )

    for requirement_type in (
        _TYPE_PRECEDENCE
    ):
        if _matches_any(
            text,
            _TYPE_PATTERNS[
                requirement_type
            ],
        ):
            return requirement_type

    return RequirementType.FUNCTIONAL


def extract_requirements(
    request: RequirementsRequest,
    *,
    parsed: (
        ParsedRequirementsSource
        | None
    ) = None,
) -> RequirementExtractionResult:
    """Extract explicit requirements from bilingual source text."""

    if not isinstance(
        request,
        RequirementsRequest,
    ):
        raise TypeError(
            "request must be a "
            "RequirementsRequest."
        )

    if parsed is None:
        parsed = parse_requirements_source(
            request
        )

    if not isinstance(
        parsed,
        ParsedRequirementsSource,
    ):
        raise TypeError(
            "parsed must be a "
            "ParsedRequirementsSource."
        )

    if (
        parsed.request_id
        != request.request_id
    ):
        raise ValueError(
            "Parsed source request ID "
            "does not match request."
        )

    candidates = []
    seen_statements = set()

    for unit in parsed.units:
        priority = (
            detect_requirement_priority(
                unit
            )
        )

        if priority is None:
            continue

        statement_hash = _sha256(
            unit
        )

        if statement_hash in seen_statements:
            continue

        seen_statements.add(
            statement_hash
        )

        requirement_type = (
            classify_requirement_type(
                unit
            )
        )

        candidates.append(
            (
                requirement_type,
                priority,
                unit,
            )
        )

    source_references = (
        (
            request.source_reference,
        )
        if request.source_reference
        else ()
    )

    requirements = tuple(
        RequirementItem(
            request_id=(
                request.request_id
            ),
            sequence=index,
            requirement_type=(
                requirement_type
            ),
            priority=priority,
            status=(
                RequirementStatus.DRAFT
            ),
            statement=statement,
            source_references=(
                source_references
            ),
        )
        for index, (
            requirement_type,
            priority,
            statement,
        )
        in enumerate(
            candidates
        )
    )

    return RequirementExtractionResult(
        request_id=request.request_id,
        project_id=request.project_id,
        trace_id=request.trace_id,
        parser_id=parsed.parser_id,
        requirements=requirements,
    )


def build_extracted_brd(
    request: RequirementsRequest,
    extraction: RequirementExtractionResult,
    *,
    title: str | None = None,
) -> BusinessRequirementsDocument:
    """Build a draft BRD directly from deterministic extraction."""

    if not isinstance(
        request,
        RequirementsRequest,
    ):
        raise TypeError(
            "request must be a "
            "RequirementsRequest."
        )

    if not isinstance(
        extraction,
        RequirementExtractionResult,
    ):
        raise TypeError(
            "extraction must be a "
            "RequirementExtractionResult."
        )

    if (
        extraction.request_id
        != request.request_id
    ):
        raise ValueError(
            "Extraction request ID "
            "does not match request."
        )

    if (
        extraction.project_id
        != request.project_id
    ):
        raise ValueError(
            "Extraction project ID "
            "does not match request."
        )

    if (
        extraction.trace_id
        != request.trace_id
    ):
        raise ValueError(
            "Extraction trace ID "
            "does not match request."
        )

    if title is None:
        title = (
            request.document_title
            or "Business Requirements Document"
        )

    title = _required_text(
        title,
        "title",
    )

    return BusinessRequirementsDocument(
        request_id=(
            request.request_id
        ),
        project_id=(
            request.project_id
        ),
        trace_id=(
            request.trace_id
        ),
        title=title,
        requirements=(
            extraction.requirements
        ),
    )
