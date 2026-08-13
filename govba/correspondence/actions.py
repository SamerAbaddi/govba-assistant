"""Deterministic action and commitment extraction for GovBA-GAR."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from enum import Enum

from govba.correspondence.classification import (
    ParsedCorrespondence,
    parse_correspondence,
)
from govba.correspondence.contract import (
    CorrespondenceRequest,
)


CORRESPONDENCE_ACTION_VERSION = (
    "govba-correspondence-action-v1"
)

CORRESPONDENCE_ACTION_ALGORITHM = (
    "deterministic-action-extraction-v1"
)


class CorrespondenceActionKind(
    str,
    Enum,
):
    SUBMIT = "submit"
    PROVIDE = "provide"
    REVIEW = "review"
    APPROVE = "approve"
    RESPOND = "respond"
    ATTEND = "attend"
    PAY = "pay"
    COMPLETE = "complete"
    NOTIFY = "notify"
    OTHER = "other"


class CorrespondenceActionOwner(
    str,
    Enum,
):
    SENDER = "sender"
    RECIPIENT = "recipient"


class CorrespondenceActionNature(
    str,
    Enum,
):
    REQUESTED = "requested"
    REQUIRED = "required"
    COMMITTED = "committed"


_ACTION_PATTERNS = {
    CorrespondenceActionKind.SUBMIT: (
        re.compile(
            r"\bsubmit(?:ted|ting)?\b"
            r"|\bfile\b"
            r"|\blodge\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"تقديم"
            r"|يقدم"
            r"|قدموا"
            r"|إيداع"
            r"|ايداع"
        ),
    ),

    CorrespondenceActionKind.PROVIDE: (
        re.compile(
            r"\bprovide\b"
            r"|\bsupply\b"
            r"|\bfurnish\b"
            r"|\bsend\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"تزويد"
            r"|زوّد"
            r"|زود"
            r"|إرسال"
            r"|ارسال"
        ),
    ),

    CorrespondenceActionKind.REVIEW: (
        re.compile(
            r"\breview\b"
            r"|\bexamine\b"
            r"|\bassess\b"
            r"|\bcheck\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"مراجع(?:ة|ته)"
            r"|دراس(?:ة|ته)"
            r"|فحص"
            r"|تدقيق"
        ),
    ),

    CorrespondenceActionKind.APPROVE: (
        re.compile(
            r"\bapprove\b"
            r"|\bapproval\b"
            r"|\bauthori[sz]e\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"موافق(?:ة|ات)"
            r"|اعتماد"
            r"|إقرار"
            r"|اقرار"
        ),
    ),

    CorrespondenceActionKind.RESPOND: (
        re.compile(
            r"\brespond\b"
            r"|\breply\b"
            r"|\banswer\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"الرد"
            r"|ردكم"
            r"|الإجابة"
            r"|الاجابة"
        ),
    ),

    CorrespondenceActionKind.ATTEND: (
        re.compile(
            r"\battend\b"
            r"|\bparticipate\b"
            r"|\bjoin\s+the\s+meeting\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"حضور"
            r"|المشاركة"
            r"|شاركوا"
        ),
    ),

    CorrespondenceActionKind.PAY: (
        re.compile(
            r"\bpay\b"
            r"|\bpayment\b"
            r"|\bsettle\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"دفع"
            r"|سداد"
            r"|تسديد"
        ),
    ),

    CorrespondenceActionKind.COMPLETE: (
        re.compile(
            r"\bcomplete\b"
            r"|\bfinali[sz]e\b"
            r"|\bfinish\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"استكمال"
            r"|إكمال"
            r"|اكمال"
            r"|إنهاء"
            r"|انهاء"
        ),
    ),

    CorrespondenceActionKind.NOTIFY: (
        re.compile(
            r"\bnotify\b"
            r"|\binform\b"
            r"|\badvise\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"إبلاغ"
            r"|ابلاغ"
            r"|إخطار"
            r"|اخطار"
            r"|إعلام"
            r"|اعلام"
        ),
    ),
}


_SENDER_COMMITMENT_PATTERNS = (
    re.compile(
        r"\b(?:we|i)\s+will\b"
        r"|\b(?:we|i)\s+shall\b"
        r"|\b(?:we|i)\s+undertake\s+to\b"
        r"|\b(?:we|i)\s+commit\s+to\b"
        r"|\b(?:we|i)\s+agree\s+to\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"سنقوم"
        r"|سوف\s+نقوم"
        r"|سأقوم"
        r"|سوف\s+أقوم"
        r"|نتعهد"
        r"|أتعهد"
        r"|نلتزم"
        r"|ألتزم"
    ),
)


_RECIPIENT_REQUIRED_PATTERNS = (
    re.compile(
        r"\byou\s+must\b"
        r"|\byou\s+shall\b"
        r"|\byou\s+are\s+required\s+to\b"
        r"|\baction\s+required\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"يجب\s+عليكم"
        r"|يتعين\s+عليكم"
        r"|يلزمكم"
        r"|مطلوب\s+منكم"
    ),
)


_RECIPIENT_REQUEST_PATTERNS = (
    re.compile(
        r"\bplease\b"
        r"|\bkindly\b"
        r"|\bwe\s+request\s+you\s+to\b"
        r"|\bcould\s+you\b"
        r"|\bwould\s+you\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"يرجى"
        r"|نرجو"
        r"|نطلب\s+منكم"
        r"|نلتمس\s+منكم"
    ),
)


_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?؟؛])\s+|\n+"
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


def _sha256(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode(
            "utf-8"
        )
    ).hexdigest()


def _matches_any(
    text: str,
    patterns: tuple[
        re.Pattern[str],
        ...
    ],
) -> bool:
    return any(
        pattern.search(
            text
        )
        for pattern
        in patterns
    )


def _detect_nature(
    sentence: str,
) -> tuple[
    CorrespondenceActionOwner,
    CorrespondenceActionNature,
] | None:
    if _matches_any(
        sentence,
        _SENDER_COMMITMENT_PATTERNS,
    ):
        return (
            CorrespondenceActionOwner.SENDER,
            CorrespondenceActionNature.COMMITTED,
        )

    if _matches_any(
        sentence,
        _RECIPIENT_REQUIRED_PATTERNS,
    ):
        return (
            CorrespondenceActionOwner.RECIPIENT,
            CorrespondenceActionNature.REQUIRED,
        )

    if _matches_any(
        sentence,
        _RECIPIENT_REQUEST_PATTERNS,
    ):
        return (
            CorrespondenceActionOwner.RECIPIENT,
            CorrespondenceActionNature.REQUESTED,
        )

    return None


def _detect_action_kinds(
    sentence: str,
) -> tuple[
    CorrespondenceActionKind,
    ...
]:
    kinds = []

    for kind in CorrespondenceActionKind:
        if (
            kind
            is CorrespondenceActionKind.OTHER
        ):
            continue

        patterns = _ACTION_PATTERNS[
            kind
        ]

        if _matches_any(
            sentence,
            patterns,
        ):
            kinds.append(
                kind
            )

    if not kinds:
        return (
            CorrespondenceActionKind.OTHER,
        )

    return tuple(
        kinds
    )


@dataclass(frozen=True)
class CorrespondenceAction:
    """One deterministic extracted action or commitment."""

    request_id: str

    sequence: int

    kind: CorrespondenceActionKind

    owner: CorrespondenceActionOwner

    nature: CorrespondenceActionNature

    confidence: float

    action_text: str

    reason_code: str

    version: str = (
        CORRESPONDENCE_ACTION_VERSION
    )

    source_sha256: str = field(
        init=False
    )

    action_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        request_id = _required_text(
            self.request_id,
            "request_id",
        )

        if (
            isinstance(
                self.sequence,
                bool,
            )
            or not isinstance(
                self.sequence,
                int,
            )
            or self.sequence < 0
        ):
            raise ValueError(
                "sequence must be a "
                "non-negative integer."
            )

        if not isinstance(
            self.kind,
            CorrespondenceActionKind,
        ):
            raise TypeError(
                "kind must be a "
                "CorrespondenceActionKind."
            )

        if not isinstance(
            self.owner,
            CorrespondenceActionOwner,
        ):
            raise TypeError(
                "owner must be a "
                "CorrespondenceActionOwner."
            )

        if not isinstance(
            self.nature,
            CorrespondenceActionNature,
        ):
            raise TypeError(
                "nature must be a "
                "CorrespondenceActionNature."
            )

        if (
            isinstance(
                self.confidence,
                bool,
            )
            or not isinstance(
                self.confidence,
                (int, float),
            )
            or not math.isfinite(
                self.confidence
            )
            or not (
                0.0
                <= self.confidence
                <= 1.0
            )
        ):
            raise ValueError(
                "confidence must be between "
                "0 and 1."
            )

        action_text = _required_text(
            self.action_text,
            "action_text",
        )

        reason_code = _required_text(
            self.reason_code,
            "reason_code",
        )

        source_hash = _sha256(
            action_text
        )

        payload = "\x1f".join(
            (
                self.version,
                request_id,
                str(
                    self.sequence
                ),
                self.kind.value,
                self.owner.value,
                self.nature.value,
                f"{float(self.confidence):.12f}",
                source_hash,
                reason_code,
            )
        )

        object.__setattr__(
            self,
            "request_id",
            request_id,
        )

        object.__setattr__(
            self,
            "confidence",
            float(
                self.confidence
            ),
        )

        object.__setattr__(
            self,
            "action_text",
            action_text,
        )

        object.__setattr__(
            self,
            "reason_code",
            reason_code,
        )

        object.__setattr__(
            self,
            "source_sha256",
            source_hash,
        )

        object.__setattr__(
            self,
            "action_id",
            _sha256(
                payload
            ),
        )

    @property
    def is_commitment(
        self,
    ) -> bool:
        return (
            self.nature
            is CorrespondenceActionNature.COMMITTED
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
            "version": (
                self.version
            ),
            "action_id": (
                self.action_id
            ),
            "request_id": (
                self.request_id
            ),
            "sequence": (
                self.sequence
            ),
            "kind": (
                self.kind.value
            ),
            "owner": (
                self.owner.value
            ),
            "nature": (
                self.nature.value
            ),
            "is_commitment": (
                self.is_commitment
            ),
            "confidence": (
                self.confidence
            ),
            "source_sha256": (
                self.source_sha256
            ),
            "reason_code": (
                self.reason_code
            ),
        }

        if include_text:
            data[
                "action_text"
            ] = self.action_text

        return data


@dataclass(frozen=True)
class CorrespondenceActionExtractionResult:
    """Privacy-safe collection of extracted actions."""

    request_id: str

    parser_id: str

    actions: tuple[
        CorrespondenceAction,
        ...
    ]

    algorithm: str = (
        CORRESPONDENCE_ACTION_ALGORITHM
    )

    version: str = (
        CORRESPONDENCE_ACTION_VERSION
    )

    extraction_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        request_id = _required_text(
            self.request_id,
            "request_id",
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
            actions = tuple(
                self.actions
            )
        except TypeError as exc:
            raise TypeError(
                "actions must be iterable."
            ) from exc

        for action in actions:
            if not isinstance(
                action,
                CorrespondenceAction,
            ):
                raise TypeError(
                    "actions must contain only "
                    "CorrespondenceAction values."
                )

            if (
                action.request_id
                != request_id
            ):
                raise ValueError(
                    "Action request ID does not "
                    "match extraction request ID."
                )

        sequences = tuple(
            action.sequence
            for action
            in actions
        )

        if sequences != tuple(
            range(
                len(actions)
            )
        ):
            raise ValueError(
                "Action sequences must be "
                "contiguous from zero."
            )

        action_ids = tuple(
            action.action_id
            for action in actions
        )

        if (
            len(set(action_ids))
            != len(action_ids)
        ):
            raise ValueError(
                "Actions must be unique."
            )

        payload = "\x1f".join(
            (
                self.version,
                request_id,
                parser_id,
                algorithm,
                ",".join(
                    action_ids
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
            "parser_id",
            parser_id,
        )

        object.__setattr__(
            self,
            "actions",
            actions,
        )

        object.__setattr__(
            self,
            "algorithm",
            algorithm,
        )

        object.__setattr__(
            self,
            "extraction_id",
            _sha256(
                payload
            ),
        )

    @property
    def action_count(
        self,
    ) -> int:
        return len(
            self.actions
        )

    @property
    def commitment_count(
        self,
    ) -> int:
        return sum(
            action.is_commitment
            for action
            in self.actions
        )

    @property
    def requested_count(
        self,
    ) -> int:
        return sum(
            action.nature
            is CorrespondenceActionNature.REQUESTED
            for action
            in self.actions
        )

    @property
    def required_count(
        self,
    ) -> int:
        return sum(
            action.nature
            is CorrespondenceActionNature.REQUIRED
            for action
            in self.actions
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        """Serialize without correspondence/action text."""

        return {
            "version": (
                self.version
            ),
            "extraction_id": (
                self.extraction_id
            ),
            "request_id": (
                self.request_id
            ),
            "parser_id": (
                self.parser_id
            ),
            "algorithm": (
                self.algorithm
            ),
            "action_count": (
                self.action_count
            ),
            "commitment_count": (
                self.commitment_count
            ),
            "requested_count": (
                self.requested_count
            ),
            "required_count": (
                self.required_count
            ),
            "actions": [
                action.to_dict()
                for action
                in self.actions
            ],
        }


def extract_correspondence_actions(
    request: CorrespondenceRequest,
    *,
    parsed: ParsedCorrespondence | None = None,
) -> CorrespondenceActionExtractionResult:
    """Extract requested, required and committed actions deterministically."""

    if not isinstance(
        request,
        CorrespondenceRequest,
    ):
        raise TypeError(
            "request must be a "
            "CorrespondenceRequest."
        )

    if parsed is None:
        parsed = parse_correspondence(
            request
        )

    if not isinstance(
        parsed,
        ParsedCorrespondence,
    ):
        raise TypeError(
            "parsed must be a "
            "ParsedCorrespondence."
        )

    if (
        parsed.request_id
        != request.request_id
    ):
        raise ValueError(
            "Parsed request ID does not "
            "match correspondence request."
        )

    extracted = []

    for paragraph in parsed.paragraphs:
        sentences = tuple(
            value.strip()
            for value in _SENTENCE_SPLIT_RE.split(
                paragraph
            )
            if value.strip()
        )

        for sentence in sentences:
            nature_result = _detect_nature(
                sentence
            )

            if nature_result is None:
                continue

            owner, nature = (
                nature_result
            )

            kinds = _detect_action_kinds(
                sentence
            )

            for kind in kinds:
                if (
                    nature
                    is CorrespondenceActionNature.COMMITTED
                ):
                    confidence = 0.90
                    reason_code = (
                        "explicit_sender_commitment"
                    )

                elif (
                    nature
                    is CorrespondenceActionNature.REQUIRED
                ):
                    confidence = 0.88
                    reason_code = (
                        "explicit_recipient_requirement"
                    )

                else:
                    confidence = 0.82
                    reason_code = (
                        "explicit_recipient_request"
                    )

                if (
                    kind
                    is CorrespondenceActionKind.OTHER
                ):
                    confidence = max(
                        0.50,
                        confidence - 0.12,
                    )

                extracted.append(
                    (
                        kind,
                        owner,
                        nature,
                        confidence,
                        sentence,
                        reason_code,
                    )
                )

    actions = tuple(
        CorrespondenceAction(
            request_id=(
                request.request_id
            ),
            sequence=index,
            kind=kind,
            owner=owner,
            nature=nature,
            confidence=confidence,
            action_text=sentence,
            reason_code=reason_code,
        )
        for index, (
            kind,
            owner,
            nature,
            confidence,
            sentence,
            reason_code,
        )
        in enumerate(
            extracted
        )
    )

    return CorrespondenceActionExtractionResult(
        request_id=(
            request.request_id
        ),
        parser_id=(
            parsed.parser_id
        ),
        actions=actions,
    )
