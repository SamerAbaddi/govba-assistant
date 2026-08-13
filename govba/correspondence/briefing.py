"""Governed correspondence briefing for GovBA-GAR."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from enum import Enum

from govba.correspondence.actions import (
    CorrespondenceAction,
    CorrespondenceActionExtractionResult,
)
from govba.correspondence.contract import (
    CorrespondenceClassification,
    CorrespondenceIntent,
    CorrespondencePriority,
    CorrespondenceRequest,
)
from govba.correspondence.priority import (
    CorrespondenceDeadline,
    CorrespondenceDeadlineResult,
    CorrespondencePriorityAssessment,
)


CORRESPONDENCE_BRIEFING_VERSION = (
    "govba-correspondence-briefing-v1"
)


class CorrespondenceBriefingDecision(
    str,
    Enum,
):
    READY = "ready"
    REVIEW = "review"
    ABSTAIN = "abstain"


class CorrespondenceBriefingReason(
    str,
    Enum,
):
    VERIFIED = "verified"
    NO_ACTION_REQUIRED = (
        "no_action_required"
    )
    ACTION_EXPECTED_BUT_NOT_EXTRACTED = (
        "action_expected_but_not_extracted"
    )
    LOW_ACTION_CONFIDENCE = (
        "low_action_confidence"
    )
    LOW_CLASSIFICATION_CONFIDENCE = (
        "low_classification_confidence"
    )
    UNKNOWN_INTENT = "unknown_intent"
    URGENT_CORRESPONDENCE = (
        "urgent_correspondence"
    )
    OVERDUE_DEADLINE = (
        "overdue_deadline"
    )
    HIGH_PRIORITY_REVIEW = (
        "high_priority_review"
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


def _rate(
    value: float,
    field_name: str,
) -> float:
    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            (int, float),
        )
        or not math.isfinite(
            value
        )
        or not (
            0.0 <= value <= 1.0
        )
    ):
        raise ValueError(
            f"{field_name} must be "
            "between 0 and 1."
        )

    return float(
        value
    )


def _sha256(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode(
            "utf-8"
        )
    ).hexdigest()


@dataclass(frozen=True)
class CorrespondenceBriefingPolicy:
    """Explicit governance policy for correspondence briefings."""

    min_classification_confidence: float = 0.50

    min_action_confidence: float = 0.70

    review_urgent: bool = True

    review_high_priority: bool = False

    version: str = (
        CORRESPONDENCE_BRIEFING_VERSION
    )

    policy_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        classification_threshold = (
            _rate(
                self.min_classification_confidence,
                "min_classification_confidence",
            )
        )

        action_threshold = _rate(
            self.min_action_confidence,
            "min_action_confidence",
        )

        if not isinstance(
            self.review_urgent,
            bool,
        ):
            raise TypeError(
                "review_urgent must be boolean."
            )

        if not isinstance(
            self.review_high_priority,
            bool,
        ):
            raise TypeError(
                "review_high_priority "
                "must be boolean."
            )

        payload = "\x1f".join(
            (
                self.version,
                f"{classification_threshold:.12f}",
                f"{action_threshold:.12f}",
                str(
                    self.review_urgent
                ).lower(),
                str(
                    self.review_high_priority
                ).lower(),
            )
        )

        object.__setattr__(
            self,
            "min_classification_confidence",
            classification_threshold,
        )

        object.__setattr__(
            self,
            "min_action_confidence",
            action_threshold,
        )

        object.__setattr__(
            self,
            "policy_id",
            _sha256(
                payload
            ),
        )


@dataclass(frozen=True)
class CorrespondenceBriefingAction:
    """Privacy-safe action reference for the briefing."""

    action_id: str
    kind: str
    owner: str
    nature: str
    confidence: float
    source_sha256: str

    version: str = (
        CORRESPONDENCE_BRIEFING_VERSION
    )

    def __post_init__(self) -> None:
        for field_name in (
            "action_id",
            "kind",
            "owner",
            "nature",
            "source_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(
                    getattr(
                        self,
                        field_name,
                    ),
                    field_name,
                ),
            )

        object.__setattr__(
            self,
            "confidence",
            _rate(
                self.confidence,
                "confidence",
            ),
        )

    @classmethod
    def from_action(
        cls,
        action: CorrespondenceAction,
    ) -> "CorrespondenceBriefingAction":
        if not isinstance(
            action,
            CorrespondenceAction,
        ):
            raise TypeError(
                "action must be a "
                "CorrespondenceAction."
            )

        return cls(
            action_id=action.action_id,
            kind=action.kind.value,
            owner=action.owner.value,
            nature=action.nature.value,
            confidence=action.confidence,
            source_sha256=(
                action.source_sha256
            ),
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "version": self.version,
            "action_id": self.action_id,
            "kind": self.kind,
            "owner": self.owner,
            "nature": self.nature,
            "confidence": self.confidence,
            "source_sha256": (
                self.source_sha256
            ),
        }


@dataclass(frozen=True)
class CorrespondenceBriefingDeadline:
    """Privacy-safe deadline reference."""

    deadline_id: str
    kind: str
    deadline_at: str
    confidence: float
    source_sha256: str

    version: str = (
        CORRESPONDENCE_BRIEFING_VERSION
    )

    def __post_init__(self) -> None:
        for field_name in (
            "deadline_id",
            "kind",
            "deadline_at",
            "source_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(
                    getattr(
                        self,
                        field_name,
                    ),
                    field_name,
                ),
            )

        object.__setattr__(
            self,
            "confidence",
            _rate(
                self.confidence,
                "confidence",
            ),
        )

    @classmethod
    def from_deadline(
        cls,
        deadline: CorrespondenceDeadline,
    ) -> "CorrespondenceBriefingDeadline":
        if not isinstance(
            deadline,
            CorrespondenceDeadline,
        ):
            raise TypeError(
                "deadline must be a "
                "CorrespondenceDeadline."
            )

        return cls(
            deadline_id=deadline.deadline_id,
            kind=deadline.kind.value,
            deadline_at=(
                deadline.deadline_at
                .isoformat()
            ),
            confidence=deadline.confidence,
            source_sha256=(
                deadline.source_sha256
            ),
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "version": self.version,
            "deadline_id": (
                self.deadline_id
            ),
            "kind": self.kind,
            "deadline_at": (
                self.deadline_at
            ),
            "confidence": (
                self.confidence
            ),
            "source_sha256": (
                self.source_sha256
            ),
        }


@dataclass(frozen=True)
class GovernedCorrespondenceBriefing:
    """Governed structured briefing without raw correspondence text."""

    request_id: str

    trace_id: str

    classification_id: str

    priority_assessment_id: str

    action_extraction_id: str

    deadline_extraction_id: str

    intent: CorrespondenceIntent

    priority: CorrespondencePriority

    requires_action: bool

    classification_confidence: float

    decision: (
        CorrespondenceBriefingDecision
    )

    reasons: tuple[
        CorrespondenceBriefingReason,
        ...
    ]

    actions: tuple[
        CorrespondenceBriefingAction,
        ...
    ]

    deadlines: tuple[
        CorrespondenceBriefingDeadline,
        ...
    ]

    overdue: bool

    policy_id: str

    version: str = (
        CORRESPONDENCE_BRIEFING_VERSION
    )

    briefing_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        text_fields = (
            "request_id",
            "trace_id",
            "classification_id",
            "priority_assessment_id",
            "action_extraction_id",
            "deadline_extraction_id",
            "policy_id",
        )

        for field_name in text_fields:
            object.__setattr__(
                self,
                field_name,
                _required_text(
                    getattr(
                        self,
                        field_name,
                    ),
                    field_name,
                ),
            )

        if not isinstance(
            self.intent,
            CorrespondenceIntent,
        ):
            raise TypeError(
                "intent must be a "
                "CorrespondenceIntent."
            )

        if not isinstance(
            self.priority,
            CorrespondencePriority,
        ):
            raise TypeError(
                "priority must be a "
                "CorrespondencePriority."
            )

        if not isinstance(
            self.requires_action,
            bool,
        ):
            raise TypeError(
                "requires_action must be boolean."
            )

        confidence = _rate(
            self.classification_confidence,
            "classification_confidence",
        )

        if not isinstance(
            self.decision,
            CorrespondenceBriefingDecision,
        ):
            raise TypeError(
                "decision must be a "
                "CorrespondenceBriefingDecision."
            )

        try:
            reasons = tuple(
                self.reasons
            )
        except TypeError as exc:
            raise TypeError(
                "reasons must be iterable."
            ) from exc

        if not reasons:
            raise ValueError(
                "At least one briefing reason "
                "is required."
            )

        for reason in reasons:
            if not isinstance(
                reason,
                CorrespondenceBriefingReason,
            ):
                raise TypeError(
                    "reasons must contain only "
                    "CorrespondenceBriefingReason values."
                )

        if (
            len(
                set(
                    reasons
                )
            )
            != len(
                reasons
            )
        ):
            raise ValueError(
                "Briefing reasons must be unique."
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
                CorrespondenceBriefingAction,
            ):
                raise TypeError(
                    "actions must contain only "
                    "CorrespondenceBriefingAction values."
                )

        try:
            deadlines = tuple(
                self.deadlines
            )
        except TypeError as exc:
            raise TypeError(
                "deadlines must be iterable."
            ) from exc

        for deadline in deadlines:
            if not isinstance(
                deadline,
                CorrespondenceBriefingDeadline,
            ):
                raise TypeError(
                    "deadlines must contain only "
                    "CorrespondenceBriefingDeadline values."
                )

        if not isinstance(
            self.overdue,
            bool,
        ):
            raise TypeError(
                "overdue must be boolean."
            )

        payload = "\x1f".join(
            (
                self.version,
                self.request_id,
                self.trace_id,
                self.classification_id,
                self.priority_assessment_id,
                self.action_extraction_id,
                self.deadline_extraction_id,
                self.intent.value,
                self.priority.value,
                str(
                    self.requires_action
                ).lower(),
                f"{confidence:.12f}",
                self.decision.value,
                ",".join(
                    reason.value
                    for reason in reasons
                ),
                ",".join(
                    action.action_id
                    for action in actions
                ),
                ",".join(
                    deadline.deadline_id
                    for deadline in deadlines
                ),
                str(
                    self.overdue
                ).lower(),
                self.policy_id,
            )
        )

        object.__setattr__(
            self,
            "classification_confidence",
            confidence,
        )

        object.__setattr__(
            self,
            "reasons",
            reasons,
        )

        object.__setattr__(
            self,
            "actions",
            actions,
        )

        object.__setattr__(
            self,
            "deadlines",
            deadlines,
        )

        object.__setattr__(
            self,
            "briefing_id",
            _sha256(
                payload
            ),
        )

    @property
    def ready(
        self,
    ) -> bool:
        return (
            self.decision
            is CorrespondenceBriefingDecision.READY
        )

    @property
    def requires_review(
        self,
    ) -> bool:
        return (
            self.decision
            is CorrespondenceBriefingDecision.REVIEW
        )

    @property
    def abstained(
        self,
    ) -> bool:
        return (
            self.decision
            is CorrespondenceBriefingDecision.ABSTAIN
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "version": self.version,
            "briefing_id": (
                self.briefing_id
            ),
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "classification_id": (
                self.classification_id
            ),
            "priority_assessment_id": (
                self.priority_assessment_id
            ),
            "action_extraction_id": (
                self.action_extraction_id
            ),
            "deadline_extraction_id": (
                self.deadline_extraction_id
            ),
            "intent": self.intent.value,
            "priority": self.priority.value,
            "requires_action": (
                self.requires_action
            ),
            "classification_confidence": (
                self.classification_confidence
            ),
            "decision": (
                self.decision.value
            ),
            "ready": self.ready,
            "requires_review": (
                self.requires_review
            ),
            "abstained": self.abstained,
            "overdue": self.overdue,
            "policy_id": self.policy_id,
            "reasons": [
                reason.value
                for reason in self.reasons
            ],
            "action_count": len(
                self.actions
            ),
            "deadline_count": len(
                self.deadlines
            ),
            "actions": [
                action.to_dict()
                for action in self.actions
            ],
            "deadlines": [
                deadline.to_dict()
                for deadline in self.deadlines
            ],
        }


def build_governed_correspondence_briefing(
    request: CorrespondenceRequest,
    classification: CorrespondenceClassification,
    actions: CorrespondenceActionExtractionResult,
    deadlines: CorrespondenceDeadlineResult,
    priority: CorrespondencePriorityAssessment,
    *,
    policy: (
        CorrespondenceBriefingPolicy
        | None
    ) = None,
) -> GovernedCorrespondenceBriefing:
    """Build a governed correspondence briefing."""

    if not isinstance(
        request,
        CorrespondenceRequest,
    ):
        raise TypeError(
            "request must be a "
            "CorrespondenceRequest."
        )

    if not isinstance(
        classification,
        CorrespondenceClassification,
    ):
        raise TypeError(
            "classification must be a "
            "CorrespondenceClassification."
        )

    if not isinstance(
        actions,
        CorrespondenceActionExtractionResult,
    ):
        raise TypeError(
            "actions must be a "
            "CorrespondenceActionExtractionResult."
        )

    if not isinstance(
        deadlines,
        CorrespondenceDeadlineResult,
    ):
        raise TypeError(
            "deadlines must be a "
            "CorrespondenceDeadlineResult."
        )

    if not isinstance(
        priority,
        CorrespondencePriorityAssessment,
    ):
        raise TypeError(
            "priority must be a "
            "CorrespondencePriorityAssessment."
        )

    if policy is None:
        policy = (
            CorrespondenceBriefingPolicy()
        )

    if not isinstance(
        policy,
        CorrespondenceBriefingPolicy,
    ):
        raise TypeError(
            "policy must be a "
            "CorrespondenceBriefingPolicy."
        )

    request_id = request.request_id

    if classification.request_id != request_id:
        raise ValueError(
            "Classification request mismatch."
        )

    if actions.request_id != request_id:
        raise ValueError(
            "Action extraction request mismatch."
        )

    if deadlines.request_id != request_id:
        raise ValueError(
            "Deadline extraction request mismatch."
        )

    if priority.request_id != request_id:
        raise ValueError(
            "Priority assessment request mismatch."
        )

    if (
        priority.classification_id
        != classification.classification_id
    ):
        raise ValueError(
            "Priority assessment classification "
            "does not match."
        )

    if (
        priority.deadline_extraction_id
        != deadlines.extraction_id
    ):
        raise ValueError(
            "Priority assessment deadline "
            "extraction does not match."
        )

    briefing_actions = tuple(
        CorrespondenceBriefingAction
        .from_action(
            action
        )
        for action in actions.actions
    )

    briefing_deadlines = tuple(
        CorrespondenceBriefingDeadline
        .from_deadline(
            deadline
        )
        for deadline in deadlines.deadlines
    )

    reasons = []

    if (
        classification.intent
        is CorrespondenceIntent.UNKNOWN
    ):
        decision = (
            CorrespondenceBriefingDecision
            .ABSTAIN
        )

        reasons.append(
            CorrespondenceBriefingReason
            .UNKNOWN_INTENT
        )

    elif (
        classification.confidence
        < policy.min_classification_confidence
    ):
        decision = (
            CorrespondenceBriefingDecision
            .ABSTAIN
        )

        reasons.append(
            CorrespondenceBriefingReason
            .LOW_CLASSIFICATION_CONFIDENCE
        )

    else:
        review_reasons = []

        if (
            classification.requires_action
            and not briefing_actions
        ):
            review_reasons.append(
                CorrespondenceBriefingReason
                .ACTION_EXPECTED_BUT_NOT_EXTRACTED
            )

        if any(
            action.confidence
            < policy.min_action_confidence
            for action in briefing_actions
        ):
            review_reasons.append(
                CorrespondenceBriefingReason
                .LOW_ACTION_CONFIDENCE
            )

        if priority.overdue:
            review_reasons.append(
                CorrespondenceBriefingReason
                .OVERDUE_DEADLINE
            )

        if (
            policy.review_urgent
            and priority.priority
            is CorrespondencePriority.URGENT
        ):
            review_reasons.append(
                CorrespondenceBriefingReason
                .URGENT_CORRESPONDENCE
            )

        if (
            policy.review_high_priority
            and priority.priority
            is CorrespondencePriority.HIGH
        ):
            review_reasons.append(
                CorrespondenceBriefingReason
                .HIGH_PRIORITY_REVIEW
            )

        if review_reasons:
            decision = (
                CorrespondenceBriefingDecision
                .REVIEW
            )

            reasons.extend(
                review_reasons
            )

        else:
            decision = (
                CorrespondenceBriefingDecision
                .READY
            )

            if classification.requires_action:
                reasons.append(
                    CorrespondenceBriefingReason
                    .VERIFIED
                )
            else:
                reasons.append(
                    CorrespondenceBriefingReason
                    .NO_ACTION_REQUIRED
                )

    reasons = tuple(
        dict.fromkeys(
            reasons
        )
    )

    return GovernedCorrespondenceBriefing(
        request_id=request_id,
        trace_id=request.trace_id,
        classification_id=(
            classification.classification_id
        ),
        priority_assessment_id=(
            priority.assessment_id
        ),
        action_extraction_id=(
            actions.extraction_id
        ),
        deadline_extraction_id=(
            deadlines.extraction_id
        ),
        intent=classification.intent,
        priority=priority.priority,
        requires_action=(
            classification.requires_action
        ),
        classification_confidence=(
            classification.confidence
        ),
        decision=decision,
        reasons=reasons,
        actions=briefing_actions,
        deadlines=briefing_deadlines,
        overdue=priority.overdue,
        policy_id=policy.policy_id,
    )
