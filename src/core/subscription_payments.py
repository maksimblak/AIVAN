from __future__ import annotations

import re
import time
from dataclasses import dataclass

PAYLOAD_PREFIX = "sub"


@dataclass(frozen=True)
class SubscriptionPaymentPayload:
    """Structured view of a subscription payment payload."""

    raw: str
    plan_id: str | None
    method: str | None
    user_id: int | None
    timestamp: int | None
    is_legacy: bool = False


_PAYLOAD_SPLIT_RE = re.compile(r"[:]")
_PLAN_ID_RE = re.compile(r"^[a-z0-9_]+$")
_METHOD_RE = re.compile(r"^[a-z]+$")


class SubscriptionPayloadError(ValueError):
    """Raised when payload cannot be parsed."""


def build_subscription_payload(
    plan_id: str, method: str, user_id: int, *, timestamp: int | None = None
) -> str:
    """Create payload string for new subscription purchase."""

    if ":" in plan_id:
        raise SubscriptionPayloadError("plan_id must not contain colon")
    if ":" in method:
        raise SubscriptionPayloadError("method must not contain colon")
    if not _PLAN_ID_RE.match(plan_id):
        raise SubscriptionPayloadError("plan_id contains unsupported characters")
    if not _METHOD_RE.match(method):
        raise SubscriptionPayloadError("method contains unsupported characters")
    ts = int(timestamp or time.time())
    if ts <= 0:
        raise SubscriptionPayloadError("timestamp must be positive")
    return f"{PAYLOAD_PREFIX}:{plan_id}:{method}:{int(user_id)}:{ts}"


def parse_subscription_payload(payload: str) -> SubscriptionPaymentPayload:
    """Parse payload string into structured form (supports legacy format)."""

    if not payload:
        raise SubscriptionPayloadError("payload is empty")
    parts = payload.split(":")
    if not parts or parts[0] != PAYLOAD_PREFIX:
        raise SubscriptionPayloadError("payload has unsupported prefix")

    if len(parts) == 4:
        # Legacy format: sub:{method}:{user_id}:{timestamp}
        _, method, user_id_raw, timestamp_raw = parts
        plan_id = None
        is_legacy = True
    elif len(parts) >= 5:
        _, plan_id, method, user_id_raw, timestamp_raw, *rest = parts
        if rest:
            # If there are extra fields, treat as invalid to avoid ambiguity
            raise SubscriptionPayloadError("payload has extra segments")
        is_legacy = False
    else:
        raise SubscriptionPayloadError("payload has insufficient segments")

    if len(parts) == 4:
        plan_id = None
    else:
        if not plan_id or not _PLAN_ID_RE.match(plan_id):
            raise SubscriptionPayloadError("invalid plan_id")

    if not method or not _METHOD_RE.match(method):
        raise SubscriptionPayloadError("invalid method")

    try:
        user_id = int(user_id_raw)
    except (TypeError, ValueError):
        raise SubscriptionPayloadError("invalid user_id") from None
    if user_id <= 0:
        raise SubscriptionPayloadError("user_id must be positive")

    try:
        timestamp = int(timestamp_raw)
    except (TypeError, ValueError):
        raise SubscriptionPayloadError("invalid timestamp") from None
    if timestamp <= 0:
        raise SubscriptionPayloadError("timestamp must be positive")

    return SubscriptionPaymentPayload(
        raw=payload,
        plan_id=plan_id,
        method=method,
        user_id=user_id,
        timestamp=timestamp,
        is_legacy=is_legacy,
    )
