import pytest

from src.core.subscription_payments import (
    build_subscription_payload,
    parse_subscription_payload,
    SubscriptionPayloadError,
)


def test_build_subscription_payload_valid():
    payload = build_subscription_payload("base_1m", "rub", 123, timestamp=1700000000)
    assert payload == "sub:base_1m:rub:123:1700000000"


def test_parse_new_payload_format():
    parsed = parse_subscription_payload("sub:premium_1m:stars:42:1700000000")
    assert parsed.plan_id == "premium_1m"
    assert parsed.method == "stars"
    assert parsed.user_id == 42
    assert parsed.timestamp == 1700000000
    assert parsed.is_legacy is False


def test_parse_legacy_payload_format():
    parsed = parse_subscription_payload("sub:crypto:5:1700000000")
    assert parsed.plan_id is None
    assert parsed.method == "crypto"
    assert parsed.user_id == 5
    assert parsed.is_legacy is True


def test_parse_payload_with_extra_segments_raises():
    with pytest.raises(SubscriptionPayloadError):
        parse_subscription_payload("sub:base_1m:rub:1:2:extra")


def test_build_payload_rejects_colon():
    with pytest.raises(SubscriptionPayloadError):
        build_subscription_payload("base:1m", "rub", 1)


def test_parse_invalid_prefix():
    with pytest.raises(SubscriptionPayloadError):
        parse_subscription_payload("legacy:rub:1:1700000000")
