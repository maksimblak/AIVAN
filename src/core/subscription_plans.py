from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SubscriptionPlan:
    """Represents a purchasable subscription option."""

    plan_id: str
    name: str
    price_rub: int
    duration_days: int
    request_quota: int
    description: str = ""

    @property
    def price_rub_kopeks(self) -> int:
        return int(self.price_rub * 100)


DEFAULT_SUBSCRIPTION_PLANS: tuple[SubscriptionPlan, ...] = (
    SubscriptionPlan(
        plan_id="base_1m",
        name="Базовый",
        price_rub=1499,
        duration_days=30,
        request_quota=60,
        description="Регулярные консультации",
    ),
    SubscriptionPlan(
        plan_id="standard_1m",
        name="Стандарт",
        price_rub=2500,
        duration_days=30,
        request_quota=100,
        description="Расширенный лимит запросов",
    ),
    SubscriptionPlan(
        plan_id="premium_1m",
        name="Премиум",
        price_rub=4000,
        duration_days=30,
        request_quota=200,
        description="Максимум запросов и приоритет",
    ),
)

def get_default_subscription_plans() -> tuple[SubscriptionPlan, ...]:
    """Return default subscription plan catalog."""

    return DEFAULT_SUBSCRIPTION_PLANS


def build_plan_map(plans: Iterable[SubscriptionPlan]) -> dict[str, SubscriptionPlan]:
    """Create a fast lookup dictionary by plan_id."""

    return {plan.plan_id: plan for plan in plans}

