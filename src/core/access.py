from __future__ import annotations

from dataclasses import dataclass

from .db_advanced import DatabaseAdvanced


@dataclass
class AccessDecision:
    allowed: bool
    is_admin: bool = False
    has_subscription: bool = False
    subscription_until: int | None = None
    subscription_plan: str | None = None
    subscription_requests_remaining: int | None = None
    trial_used: int | None = None
    trial_remaining: int | None = None


class AccessService:
    """Encapsulates access control: admin, subscription, and trial consumption."""

    def __init__(self, *, db: DatabaseAdvanced, trial_limit: int, admin_ids: set[int]):
        self._db = db
        self._trial_limit = trial_limit
        self._admin_ids = set(admin_ids)

    async def check_and_consume(self, user_id: int) -> AccessDecision:
        """Ensure user exists; if admin/subscriber -> allowed. Otherwise consume one trial.

        Returns AccessDecision with details for UI formatting.
        """
        user = await self._db.ensure_user(
            user_id, default_trial=self._trial_limit, is_admin=user_id in self._admin_ids
        )
        is_admin = bool(user.is_admin) or (user_id in self._admin_ids)
        if is_admin:
            return AccessDecision(allowed=True, is_admin=True)

        has_subscription = await self._db.has_active_subscription(user_id)
        subscription_plan = getattr(user, "subscription_plan", None)
        subscription_balance_raw = getattr(user, "subscription_requests_balance", None)
        subscription_balance: int | None = None
        if subscription_balance_raw is not None:
            try:
                subscription_balance = max(0, int(subscription_balance_raw))
            except (TypeError, ValueError):
                subscription_balance = None
        subscription_until = int(user.subscription_until) if user.subscription_until else None

        if has_subscription:
            if subscription_plan and subscription_balance is not None:
                if subscription_balance <= 0:
                    return AccessDecision(
                        allowed=False,
                        has_subscription=True,
                        subscription_until=subscription_until,
                        subscription_plan=subscription_plan,
                        subscription_requests_remaining=0,
                    )
                return AccessDecision(
                    allowed=True,
                    has_subscription=True,
                    subscription_until=subscription_until,
                    subscription_plan=subscription_plan,
                    subscription_requests_remaining=subscription_balance,
                )
            return AccessDecision(
                allowed=True,
                has_subscription=True,
                subscription_until=subscription_until,
            )

        # Try to decrement trial
        trial_before = int(user.trial_remaining)
        if await self._db.decrement_trial(user_id):
            # Re-fetch remaining to be precise
            user_after = await self._db.get_user(user_id)
            trial_after = (
                int(user_after.trial_remaining) if user_after else max(0, trial_before - 1)
            )
            used = max(0, self._trial_limit - trial_after)
            return AccessDecision(
                allowed=True,
                trial_used=used,
                trial_remaining=trial_after,
            )

        # No access
        return AccessDecision(allowed=False)
