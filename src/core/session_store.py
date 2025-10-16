from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Mapping


class GuidedFlowStep(StrEnum):
    """Enumeration of the guided onboarding flow steps."""

    LEGAL_QUESTION = "legal_question"
    PRACTICE_SEARCH = "practice_search"
    DOCUMENT_DRAFT = "document_draft"


@dataclass
class GuidedFlowState:
    """
    Track user onboarding progress through the three core bot features.

    The state keeps the canonical order of steps, what has already been completed
    and the user input gathered at every stage. This allows the bot to:
      * Offer contextual hints (next step vs. already explored);
      * Avoid asking the same clarifying questions again;
      * Resume the flow seamlessly after interruptions.
    """

    step_order: tuple[GuidedFlowStep, ...] = (
        GuidedFlowStep.LEGAL_QUESTION,
        GuidedFlowStep.PRACTICE_SEARCH,
        GuidedFlowStep.DOCUMENT_DRAFT,
    )
    completed_steps: list[GuidedFlowStep] = field(default_factory=list)
    collected_inputs: dict[GuidedFlowStep, dict[str, Any]] = field(default_factory=dict)
    last_prompted_step: GuidedFlowStep | None = None
    started_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def reset(self) -> None:
        """Start the guided flow from scratch."""
        self.completed_steps.clear()
        self.collected_inputs.clear()
        self.last_prompted_step = None
        self.started_at = datetime.now()
        self.updated_at = self.started_at

    def mark_prompted(self, step: GuidedFlowStep) -> None:
        """Remember that the bot already prompted the user for this step."""
        self.last_prompted_step = step
        self.updated_at = datetime.now()

    def upsert_step_payload(
        self,
        step: GuidedFlowStep,
        payload: Mapping[str, Any] | None = None,
        **extras: Any,
    ) -> dict[str, Any]:
        """Store (and merge) user answers for the specific step."""
        if step not in self.step_order:
            raise ValueError(f"Unknown guided flow step: {step!r}")

        merged_payload = self.collected_inputs.setdefault(step, {})
        if payload:
            merged_payload.update(payload)
        if extras:
            merged_payload.update(extras)

        self.updated_at = datetime.now()
        return merged_payload

    def mark_completed(
        self,
        step: GuidedFlowStep,
        *,
        payload: Mapping[str, Any] | None = None,
        **extras: Any,
    ) -> None:
        """
        Mark the step as completed and persist any contextual data provided by the user.
        """
        self.upsert_step_payload(step, payload, **extras)

        if step not in self.completed_steps:
            self.completed_steps.append(step)
        else:
            # Preserve the original order in case of re-submissions.
            idx = self.completed_steps.index(step)
            self.completed_steps[idx] = step

        self.updated_at = datetime.now()

    def get_step_payload(self, step: GuidedFlowStep) -> dict[str, Any] | None:
        """Return user-provided data for a step, if any."""
        return self.collected_inputs.get(step)

    def is_step_completed(self, step: GuidedFlowStep) -> bool:
        """Check whether the user already finished the step."""
        return step in self.completed_steps

    def get_next_step(self) -> GuidedFlowStep | None:
        """Find the first step that has not yet been completed."""
        for step in self.step_order:
            if step not in self.completed_steps:
                return step
        return None

    def is_flow_completed(self) -> bool:
        """Return True when all steps are done."""
        return self.get_next_step() is None

    def progress_snapshot(self) -> dict[str, Any]:
        """
        Provide a serialisable snapshot of the current journey state.

        Intended for logging/debugging or external persistence layers.
        """
        next_step = self.get_next_step()
        return {
            "step_order": [step.value for step in self.step_order],
            "completed_steps": [step.value for step in self.completed_steps],
            "next_step": next_step.value if next_step else None,
            "last_prompted_step": self.last_prompted_step.value if self.last_prompted_step else None,
            "collected_inputs": {
                step.value: data.copy() for step, data in self.collected_inputs.items()
            },
            "started_at": self.started_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class UserSession:
    user_id: int
    questions_count: int = 0
    total_response_time: float = 0.0
    last_question_time: datetime | None = None
    created_at: datetime = field(default_factory=datetime.now)
    pending_feedback_request_id: int | None = None
    voice_tts_enabled: bool = True
    guided_flow: GuidedFlowState = field(default_factory=GuidedFlowState)

    def add_question_stats(self, response_time: float) -> None:
        self.questions_count += 1
        self.total_response_time += response_time
        self.last_question_time = datetime.now()

    def get_guided_flow(self) -> GuidedFlowState:
        """Expose the guided flow state for external orchestration code."""
        return self.guided_flow

    def reset_guided_flow(self) -> GuidedFlowState:
        """Reset the guided flow (useful when the user restarts onboarding)."""
        self.guided_flow.reset()
        return self.guided_flow

    def mark_guided_step_prompted(self, step: GuidedFlowStep) -> None:
        """Record that we already asked the user about the given step."""
        self.guided_flow.mark_prompted(step)

    def update_guided_step_payload(
        self,
        step: GuidedFlowStep,
        payload: Mapping[str, Any] | None = None,
        **extras: Any,
    ) -> dict[str, Any]:
        """Store user answers for the guided step without marking it completed."""
        return self.guided_flow.upsert_step_payload(step, payload, **extras)

    def complete_guided_step(
        self,
        step: GuidedFlowStep,
        payload: Mapping[str, Any] | None = None,
        **extras: Any,
    ) -> None:
        """Mark the step complete and keep the latest payload snapshot."""
        self.guided_flow.mark_completed(step, payload=payload, **extras)


class SessionStore:
    """In-memory session store with TTL and max-size enforcement."""

    def __init__(self, *, max_size: int, ttl_seconds: int) -> None:
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._sessions: dict[int, UserSession] = {}

    def get_or_create(self, user_id: int) -> UserSession:
        self.cleanup()
        sess = self._sessions.get(user_id)
        if sess is None:
            sess = UserSession(user_id=user_id)
            self._sessions[user_id] = sess
        return sess

    def cleanup(self) -> None:
        if not self._sessions:
            return
        now = datetime.now()
        if self._ttl_seconds > 0:
            to_drop = []
            for uid, sess in self._sessions.items():
                last_active = sess.last_question_time or sess.created_at
                if (now - last_active).total_seconds() > self._ttl_seconds:
                    to_drop.append(uid)
            for uid in to_drop:
                self._sessions.pop(uid, None)
        if self._max_size > 0 and len(self._sessions) > self._max_size:
            items_sorted = sorted(
                self._sessions.items(), key=lambda kv: kv[1].last_question_time or kv[1].created_at
            )
            excess = len(self._sessions) - self._max_size
            for i in range(excess):
                uid, _ = items_sorted[i]
                self._sessions.pop(uid, None)
