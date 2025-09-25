from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class UserSession:
    user_id: int
    questions_count: int = 0
    total_response_time: float = 0.0
    last_question_time: datetime | None = None
    created_at: datetime = field(default_factory=datetime.now)
    pending_feedback_request_id: int | None = None

    def add_question_stats(self, response_time: float) -> None:
        self.questions_count += 1
        self.total_response_time += response_time
        self.last_question_time = datetime.now()


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
