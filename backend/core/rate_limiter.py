"""Thread-safe sliding-window rate limiter.

Extracted from backend/api/routes.py so both entrypoints (FastAPI and the
Gradio app) share one implementation instead of each having their own copy --
same reasoning as guardrails.py: two independently-maintained copies of the
same logic drift, and only one of them was ever protecting the thing users
actually hit (the deployed Gradio app had none at all -- see CHANGES.md
2026-09-02 for the review finding).
"""

import time
from collections import defaultdict
from threading import Lock
from typing import Optional


class RateLimitExceeded(Exception):
    """Raised by check() when a key has exceeded its requests-per-minute budget."""
    def __init__(self, key: str, rpm: int):
        self.key = key
        self.rpm = rpm
        super().__init__(f"Rate limit exceeded: maximum {rpm} requests per minute.")


class SlidingWindowRateLimiter:
    """Per-key (API key, IP, or session id) sliding window rate limiter."""

    def __init__(self, requests_per_minute: int = 20):
        self.rpm = requests_per_minute
        self.requests = defaultdict(list)
        self.lock = Lock()

    def check(self, key: str) -> None:
        """Raises RateLimitExceeded if `key` is over budget; otherwise records this call."""
        now = time.time()
        window_start = now - 60.0
        with self.lock:
            timestamps = [t for t in self.requests[key] if t > window_start]
            if len(timestamps) >= self.rpm:
                raise RateLimitExceeded(key, self.rpm)
            timestamps.append(now)
            self.requests[key] = timestamps

    def is_allowed(self, key: str) -> bool:
        """Non-raising variant for callers that want to branch instead of catch."""
        try:
            self.check(key)
            return True
        except RateLimitExceeded:
            return False
