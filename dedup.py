"""Incident dedup for the SRE agent.

Multiple pods with the same alert+reason in the same namespace = one incident.
Same incident firing repeatedly = still one incident.

In-memory only; agent restart clears state. Fine for MVP.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import Lock

log = logging.getLogger(__name__)


@dataclass
class Incident:
    fingerprint: str
    first_seen: float
    last_rca_at: float
    pods: set[str] = field(default_factory=set)
    slack_thread_ts: str | None = None
    rca_count: int = 0
    reminder_count: int = 0


class Deduper:
    """Tracks incidents in memory and decides what action each alert warrants.

    Windows (all in seconds):
    - `full_rca_window`: within this since last RCA → suppress entirely
    - `stale_after`: forget the incident entirely (next occurrence is new)
    """

    def __init__(
        self,
        full_rca_window: float = 30 * 60,   # 30 min: skip re-RCA
        stale_after: float = 6 * 3600,       # 6h without a hit → forget
    ) -> None:
        self.full_rca_window = full_rca_window
        self.stale_after = stale_after
        self._incidents: dict[str, Incident] = {}
        self._lock = Lock()

    @staticmethod
    def fingerprint(alertname: str, labels: dict[str, str], reason: str | None) -> str:
        """Same alert + same namespace + same failure reason = same incident.
        Deliberately excludes `pod` — many pods with one cause is ONE incident."""
        parts = [
            alertname,
            labels.get("namespace", ""),
            labels.get("container", ""),
            reason or labels.get("reason") or "",
        ]
        return "|".join(parts)

    def decide(self, fingerprint: str, pod: str) -> tuple[str, Incident]:
        """Returns (action, incident). Action is one of:
        - "rca"       → run collector + LLM + post full RCA
        - "reminder"  → short "still firing" note in thread; no LLM call
        - "suppress"  → do nothing (log only)
        """
        now = time.time()
        with self._lock:
            self._evict_stale(now)
            inc = self._incidents.get(fingerprint)

            if inc is None:
                inc = Incident(
                    fingerprint=fingerprint,
                    first_seen=now,
                    last_rca_at=now,
                    rca_count=1,
                )
                inc.pods.add(pod)
                self._incidents[fingerprint] = inc
                return "rca", inc

            inc.pods.add(pod)
            if now - inc.last_rca_at < self.full_rca_window:
                return "suppress", inc

            inc.last_rca_at = now
            inc.reminder_count += 1
            return "reminder", inc

    def resolve(self, fingerprint: str, pod: str | None = None) -> tuple[str, Incident | None]:
        """A single pod's alert cleared.

        The incident is only dropped once EVERY tracked pod has resolved —
        otherwise one flapping pod would reset suppression for pods that are
        still broken.

        Returns (action, incident) where action is:
        - "closed"   → all pods recovered; incident removed (post ✅)
        - "partial"  → this pod recovered, others still firing (stay quiet)
        - "unknown"  → we weren't tracking this incident
        """
        with self._lock:
            inc = self._incidents.get(fingerprint)
            if inc is None:
                return "unknown", None

            if pod:
                inc.pods.discard(pod)

            if not inc.pods:
                del self._incidents[fingerprint]
                return "closed", inc

            return "partial", inc

    def record_slack_thread(self, fingerprint: str, ts: str) -> None:
        """Save the thread ts so reminders/resolutions reply in the same thread."""
        with self._lock:
            inc = self._incidents.get(fingerprint)
            if inc and not inc.slack_thread_ts:
                inc.slack_thread_ts = ts

    def _evict_stale(self, now: float) -> None:
        stale = [f for f, i in self._incidents.items()
                 if now - i.last_rca_at > self.stale_after]
        for f in stale:
            log.info("Evicting stale incident %s", f)
            del self._incidents[f]
