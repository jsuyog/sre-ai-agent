"""Collects diagnostic context for a Kubernetes pod.

Given a namespace and pod name, returns pod status, events, and logs —
the raw material the LLM will reason over.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any

from kubernetes import client, config
from kubernetes.client.rest import ApiException

log = logging.getLogger(__name__)


@dataclass
class ContainerState:
    name: str
    ready: bool
    restart_count: int
    state: str                    # "running" | "waiting" | "terminated"
    reason: str | None = None     # current-state reason, e.g. "CrashLoopBackOff"
    message: str | None = None
    exit_code: int | None = None
    # Previous termination — where OOMKilled / Error / exit codes actually live.
    last_reason: str | None = None      # e.g. "OOMKilled", "Error"
    last_exit_code: int | None = None
    last_signal: int | None = None


@dataclass
class PodContext:
    namespace: str
    pod: str
    phase: str
    node: str | None
    containers: list[ContainerState] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    logs: dict[str, str] = field(default_factory=dict)
    previous_logs: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Collector:
    """Thin wrapper over the k8s Python client.

    Uses your local kubeconfig when running outside the cluster,
    and the in-cluster ServiceAccount when running inside.
    """

    LOG_TAIL_LINES = 200

    def __init__(self) -> None:
        try:
            config.load_incluster_config()
            log.info("Loaded in-cluster kubeconfig")
        except config.ConfigException:
            config.load_kube_config()
            log.info("Loaded local kubeconfig")
        self.core = client.CoreV1Api()

    def collect(self, namespace: str, pod: str) -> PodContext:
        ctx = PodContext(namespace=namespace, pod=pod, phase="Unknown", node=None)
        self._fill_pod_status(ctx)
        self._fill_events(ctx)
        self._fill_logs(ctx)
        return ctx

    def _fill_pod_status(self, ctx: PodContext) -> None:
        try:
            p = self.core.read_namespaced_pod(name=ctx.pod, namespace=ctx.namespace)
        except ApiException as e:
            ctx.errors.append(f"read_pod failed: {e.status} {e.reason}")
            return

        ctx.phase = p.status.phase or "Unknown"
        ctx.node = p.spec.node_name

        for cs in (p.status.container_statuses or []):
            state, reason, message, exit_code = "unknown", None, None, None
            st = cs.state
            if st.running:
                state = "running"
            elif st.waiting:
                state = "waiting"
                reason = st.waiting.reason
                message = st.waiting.message
            elif st.terminated:
                state = "terminated"
                reason = st.terminated.reason
                message = st.terminated.message
                exit_code = st.terminated.exit_code

            # Previous termination — the real home of OOMKilled/exit codes.
            last_reason = last_exit_code = last_signal = None
            last = cs.last_state
            if last and last.terminated:
                last_reason = last.terminated.reason
                last_exit_code = last.terminated.exit_code
                last_signal = last.terminated.signal

            ctx.containers.append(
                ContainerState(
                    name=cs.name,
                    ready=cs.ready,
                    restart_count=cs.restart_count,
                    state=state,
                    reason=reason,
                    message=message,
                    exit_code=exit_code,
                    last_reason=last_reason,
                    last_exit_code=last_exit_code,
                    last_signal=last_signal,
                )
            )

    def _fill_events(self, ctx: PodContext) -> None:
        try:
            selector = f"involvedObject.name={ctx.pod}"
            evs = self.core.list_namespaced_event(
                namespace=ctx.namespace, field_selector=selector
            )
        except ApiException as e:
            ctx.errors.append(f"list_events failed: {e.status} {e.reason}")
            return

        items = sorted(
            evs.items,
            key=lambda e: e.last_timestamp or e.event_time or e.metadata.creation_timestamp,
            reverse=True,
        )[:25]

        for e in items:
            ts = e.last_timestamp or e.event_time or e.metadata.creation_timestamp
            ctx.events.append({
                "time": ts.isoformat() if ts else None,
                "type": e.type,
                "reason": e.reason,
                "message": e.message,
                "count": e.count,
            })

    def _fill_logs(self, ctx: PodContext) -> None:
        for c in ctx.containers:
            self._safe_logs(ctx, c.name, previous=False)
            if c.restart_count > 0:
                self._safe_logs(ctx, c.name, previous=True)

    def _safe_logs(self, ctx: PodContext, container: str, *, previous: bool) -> None:
        target = ctx.previous_logs if previous else ctx.logs
        try:
            raw = self.core.read_namespaced_pod_log(
                name=ctx.pod,
                namespace=ctx.namespace,
                container=container,
                previous=previous,
                tail_lines=self.LOG_TAIL_LINES,
                _preload_content=False,
            )
            data = raw.data
            text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else str(data)
            target[container] = text or "(empty)"
        except ApiException as e:
            if previous and e.status == 400:
                return
            ctx.errors.append(
                f"logs failed (container={container}, previous={previous}): "
                f"{e.status} {e.reason}"
            )


if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if len(sys.argv) != 3:
        print("Usage: python collector.py <namespace> <pod>", file=sys.stderr)
        sys.exit(2)
    ns, pod = sys.argv[1], sys.argv[2]
    ctx = Collector().collect(ns, pod)
    print(json.dumps(ctx.to_dict(), indent=2, default=str))
