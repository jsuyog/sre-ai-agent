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
    reason: str | None = None     # e.g. "CrashLoopBackOff", "OOMKilled"
    message: str | None = None
    exit_code: int | None = None


@dataclass
class PodContext:
    namespace: str
    pod: str
    phase: str                    # Pending | Running | Succeeded | Failed | Unknown
    node: str | None
    containers: list[ContainerState] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    logs: dict[str, str] = field(default_factory=dict)          # container -> current logs
    previous_logs: dict[str, str] = field(default_factory=dict)  # container -> logs from last run
    errors: list[str] = field(default_factory=list)              # non-fatal collection errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Collector:
    """Thin wrapper over the k8s Python client.

    Uses your local kubeconfig when running outside the cluster,
    and the in-cluster ServiceAccount when running inside.
    """

    LOG_TAIL_LINES = 200  # per container per run

    def __init__(self) -> None:
        try:
            config.load_incluster_config()
            log.info("Loaded in-cluster kubeconfig")
        except config.ConfigException:
            config.load_kube_config()
            log.info("Loaded local kubeconfig")

        self.core = client.CoreV1Api()

    # ---- public API ----

    def collect(self, namespace: str, pod: str) -> PodContext:
        ctx = PodContext(namespace=namespace, pod=pod, phase="Unknown", node=None)

        self._fill_pod_status(ctx)
        self._fill_events(ctx)
        self._fill_logs(ctx)

        return ctx

    # ---- internals ----

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

            ctx.containers.append(
                ContainerState(
                    name=cs.name,
                    ready=cs.ready,
                    restart_count=cs.restart_count,
                    state=state,
                    reason=reason,
                    message=message,
                    exit_code=exit_code,
                )
            )

    def _fill_events(self, ctx: PodContext) -> None:
        """Events tell us what the scheduler/kubelet has been doing to this pod:
        image pulls, container kills, back-off restarts. Often the smoking gun."""
        try:
            selector = f"involvedObject.name={ctx.pod}"
            evs = self.core.list_namespaced_event(
                namespace=ctx.namespace, field_selector=selector
            )
        except ApiException as e:
            ctx.errors.append(f"list_events failed: {e.status} {e.reason}")
            return

        # Newest first, cap to avoid flooding the prompt.
        items = sorted(
            evs.items,
            key=lambda e: e.last_timestamp or e.event_time or e.metadata.creation_timestamp,
            reverse=True,
        )[:25]

        for e in items:
            ts = e.last_timestamp or e.event_time or e.metadata.creation_timestamp
            ctx.events.append({
                "time": ts.isoformat() if ts else None,
                "type": e.type,             # Normal | Warning
                "reason": e.reason,          # BackOff, Failed, Pulled, ...
                "message": e.message,
                "count": e.count,
            })

    def _fill_logs(self, ctx: PodContext) -> None:
        """Fetch logs for each container. For crashlooping containers the *previous*
        run's logs usually contain the real error — the current run may have never
        actually started."""
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
                _preload_content=False,   # get the raw HTTPResponse
            )
            data = raw.data  # bytes
            text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else str(data)
            target[container] = text or "(empty)"
        except ApiException as e:
            if previous and e.status == 400:
                return
            ctx.errors.append(
                f"logs failed (container={container}, previous={previous}): "
                f"{e.status} {e.reason}"
            )


# ---- CLI for manual testing ----
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
