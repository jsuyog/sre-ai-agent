"""SRE Agent — Alertmanager webhook receiver.

Flow per webhook:
  1. Handle 'resolved' alerts — only close an incident once EVERY tracked pod
     has recovered, so one flapping pod doesn't reset suppression.
  2. Group firing alerts by incident fingerprint (alert + ns + container + reason).
  3. For each group, decide with the deduper: rca / reminder / suppress.
     - rca:      collect → Claude → Slack (top-level message)
     - reminder: short "still firing" post, threaded under the original
     - suppress: log and move on
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Request

from collector import Collector, PodContext
from dedup import Deduper, Incident
from llm import LLM
from slack import Slack, SlackError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("agent")

app = FastAPI(title="SRE Agent")
collector = Collector()
slack = Slack()
llm = LLM()
deduper = Deduper()


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/alert")
async def alert(req: Request, background: BackgroundTasks) -> dict[str, Any]:
    payload = await req.json()
    raw = payload.get("alerts", [])
    firing = [a for a in raw if a.get("status") == "firing"]
    resolved = [a for a in raw if a.get("status") == "resolved"]
    log.info("Received webhook: %d firing, %d resolved", len(firing), len(resolved))

    # ---- Handle resolutions first ----
    # Only close an incident once every tracked pod has recovered; a single
    # flapping pod must not reset suppression for pods still broken.
    for a in resolved:
        labels = a.get("labels", {}) or {}
        fp = Deduper.fingerprint(
            alertname=labels.get("alertname", "UnknownAlert"),
            labels=labels,
            reason=labels.get("reason"),
        )
        action, inc = deduper.resolve(fp, labels.get("pod"))
        log.info("resolve fp=%s pod=%s action=%s remaining=%s",
                 fp, labels.get("pod"), action,
                 len(inc.pods) if inc else 0)

        if action == "closed" and inc and inc.slack_thread_ts:
            try:
                slack.post_text(
                    f"✅ Resolved — all pods for {labels.get('namespace')} "
                    f"{labels.get('alertname')} have recovered.",
                    thread_ts=inc.slack_thread_ts,
                )
            except SlackError as e:
                log.error("Slack resolve post failed: %s", e)

    # ---- Group firing alerts by fingerprint ----
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for a in firing:
        labels = a.get("labels", {}) or {}
        fp = Deduper.fingerprint(
            alertname=labels.get("alertname", "UnknownAlert"),
            labels=labels,
            reason=labels.get("reason"),
        )
        groups[fp].append(a)

    accepted = 0
    for fp, group in groups.items():
        rep = group[0]
        pod = (rep.get("labels", {}) or {}).get("pod")
        if not pod:
            log.warning("Alert missing 'pod' label, skipping: %s", rep.get("labels"))
            continue

        action, inc = deduper.decide(fp, pod)
        related = sorted(
            {(a.get("labels", {}) or {}).get("pod")
             for a in group[1:]
             if (a.get("labels") or {}).get("pod")}
        )
        log.info("fp=%s action=%s pods=%d", fp, action, len(inc.pods))

        if action == "rca":
            background.add_task(_do_rca, fp, rep, related)
            accepted += 1
        elif action == "reminder":
            background.add_task(_do_reminder, fp, rep, inc)
        # suppress → nothing

    return {"accepted": accepted, "groups": len(groups), "resolved": len(resolved)}


# ------------- worker tasks -------------

def _do_rca(fingerprint: str, alert_obj: dict[str, Any], related_pods: list[str]) -> None:
    labels = alert_obj.get("labels", {}) or {}
    annotations = alert_obj.get("annotations", {}) or {}
    alertname = labels.get("alertname", "UnknownAlert")
    namespace, pod = labels["namespace"], labels["pod"]
    container = labels.get("container")
    severity = labels.get("severity", "unknown")

    # ---- collect ----
    try:
        ctx: PodContext = collector.collect(namespace, pod)
    except Exception as e:
        log.exception("Collector failed: %s", e)
        _post_error(alertname, namespace, pod, f"Collector failed: {e}")
        return

    # ---- LLM ----
    try:
        rca = llm.analyze(
            alertname=alertname,
            ctx=ctx,
            annotations=annotations,
            related_pods=related_pods,
        )
        log.info(
            "LLM ok: model=%s in=%d out=%d",
            rca.model, rca.input_tokens, rca.output_tokens,
        )
        rca_text = rca.text
    except Exception as e:
        log.exception("LLM call failed: %s", e)
        rca_text = f"*LLM call failed* — falling back to raw context.\n```\n{e}\n```"

    # ---- Slack ----
    fields = {
        "Alert": alertname,
        "Namespace": namespace,
        "Pod": pod,
        "Severity": severity,
    }
    if container:
        fields["Container"] = container
    if ctx.containers:
        fields["Restarts"] = str(ctx.containers[0].restart_count)
        if ctx.containers[0].reason:
            fields["Reason"] = ctx.containers[0].reason
    if related_pods:
        fields["Also affects"] = f"{len(related_pods)} more pod(s)"

    summary = annotations.get("summary") or f"{alertname} on {namespace}/{pod}"

    try:
        ts = slack.post_rca(
            title=f"🚨 {alertname} — {namespace}/{pod}",
            summary=summary,
            fields=fields,
            rca_body=rca_text,
        )
        deduper.record_slack_thread(fingerprint, ts)
        log.info("Posted RCA to Slack ts=%s", ts)
    except SlackError as e:
        log.error("Slack post failed: %s", e)


def _do_reminder(fingerprint: str, alert_obj: dict[str, Any], inc: Incident) -> None:
    labels = alert_obj.get("labels", {}) or {}
    ns, pod = labels.get("namespace"), labels.get("pod")
    try:
        slack.post_text(
            f"🔁 Still firing (reminder #{inc.reminder_count}) — {ns}/{pod}. "
            f"Affected pods so far: {len(inc.pods)}. No new analysis; see original RCA above.",
            thread_ts=inc.slack_thread_ts,
        )
    except SlackError as e:
        log.error("Slack reminder failed: %s", e)


def _post_error(alertname: str, ns: str, pod: str, err: str) -> None:
    try:
        slack.post_rca(
            title=f"⚠️ Agent error — {ns}/{pod}",
            summary=f"Failed to investigate {alertname}",
            fields={"Namespace": ns, "Pod": pod, "Alert": alertname},
            rca_body=f"```\n{err}\n```",
        )
    except SlackError:
        pass
