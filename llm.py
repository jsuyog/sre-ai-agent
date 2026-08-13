"""Claude API wrapper — turns collected pod context into an RCA.

Output is deliberately structured so it stays useful even when the hypothesis
is wrong: observations are transcription (near-always correct), the hypothesis
ships with a one-command falsification test, and alternatives keep a bad guess
from dead-ending triage.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from anthropic import Anthropic, APIError

from collector import PodContext

log = logging.getLogger(__name__)

# Guardrails so we don't send absurd prompts (cost + latency + context length)
MAX_LOG_LINES_PER_CONTAINER = 60
MAX_EVENTS = 10


@dataclass
class RCAResult:
    text: str            # Slack-markdown-ready RCA body
    model: str
    input_tokens: int
    output_tokens: int


class LLM:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        # Override with `export ANTHROPIC_MODEL=...` if you want a different model
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
        self.client = Anthropic(api_key=api_key)  # reads ANTHROPIC_API_KEY from env

    def analyze(
        self,
        *,
        alertname: str,
        ctx: PodContext,
        annotations: dict[str, str] | None = None,
        related_pods: list[str] | None = None,
    ) -> RCAResult:
        prompt = _build_prompt(alertname, ctx, annotations or {}, related_pods or [])

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=1200,
                temperature=0,   # reduce run-to-run variance on identical input
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
        except APIError as e:
            log.error("Claude API error: %s", e)
            raise

        text = "".join(b.text for b in resp.content if b.type == "text")
        return RCAResult(
            text=text.strip(),
            model=resp.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )


_SYSTEM_PROMPT = """You are a senior Site Reliability Engineer triaging a Kubernetes alert.

Your output is a FIRST-PASS triage aid for an on-call engineer, not a verdict.
It must stay useful even if your leading hypothesis turns out to be wrong.

Evidence priority:
- Kubernetes events are the primary evidence for WHY a container was killed or
  failed to start. Weigh them above log contents.
- A container's LAST termination reason (e.g. OOMKilled, exit 137) is decisive
  and often the only place a crash cause appears — Kubernetes emits no event for
  OOM kills. Always check it before blaming the application code.
- Distinguish cause from effect. A container killed by a failing liveness probe,
  an OOM kill, or a node eviction often logs a clean, graceful shutdown.
  A graceful exit (SIGTERM/SIGQUIT, exit code 0) is NOT evidence the application
  failed on its own — look to the events for an external cause first.
- If an event shows a probe failing against a port, treat the port the app
  actually binds to as unverified until checked — do not assume either way.

Rules:
- Every claim must trace to a specific event, log line, or number in the input.
- Never invent details. If the evidence is thin, say so and lower your confidence.
- Assume the reader can run kubectl and edit manifests.
- Keep the whole response under ~350 words.

Format as Slack-flavored markdown, exactly these four sections:

*Observations*
3-5 bullets of raw facts only — no interpretation. Quote counts, exit codes,
event reasons, and timestamps as given. This section must be verifiable line by line.

*Most likely cause*
1-3 sentences. Name the specific failure and cite the exact evidence for it.

*Confirm it*
The single best command to prove or disprove the hypothesis above, and what
result would confirm vs. refute it. Then the fix to apply if confirmed.

*If that's not it*
1-2 alternative hypotheses ranked by likelihood, each with the one check that
would test it. This section is mandatory — never omit it, even at high confidence.
"""


def _build_prompt(
    alertname: str,
    ctx: PodContext,
    annotations: dict[str, str],
    related_pods: list[str],
) -> str:
    p: list[str] = [f"# Alert: {alertname}"]
    if annotations.get("summary"):
        p.append(f"Summary: {annotations['summary']}")
    if annotations.get("description"):
        p.append(f"Description: {annotations['description']}")

    p.append("\n# Pod")
    p.append(f"Namespace: {ctx.namespace}")
    p.append(f"Pod: {ctx.pod}")
    p.append(f"Phase: {ctx.phase}")
    p.append(f"Node: {ctx.node}")
    if related_pods:
        p.append(f"Same incident also affects: {', '.join(related_pods)}")

    p.append("\n# Container status")
    for c in ctx.containers:
        line = f"- {c.name}: state={c.state}"
        if c.reason:
            line += f", reason={c.reason}"
        if c.exit_code is not None:
            line += f", exit_code={c.exit_code}"
        line += f", restarts={c.restart_count}"
        if c.last_reason:
            line += f", last_termination={c.last_reason}"
            if c.last_exit_code is not None:
                line += f" (exit {c.last_exit_code})"
        p.append(line)
        if c.message:
            p.append(f"  message: {c.message}")

    # Events come BEFORE logs: kubelet's account of *why* a container was
    # killed is usually the root cause, while logs often show only the *effect*
    # (e.g. a graceful shutdown that looks like a self-inflicted exit).
    if ctx.events:
        p.append("\n# Kubernetes events (newest first) — PRIMARY EVIDENCE")
        for e in ctx.events[:MAX_EVENTS]:
            p.append(
                f"- [{e['type']}/{e['reason']}] ×{e['count']} at {e['time']}: {e['message']}"
            )

    p.append(
        "\n# Log tails — SUPPORTING EVIDENCE ONLY\n"
        "Note: a container terminated by a failing probe, an OOM kill, or an "
        "eviction logs a normal graceful shutdown. Read such logs as the *effect* "
        "of an external kill, not proof the app exited on its own."
    )

    if ctx.previous_logs:
        p.append("\n## Previous run (before last restart)")
        for name, txt in ctx.previous_logs.items():
            p.append(f"### {name}\n```\n{_tail(txt, MAX_LOG_LINES_PER_CONTAINER)}\n```")

    if ctx.logs:
        p.append("\n## Current run")
        for name, txt in ctx.logs.items():
            p.append(f"### {name}\n```\n{_tail(txt, MAX_LOG_LINES_PER_CONTAINER)}\n```")

    if ctx.errors:
        p.append(f"\n# Collector warnings\n{'; '.join(ctx.errors)}")

    return "\n".join(p)


def _tail(text: str, n_lines: int) -> str:
    return "\n".join(text.strip().splitlines()[-n_lines:])


if __name__ == "__main__":
    import sys
    from collector import Collector

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if len(sys.argv) != 3:
        print("Usage: python llm.py <namespace> <pod>", file=sys.stderr)
        sys.exit(2)
    ns, pod = sys.argv[1], sys.argv[2]
    ctx = Collector().collect(ns, pod)
    result = LLM().analyze(alertname="ManualTest", ctx=ctx)
    print("=" * 60)
    print(result.text)
    print("=" * 60)
    print(f"model={result.model} in={result.input_tokens} out={result.output_tokens}")
