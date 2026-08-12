"""Claude API wrapper — turns collected pod context into an RCA."""

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
        # Override with `export ANTHROPIC_MODEL=claude-sonnet-5` if you want newer
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
                max_tokens=1024,
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


_SYSTEM_PROMPT = """You are a senior Site Reliability Engineer diagnosing Kubernetes issues.

Given a firing alert plus the pod's status, events, and log tails, produce a concise
root-cause analysis and remediation plan.

Rules:
- Be specific — point to exact events, log lines, or numeric evidence.
- If the signal is ambiguous, say so; do not invent details.
- Assume the reader can run kubectl and edit manifests.
- Keep the whole response under ~300 words.

Format as Slack-flavored markdown, exactly these three sections:

*Root cause*
1-3 sentences naming the specific failure and the evidence pointing to it.

*Suggested fix*
Numbered list of concrete steps. Include kubectl commands or manifest changes where relevant.

*Confidence*
High / Medium / Low — with one sentence why.
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
        p.append(line)
        if c.message:
            p.append(f"  message: {c.message}")

    if ctx.events:
        p.append("\n# Events (newest first)")
        for e in ctx.events[:MAX_EVENTS]:
            p.append(
                f"- [{e['type']}/{e['reason']}] ×{e['count']} at {e['time']}: {e['message']}"
            )

    if ctx.previous_logs:
        p.append("\n# Previous-run log tails (before last restart — usually the smoking gun)")
        for name, txt in ctx.previous_logs.items():
            p.append(f"## {name}\n```\n{_tail(txt, MAX_LOG_LINES_PER_CONTAINER)}\n```")

    if ctx.logs:
        p.append("\n# Current log tails")
        for name, txt in ctx.logs.items():
            p.append(f"## {name}\n```\n{_tail(txt, MAX_LOG_LINES_PER_CONTAINER)}\n```")

    if ctx.errors:
        p.append(f"\n# Collector warnings\n{'; '.join(ctx.errors)}")

    return "\n".join(p)


def _tail(text: str, n_lines: int) -> str:
    return "\n".join(text.strip().splitlines()[-n_lines:])


# ---- CLI for manual testing: python llm.py <namespace> <pod> ----
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
