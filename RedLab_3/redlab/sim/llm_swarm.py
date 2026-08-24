"""The Proposer: one LLM call per taxonomy vector, generating a short
campaign script (pretext, message, or injected instruction) grounded in that
vector's own summary and preconditions - already-validated research content
from RedLab_1, not invented here.

Shells out to the `claude` CLI (non-interactive `claude -p`) rather than the
raw Anthropic API, since no ANTHROPIC_API_KEY is configured in this
environment - the CLI's own authenticated session is reused instead. Kept to
ONE call per vector, no separate Critic call, as an explicit simplification
for a solution scoped for build speed: a production version would add a
second pass that rejects outputs inconsistent with the vector's stated
preconditions, described but not implemented here.

RESPONSIBLE SCOPE: every prompt frames the request as fraud-awareness /
detection-training material and asks for a short, clearly-labelled synthetic
example - not a deployable script. This mirrors the same boundary RedLab_1
and RedLab_2 already state for other generated artefacts (no operational
exploit tooling, no real target data).
"""

import subprocess
from dataclasses import dataclass
from typing import List, Optional

from ..taxonomy.schema import AttackVector

PROMPT_TEMPLATE = """You are helping build a FRAUD-AWARENESS TRAINING dataset for a bank's \
security team, illustrating a known social-engineering pattern so staff and detection \
systems can recognise it. This is defensive research content, clearly labelled as a \
synthetic training example - not for operational use.

Attack pattern: {name}
Mechanism: {summary}
Known preconditions: {preconditions}

Write ONE short synthetic example (60-120 words) of what the attacker-side message or \
script might sound like in this pattern - a vishing call opener, an SMS, or a chat \
message, whichever fits the mechanism. Label it clearly as a synthetic training example. \
Output ONLY the example text, no preamble or explanation."""


@dataclass
class GeneratedCampaign:
    vector_id: str
    vector_name: str
    prompt: str
    text: str


def _call_claude(prompt: str, timeout: int = 60) -> str:
    result = subprocess.run(
        ["claude", "-p", prompt],
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI failed: {result.stderr[:300]}")
    return result.stdout.strip()


def generate_campaign(vector: AttackVector) -> GeneratedCampaign:
    prompt = PROMPT_TEMPLATE.format(
        name=vector.name,
        summary=vector.summary.strip(),
        preconditions="; ".join(vector.preconditions) or "none listed",
    )
    text = _call_claude(prompt)
    return GeneratedCampaign(vector_id=vector.id, vector_name=vector.name,
                             prompt=prompt, text=text)


def generate_swarm(vectors: List[AttackVector]) -> List[GeneratedCampaign]:
    out = []
    for v in vectors:
        print(f"  generating for {v.id} ({v.name})...")
        out.append(generate_campaign(v))
    return out
