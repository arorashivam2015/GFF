# Responsible Red-Teaming

This project builds an attack simulator. That is the brief. It is also a reason
to be explicit about where the boundary sits, because a red-team artefact that
cannot state its own limits is not a credible input to a payments environment.

## What this system does

- Generates **synthetic** payment traces, entities and campaign structures.
- Models attacks at the level of **observable signal** — amounts, timing, graph
  structure, score distributions, decision outcomes.
- Uses those traces to train and stress-test a detection model.

## What this system deliberately does not do

| Not built | Why |
|---|---|
| Working exploit tooling against any live rail, PSP or issuer | The simulator emits data, never traffic. No component holds a network client for a real payment endpoint. |
| Deepfake audio or video generation | The taxonomy models these vectors as *signal distributions* (liveness scores, challenge-response latency, retry counts). Producing the media adds no detection value and creates a redistributable harm artefact. |
| Forged identity-document images | Same reasoning. `PF-SID-001` is simulated as onboarding metadata and attribute-overlap structure, not as generated documents. |
| Real personal data of any kind | Every entity is synthetic. No breach corpus, no scraped PII, no real card numbers, VPAs or Aadhaar numbers. PANs are generated in reserved test ranges. |
| Scam scripts optimised for deployment | Where the text channel is modelled, content is generated to be *classifiable*, not persuasive, and is confined to the evaluation corpus. |
| Operational evasion guidance | `PF-ADV-*` vectors are simulated against our own detector inside the loop. The output is a hardened model, not a transferable evasion playbook. |

## Why the anti-defense family is in scope

`PF-ADV-001` through `PF-ADV-005` describe attacks on fraud models themselves.
Including them is defensive: each one is paired with a mitigation, and the
adversarial loop exists to measure whether our detector survives them. They are
evaluated in a closed sandbox against a model we own.

## Containment

- The simulator has no network egress path by design.
- Generated corpora stay under `data/` and `artifacts/`, both gitignored.
- The taxonomy records *mechanism and observable*, at the altitude an analyst
  needs to build detection — not step-by-step operational instructions.

## Deployment caveat

Nothing here is validated for production decisioning. Detection thresholds are
fitted to simulated data and would require recalibration against real
authorisation traffic, with the false-positive cost model re-derived from the
deploying institution's own economics.
