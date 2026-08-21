# Mastercard Innovation Challenge @ GFF 2026
## Track: AI Defense Lab for Payment Security

> **Build the attack, then build the defense.**

GenAI is making payment fraud faster, cheaper, and harder to spot. In this red team / blue team
challenge, you take on both sides of the problem by building **one end-to-end Red Teaming AI system**
that identifies novel GenAI-powered payment fraud, generates realistic simulations of it at scale,
and defends against it with an accurate detection model.

**Goal:** Build a closed-loop AI system that discovers emerging GenAI payment fraud, recreates it
with high fidelity, and reliably detects it. The best solutions turn their own simulated attacks into
the training ground for a stronger defense.

**Host event:** Global Fintech Fest (GFF) 2026 — 9–11 September, Jio World Centre, Mumbai.

**Strategy & direction map:** see [SOLUTION-TREE.md](SOLUTION-TREE.md).

**Responsible use:** see [docs/RESPONSIBLE-RED-TEAMING.md](docs/RESPONSIBLE-RED-TEAMING.md).

---

## Repository

```
redlab/
  taxonomy/         IDENTIFY - machine-readable attack corpus (42 vectors, 8 families)
    schema.py       morphological axes + AttackVector contract
    loader.py       validation, coverage reporting, leave-one-family-out splits
    cli.py          stats / matrix / top / show / split
    vectors/*.yaml  the corpus
  sim/              GENERATE - agent-based world + attack injection
    calibration.py  calibration targets + the fidelity-honesty note
    conditionals.py joint structure (amount|mcc, hour|mcc, channel|mcc, loyalty)
    world.py        entity-persistent legitimate population
    fraud_profile.py how real fraud behaves, measured on 28,619 labelled frauds
    attacks.py      spec-driven campaign injection (all 42 vectors, no per-vector code)
    fidelity.py     marginal / stylised / adversarial scoring of the legit population
    attack_fidelity.py detectability-signature comparison vs reference fraud
  defend/           DEFEND - detection and evaluation
    features.py     30 causal features (past-only by construction)
    detect.py       detector + FPR-anchored and unseen-family evaluation
  loop/             adversarial curriculum (D2) and probing agent (D3)   [next]
scripts/            build_dataset / eval_world / eval_attacks / train_detector / ablation
tests/              28 guardrails on corpus integrity, world structure, attack realism
docs/               FINDINGS.md (measured results), RESPONSIBLE-RED-TEAMING.md
```

### Quickstart

```bash
python3 -m pytest tests/ -q                      # 28 tests, ~8s

# IDENTIFY
python3 -m redlab.taxonomy.cli stats
python3 -m redlab.taxonomy.cli split --holdout agentic_commerce anti_defense

# One-time calibration (needs the reference corpus in data/raw/)
python3 -m redlab.sim.extract_profile
python3 -m redlab.sim.conditionals
python3 -m redlab.sim.fraud_profile

# GENERATE -> DEFEND
python3 scripts/build_dataset.py                 # world + attacks + features
python3 scripts/eval_world.py                    # legit-population fidelity
python3 scripts/eval_attacks.py                  # attack fidelity
python3 scripts/train_detector.py                # detector, honest evaluation
python3 scripts/ablation.py                      # feature ablation + mechanism holdout
```

Runs on Python 3.9+ with the dependencies in `requirements.txt`.

---

## Background

Generative AI has lowered the barrier for sophisticated, fast-evolving payment fraud. Static,
rule-based defenses struggle to keep pace with attacks that are novel, adaptive, and generated at
scale. The challenge invites participants to fight fire with fire: use AI to stress-test and harden
payment security.

Rather than tackling detection in isolation, participants own the full cycle — first imagining how
fraud could evolve, then building the tooling to reproduce it faithfully, and finally engineering a
model that catches it. The strongest submissions treat the three pillars as a **single feedback
loop**: the attacks you generate become the training and stress-testing ground for the defense you
build, and the gaps your defense reveals feed back into new attack ideas.

```
   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
   │   IDENTIFY   │─────▶│   GENERATE   │─────▶│    DEFEND    │
   │ map emerging │      │ simulate at  │      │ detect, flag │
   │ attack space │      │ scale, high  │      │ and mitigate │
   └──────────────┘      │   fidelity   │      └──────────────┘
          ▲              └──────────────┘             │
          │                                           │
          └──────── gaps revealed feed new ideas ─────┘
```

---

## The Problem: Identify → Generate → Defend

### 1. Identify (ideate)
Research and map the landscape of emerging, novel GenAI-powered fraud attacks targeting payments.
Be thorough and exhaustive: the goal is **breadth and depth**. Surface as many distinct, plausible
attack vectors as possible across channels, rails, and social-engineering surfaces rather than a
narrow handful. Ground each idea in how real payment systems and fraud actually work.

### 2. Generate
Build algorithms and agents that generate and simulate those attacks at scale. Prioritise
**fidelity**: the synthetic attacks and transactions should closely resemble real payment data and
real fraud patterns — realistic distributions, behaviours and edge cases — so they are genuinely
useful for training and stress-testing a defense.

### 3. Defend
Build an AI/ML solution (for example, a classifier) that detects, flags, and mitigates the generated
attacks. Prioritise **accuracy**: maximise detection performance (precision, recall, F1 / AUC) on the
simulated attacks while keeping false positives on legitimate payments low.

---

## Evaluation Criteria

Submissions are scored on:

| # | Criterion |
|---|-----------|
| 1 | Diversity of attacks identified |
| 2 | Fidelity of attacks in simulation |
| 3 | Detection algorithms and their efficacy |
| 4 | Novelty of the overall solution |
| 5 | Real-world feasibility in live payments |

---

## Submission Requirements

A valid submission (write-up) must contain the following three artifacts, submitted from the
**"Writeups"** section prior to the deadline. **Any un-submitted or draft work by the deadline will
not be considered by the judges.**

### 1. Code Repository
A complete, runnable code repository (hosted on GitHub) covering all three pillars — identify,
generate, and defend. Code should be organized, documented, and reproducible.

### 2. Solution Walkthrough
A Word document (**.docx**) walking through the approach, including:
- The novel fraud attacks identified
- How the system generates and simulates those attacks
- The detection and mitigation model, with efficacy results
- Real-world feasibility in live payment environments

### 3. Working Prototype (Web)
A working web-based prototype with a presentable UI that demonstrates the closed-loop system in
action.

---

## Timeline

| Date | Milestone | Status |
|------|-----------|--------|
| 10 Aug 2026 | Registration opens | ✅ Done |
| 20 Aug 2026 | Registration closes | ✅ Done |
| **31 Aug 2026** | **Submission deadline** | ⏳ Upcoming |
| 5 Sep 2026 | Results announced | — |
| 8–11 Sep 2026 | Top teams present at GFF 2026, Mumbai | — |

---

## Prizes & Recognition

Total track pool: **$4,707**

| Place | Prize (INR) | Prize (USD approx.) |
|-------|-------------|---------------------|
| 🥇 1st | ₹2,56,000 | ~$2,690 |
| 🥈 2nd | ₹1,28,000 | ~$1,345 |
| 🥉 3rd | ₹64,000 | ~$672 |

Winning teams also receive a showcase opportunity at GFF 2026, Mumbai.

---

## Who Should Participate

- Startups
- Individuals (tech professionals, market researchers, and others)
- Students (undergraduate, postgraduate, doctoral)
- Financial Institutions, Fintechs & DeepTech teams

**Team size:** 1–5 members.

---

## Submission Checklist

- [ ] GitHub repository — public/accessible, runnable end to end
- [ ] Repo covers all three pillars (identify / generate / defend)
- [ ] README with setup, run, and reproduction instructions
- [ ] Attack taxonomy documented (breadth + depth of vectors)
- [ ] Attack generation agents/algorithms with fidelity evidence
- [ ] Detection model with reported precision / recall / F1 / AUC and false-positive rate
- [ ] Closed-loop demonstrated (generated attacks feed defense training)
- [ ] Solution walkthrough as **.docx**
- [ ] Web prototype deployed and reachable, with presentable UI
- [ ] All artifacts **submitted** (not draft) in the Writeups section before 31 Aug 2026
