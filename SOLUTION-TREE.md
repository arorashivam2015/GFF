# Solution Tree — Mastercard Innovation Challenge @ GFF 2026
### AI Defense Lab for Payment Security · Strategy & Direction Map

> Companion to [README.md](README.md) (the challenge brief).
> Written 21 Aug 2026 · **10 days to the 31 Aug submission deadline.**

---

## Strategic Read

**What most teams will submit:** a list of ~15 attacks scraped from news articles → CTGAN on a
Kaggle fraud CSV → XGBoost → a Streamlit dashboard showing 0.99 AUC. That scores mid on every
criterion and low on novelty.

**Where the scoring has slack:** three of the five criteria — *diversity*, *novelty*,
*feasibility* — are judged on framing and method, not on model performance.

**The trap in criterion 3:** if you train a classifier on your own generated attacks, 0.99 AUC is
meaningless, and judges from Mastercard will know it. Building the *honest* version of that metric
is the single biggest differentiator available.

Legend: ★★ core move · ★ recommended · ○ if time · ✗ avoid

---

## Branch F — Data Anchoring
*Decide first — it gates every fidelity claim you can make.*

```
F1  IEEE-CIS Fraud Detection (Kaggle)                    ★ RECOMMENDED anchor
    ├─ real card-not-present e-commerce, ~590k txns, 400+ features
    └─ has card / device / email / address columns → real entity-graph structure

F2  PaySim / BankSim / Sparkov
    ├─ already synthetic, so fidelity claims against it are circular
    └─ but PaySim's transfer semantics are the closest public proxy to UPI-style P2P

F3  ULB creditcard.csv (PCA'd V1..V28)                   ✗ SKIP
    └─ anonymized components kill the fidelity narrative entirely

F4  Domain priors from published NPCI / RBI aggregates   ★ pair with F1
    ├─ UPI txn value distributions, MCC mix, time-of-day curves, decline-reason mix
    └─ no real UPI microdata is public — state that openly and show calibration to
       published aggregates instead. Honesty here reads as domain fluency.
```

**Take: F1 + F4.** Card rails anchored on real data; UPI rails calibrated to public aggregates
with the gap stated explicitly.

---

## Branch A — Identify
*Criterion: diversity of attacks identified.*

```
A1  Curated literature list, 20–40 attacks                        ✗ what everyone does

A2  Morphological taxonomy → machine-readable attack specs        ★★ CORE MOVE
    ├─ cross-product: Rail × Actor capability × GenAI uplift × Lifecycle stage × Victim surface
    ├─ mechanically generates hundreds of combinations; filter for plausibility
    ├─ each vector is a YAML/JSON object the GENERATOR consumes as input
    └─ pitch: "MITRE ATT&CK for GenAI payment fraud"

A3  LLM red-team ideation swarm layered on A2                     ★★
    ├─ proposer agent → critic agent (plausibility) → novelty scorer vs known corpus
    └─ this is the claim to "emerging / novel", and it demos live

A4  Threat-intel RAG over RBI/NPCI advisories, FinCEN, Europol IOCTA, court filings
    └─ grounds the taxonomy in what is actually happening; cheap credibility
```

**Structural insight:** A2 makes the taxonomy *machine-readable*, which is what physically wires
Identify → Generate. Without it, the three pillars are three separate projects with a slide
between them.

### Attack families to cover
Aim for **~8 families × ~10 vectors**, not 15 flat items.

| # | Family | Representative vectors |
|---|--------|------------------------|
| 1 | GenAI social engineering / APP fraud | deepfake voice vishing, video-KYC deepfake, LLM-personalized smishing at scale, pig-butchering chat agents, SEO-poisoned fake support numbers |
| 2 | Account takeover | OTP pretexting scripts, SIM-swap social engineering, credential stuffing with LLM-shaped behavioral evasion |
| 3 | Synthetic identity | GenAI-forged KYC docs, face-morph / injection vs liveness, long-horizon identity nurturing, bust-out fraud |
| 4 | Merchant-side | fake merchant onboarding with GenAI collateral, transaction laundering / MCC miscoding, LLM-written friendly-fraud chargeback narratives, refund abuse |
| 5 | Card testing / BIN attacks | distributed testing with adaptive rate + geo shaping, CoF token provisioning abuse, 3DS challenge evasion |
| 6 | **India rails** | UPI collect-request pretexting, AutoPay mandate abuse, QR swap/tamper, AePS biometric spoofing, mule-account layering networks, "digital arrest" scams, RuPay-credit-on-UPI |
| 7 | **Agentic commerce** | prompt injection against AI shopping agents, malicious merchant pages hijacking agent checkout, agent-to-agent payment protocol abuse |
| 8 | **Attacks on the defense itself** | black-box evasion, feedback-loop poisoning, threshold reverse-engineering via decline-response probing |

Families **6, 7, 8** are where the novelty points live:
- **6** — the judges are Indian payments practitioners at GFF.
- **7** — Mastercard ships Agent Pay; almost nobody else will cover this.
- **8** — closes the loop conceptually and feeds Branch D.

---

## Branch B — Generate
*Criterion: fidelity of attacks in simulation.*

```
B1  Tabular synthesis only (SDV / CTGAN / TVAE / copulas)
    └─ row-level realism, zero behavioral realism. Necessary floor, not sufficient.

B2  Agent-based payment world simulator                           ★★ CORE
    ├─ persistent entities: cardholders, merchants, devices, accounts, beneficiaries
    ├─ each with calendars, habits, geo, spend envelopes
    ├─ produces realistic SEQUENCES and GRAPH STRUCTURE — what B1 cannot fake
    └─ attack modules inject into a running legitimate population

B3  LLM attack agents layered on B2                               ★★
    ├─ input: an A2 attack spec
    ├─ output: a planned multi-step attack trace — txn sequence + timing + entity
    │  selection + the social-engineering text
    └─ this is what makes it "GenAI-powered fraud" rather than scripted fraud

B4  Multimodal artifacts
    ├─ DO:    LLM-generated scam SMS / email / chat transcripts — safe, cheap, and
    │         gives you a second detection channel
    └─ DON'T: actual deepfake audio or forged ID images — high effort, ethically
              fraught, and unnecessary. Generate the *scripts* and simulate the
              *signal* (e.g. liveness-score distributions under attack) instead.
```

### Prove fidelity numerically — never assert it
Ship a fidelity report containing:

- **KS distance** per marginal, real vs synthetic
- **Correlation / mutual-information matrix delta**
- **Discriminator AUC** — train a classifier to separate synthetic from real. AUC → 0.5 means
  indistinguishable. One number, honest, and it is your headline fidelity metric.
- **TSTR** (train-on-synthetic, test-on-real) — if it transfers, your synthesis is load-bearing
- **Graph-level stats** — degree distribution, component sizes, entity-reuse rates

---

## Branch C — Defend
*Criterion: detection algorithm efficacy.*

```
C1  Gradient-boosted trees (LightGBM / XGBoost / CatBoost)        ★ mandatory baseline
    └─ tabular + velocity features; the actual industry workhorse

C2  Sequence model per entity (GRU / small Transformer)           ○ if time
    └─ catches velocity drift and behavioral change over a history

C3  Graph detection                                               ★ high ROI
    ├─ account ↔ device ↔ merchant ↔ beneficiary graph
    ├─ catches mule networks and card-testing rings — big for the India story
    └─ graph FEATURES into C1 beat a hand-rolled GNN at 10-day scale

C4  Novelty / unsupervised channel                                ★★ CREDIBILITY MOVE
    ├─ isolation forest / autoencoder / conformal prediction
    └─ the premise of the challenge is UNSEEN attacks — a supervised model trained
       only on your own attacks is a tautology

C5  LLM analyst layer
    └─ text-channel classification + reason-code generation for the alert queue

C6  Risk DECISION engine, not a binary flag                       ★ underrated
    ├─ cost-sensitive thresholds → actions: allow / step-up auth / hold / decline / revoke mandate
    └─ the brief says "mitigates"; almost everyone will ship a 0/1 classifier
```

### Design the evaluation so it cannot be dismissed

- **PR-AUC**, not just ROC-AUC — imbalance is ~0.1–1%
- **Recall @ fixed FPR** (e.g. recall @ 0.1% FPR) — how payments risk teams actually talk
- **Leave-one-attack-family-out** — train on families 1–6, test on 7–8. This is the answer to
  "does it catch *emerging* fraud." The number will be far below 0.99. **Publish it anyway** —
  it is the most persuasive table in the submission.
- **Alert review rate / precision@k** for a realistic ops queue
- **₹ cost model** — fraud prevented vs friction imposed

---

## Branch D — The Loop
*Criterion: novelty. This is where you win or blend in.*

```
D1  One-shot: generate once → train once                          ✗ a pipeline, not a loop

D2  Adversarial curriculum over rounds                            ★★ THE SPINE
    ├─ attack generator rewarded for EVADING the current detector
    ├─ detector retrains on the survivors → repeat
    ├─ track: evasion rate per round ↓, detector recall on round-N attacks
    └─ "evasion 71% → 9% over 5 rounds" is the best chart in the deck

D3  Black-box probing agent                                       ★★ HERO NOVELTY
    ├─ LLM fraudster with QUERY-ONLY access to your detector
    ├─ reads accept / decline / step-up responses, infers thresholds, mutates strategy
    ├─ "an autonomous fraudster that reverse-engineers your risk rules"
    └─ then show the blue team detecting the PROBING pattern itself

D4  Full GAN / RL co-training                                     ✗ will not converge in 10 days
```

---

## Branch E — Web Prototype
*The judges see this. It must show the loop, not a metrics table.*

```
E1  Streamlit / Gradio                                   — fastest; reads as "hackathon project"
E2  Next.js + FastAPI                                    — polished; expensive in days
E3  FastAPI + one well-designed dashboard (HTMX / light React)   ★ best effort-to-impact ratio
```

### Screens, in priority order

1. **Red vs Blue Arena** — adversarial rounds running live, evasion-rate curve. *Lead with this.*
2. **Live Defense Console** — streaming txns, risk scores, alerts firing, mitigation action taken, reason codes
3. **Attack Atlas** — browsable ATT&CK-style matrix, filter by rail / family, click into a vector
4. **Simulation Studio** — pick vectors, set intensity, hit Run
5. **Fidelity Report** — real vs synthetic distribution overlays, discriminator AUC

Screens 1 and 2 *are* the demo. If time runs short, 3–5 can be static pages generated from
your artifacts.

---

## Recommended Spine

> **F1+F4 → A2+A3 → B2+B3 → C1+C3+C4+C6 → D2+D3 → E3**

**Positioning sentence:**

> *"A payment-fraud ATT&CK matrix plus an autonomous red-team agent that continuously mints novel
> attack variants — and a blue-team stack evaluated on attack families it has never seen, rather
> than on in-distribution AUC."*

**Two hero bets:** agentic-commerce attacks (family 7) and the black-box probing fraudster (D3).
Both are frontier, both are Mastercard-relevant, both demo in 60 seconds.

---

## 10-Day Plan (21 → 31 Aug 2026)

| Days | Focus |
|------|-------|
| **21–22 Aug** | Lock data anchor. Build A2 taxonomy schema + first 40 vectors. Repo skeleton. |
| **23–24 Aug** | B2 world simulator running with legitimate population; fidelity harness (discriminator AUC) working. |
| **25 Aug** | B3 attack agents generating traces for 4 families. C1 baseline trained. |
| **26 Aug** | C3 graph features + C4 novelty channel. Leave-one-family-out eval harness. |
| **27 Aug** | D2 adversarial rounds running end to end. **Make-or-break day.** |
| **28 Aug** | D3 probing agent + C6 decision engine. |
| **29 Aug** | E3 web prototype — Arena and Console screens. |
| **30 Aug** | .docx walkthrough. Fidelity + efficacy tables final. README / reproducibility pass. |
| **31 Aug** | Buffer, record demo, **submit early** — the brief states draft work is not judged. |

### Cut list, in this order
1. C2 sequence model
2. B4 text channel
3. Atlas / Studio screens go static
4. D3 becomes a documented design with a partial demo

### Never cut
- **D2** adversarial rounds
- **C4** novelty channel
- The **leave-one-family-out** table
- The **fidelity metrics**

---

## Two Things to Get Right

### 1. Scope the ethics deliberately — and say so in the repo
Everything synthetic and sandboxed. No real PII. No working tooling that touches real rails. No
deployable deepfake pipeline. Add a **"Responsible Red-Teaming"** section stating what you
deliberately did *not* build, and why.

Mastercard is judging. This reads as operational maturity and directly supports the *real-world
feasibility* criterion.

### 2. Feasibility is a criterion most teams will ignore entirely
Budget a full section on:

- Inference latency at authorization time (sub-100ms budget)
- Where each model sits: in the auth flow vs. post-auth / near-real-time
- Cold-start behaviour on genuinely new attack families
- Model governance, drift monitoring, retraining cadence
- What the false-positive rate costs in declined-good-customer terms

Cheap to write, and it is 20% of the score.
