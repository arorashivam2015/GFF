# Nine End-to-End Build Prompts

Nine standalone prompts for the Mastercard Innovation Challenge @ GFF 2026 (AI Defense Lab for
Payment Security). Each prompt below is **fully independent** — every one repeats the complete
challenge brief, deliverables, judging criteria, discipline, ethical boundary, and data-anchoring
recipe in full. None of them reference each other. Paste any single one into a fresh LLM session
with no other context and it has everything needed to execute the full build alone.

They share a common backbone (the paragraphs above each `SPECIFIC MANDATE` header are identical
across all nine, by design) and diverge entirely below that line — different hero attack families,
different generation architecture, different detection architecture, different closed-loop
mechanism, different headline result.

---

## 1 — Graph-Native Mule & Layering Defense

```
You are building a complete submission for the Mastercard Innovation Challenge @ GFF 2026 —
AI Defense Lab for Payment Security (Global Fintech Fest, Mumbai; submission deadline
31 August 2026).

THE BRIEF: build one closed-loop red-team/blue-team AI system across three pillars —
(1) IDENTIFY novel GenAI-powered payment-fraud attack vectors, grounded in how real payment
systems and fraud actually work; (2) GENERATE realistic simulations of those attacks at scale,
with high fidelity to real payment data and fraud patterns; (3) DEFEND with a detection/
mitigation model maximising precision/recall/F1/AUC on the simulated attacks while keeping
false positives on legitimate payments low. Treat the three as one feedback loop: attacks you
generate should stress-test the defense, and the defense's gaps should feed back into new
attacks.

REQUIRED DELIVERABLES: (a) a complete, reproducible, tested GitHub repository covering all
three pillars; (b) a solution-walkthrough Word document (.docx) covering the attacks
identified, how they're simulated, the detection/mitigation model with efficacy results, and
real-world feasibility; (c) a working web-based prototype with a presentable UI demonstrating
the closed loop live.

JUDGED ON: diversity of attacks identified, fidelity of simulation, detection efficacy,
novelty, real-world feasibility in live payments.

NON-NEGOTIABLE DISCIPLINE: measure every fidelity and efficacy claim against a stated target —
never assert a number without showing how it was produced. Report defects you find honestly,
including ones in your own earlier work. A detector's headline number must come from
evaluation on a genuinely unseen attack mechanism, never an in-distribution split. If you find
yourself reporting >0.95 AUC as a headline, that is a signal to build a harder holdout, not a
result to celebrate.

RESPONSIBLE-SCOPE BOUNDARY: everything synthetic and sandboxed. No working exploit tooling
against any live rail, PSP or issuer. No deepfake audio/video or forged document image
generation — model these as observable signal distributions instead (liveness scores, latency,
retry counts). No real personal data or real account/card numbers. State this boundary
explicitly in your repo.

DATA ANCHORING: you have no Kaggle credentials. Anchor on IBM's TabFormer credit-card
transaction dataset (24.4M rows, github.com/IBM/TabFormer) — it sits behind Git LFS on a
public repo, and public-repo LFS objects download without auth via the LFS batch API (POST to
https://github.com/IBM/TabFormer.git/info/lfs/objects/batch with the oid/size from the
pointer file at data/credit_card/transactions.tgz, then GET the returned href). State plainly
in your fidelity section that IBM documents this corpus as itself synthetic — it is a
non-circular EXTERNAL reference for distributional comparison, not ground truth.

===========================================================================
YOUR SPECIFIC MANDATE: GRAPH-NATIVE MULE & LAYERING DEFENSE
===========================================================================

Your hypothesis: money movement is a graph problem before it is a tabular one. Test whether a
true graph-native detector earns its complexity over the much cheaper alternative of just
hand-engineering a few graph-derived features (entity fan-in/fan-out counts) into a tabular
model.

IDENTIFY: build your attack taxonomy as usual (rail x lifecycle-stage x GenAI-uplift x
victim-surface x actor-tier, machine-readable, ~35-45 vectors across a handful of families),
but weight it toward mule-account layering, cross-border money-laundering chains, and
transaction-laundering through generated storefronts. For every vector, add an explicit
`network_role` field (originator / layering-hop / collection-point / none) so campaigns can be
authored as graph motifs, not just event streams.

GENERATE: build an explicit temporal multigraph simulator (NetworkX or graph-tool), not just a
tabular sampler. Fit the LEGITIMATE population's topology to match the reference corpus's real
cardholder-merchant bipartite structure (degree distribution, clustering). Inject fraud
campaigns as graph motifs: fan-out (one compromised source funding many mules), layering
chains (value passed through several intermediate accounts within a tight time window,
preserving flow conservation), and fan-in (many unrelated victims converging on a collection
point). Measure fidelity on graph statistics specifically — degree distribution, clustering
coefficient, community structure, flow conservation across chains — using the same
discriminator-style "can a classifier tell generated from reference" methodology you'd use for
tabular fidelity, adapted to graph statistics.

DEFEND: build a temporal Graph Neural Network (PyTorch Geometric — GraphSAGE or a temporal
variant) that scores nodes and edges directly on the transaction graph. Separately build a
tabular GBM baseline using ONLY hand-engineered graph-proxy features (distinct users seen per
device/merchant before this transaction) and report the GNN's improvement over it explicitly —
this comparison IS your headline finding, not an afterthought.

LOOP: the adversarial attacker should mutate graph TOPOLOGY round over round (add hops, add
branching, mix in legitimate-looking intermediate accounts) to evade community-detection-based
flags, while the GNN retrains on each round's resulting graph snapshot. Track evasion rate per
round.

BUILD ORDER: (1) recon your environment/libraries; (2) download and profile the reference
corpus, extract its bipartite graph structure as your fidelity target; (3) build the taxonomy;
(4) build the graph simulator for the legitimate population first, validate its fidelity before
adding attacks; (5) inject motif-based campaigns, re-validate fidelity; (6) build the GNN
detector and the tabular-graph-proxy baseline side by side; (7) run the topology-mutation
adversarial loop; (8) build the web prototype (a graph visualization of a live layering chain
being flagged is your strongest demo screen); (9) write the .docx from your measured results.

HEADLINE RESULT TO PRODUCE: recall on a held-out MULE-TOPOLOGY variant never seen in
training (e.g. train only on fan-out motifs, test on layering-chain motifs) — for both the GNN
and the tabular-proxy baseline, so the gap between them is the story.
```

---

## 2 — LLM Red-Team Swarm

```
You are building a complete submission for the Mastercard Innovation Challenge @ GFF 2026 —
AI Defense Lab for Payment Security (Global Fintech Fest, Mumbai; submission deadline
31 August 2026).

THE BRIEF: build one closed-loop red-team/blue-team AI system across three pillars —
(1) IDENTIFY novel GenAI-powered payment-fraud attack vectors, grounded in how real payment
systems and fraud actually work; (2) GENERATE realistic simulations of those attacks at scale,
with high fidelity to real payment data and fraud patterns; (3) DEFEND with a detection/
mitigation model maximising precision/recall/F1/AUC on the simulated attacks while keeping
false positives on legitimate payments low. Treat the three as one feedback loop: attacks you
generate should stress-test the defense, and the defense's gaps should feed back into new
attacks.

REQUIRED DELIVERABLES: (a) a complete, reproducible, tested GitHub repository covering all
three pillars; (b) a solution-walkthrough Word document (.docx) covering the attacks
identified, how they're simulated, the detection/mitigation model with efficacy results, and
real-world feasibility; (c) a working web-based prototype with a presentable UI demonstrating
the closed loop live.

JUDGED ON: diversity of attacks identified, fidelity of simulation, detection efficacy,
novelty, real-world feasibility in live payments.

NON-NEGOTIABLE DISCIPLINE: measure every fidelity and efficacy claim against a stated target —
never assert a number without showing how it was produced. Report defects you find honestly,
including ones in your own earlier work. A detector's headline number must come from
evaluation on a genuinely unseen attack mechanism, never an in-distribution split. If you find
yourself reporting >0.95 AUC as a headline, that is a signal to build a harder holdout, not a
result to celebrate.

RESPONSIBLE-SCOPE BOUNDARY: everything synthetic and sandboxed. No working exploit tooling
against any live rail, PSP or issuer. No deepfake audio/video or forged document image
generation — model these as observable signal distributions instead (liveness scores, latency,
retry counts). No real personal data or real account/card numbers. State this boundary
explicitly in your repo.

DATA ANCHORING: you have no Kaggle credentials. Anchor on IBM's TabFormer credit-card
transaction dataset (24.4M rows, github.com/IBM/TabFormer) — it sits behind Git LFS on a
public repo, and public-repo LFS objects download without auth via the LFS batch API (POST to
https://github.com/IBM/TabFormer.git/info/lfs/objects/batch with the oid/size from the
pointer file at data/credit_card/transactions.tgz, then GET the returned href). State plainly
in your fidelity section that IBM documents this corpus as itself synthetic — it is a
non-circular EXTERNAL reference for distributional comparison, not ground truth.

===========================================================================
YOUR SPECIFIC MANDATE: LLM RED-TEAM SWARM
===========================================================================

Your hypothesis: the GenAI uplift in modern payment fraud lives in the LANGUAGE and the PLAN
— the pretext, the negotiation, the persona — not just in the transaction numbers. Test
whether attacks authored by an LLM agent that reasons and revises in natural language find
evasions that a purely parametric/numeric attack generator cannot.

IDENTIFY: weight your taxonomy toward social-engineering and agentic-commerce families —
voice-cloned vishing, LLM-run pig-butchering scams, deepfake executive-authorisation fraud,
prompt injection against AI shopping agents. For every vector, add a `playbook_prompt` field:
a natural-language mission brief (e.g. "you are a vishing operator impersonating a bank RM;
extract an OTP from a victim who just received a real-looking blocked-transaction alert") that
will seed your generation agent, alongside the usual numeric simulation profile.

GENERATE: build a three-agent pipeline using an LLM API (cache prompts aggressively for cost).
A PROPOSER agent drafts a full multi-turn campaign — victim-facing messages, pretext
escalation, timing decisions — conditioned on a taxonomy vector's playbook_prompt. A CRITIC
agent rejects anything mechanically inconsistent with the vector's stated preconditions (e.g.
a pretext that requires information the attacker wouldn't plausibly have). A lightweight
transaction-world model (build a simple agent-based population simulator: persistent users,
merchants, devices, calibrated to the reference corpus's amount/category/timing marginals)
converts the ACCEPTED plan into concrete, realistic transactions — the final output must still
be a calibrated transaction corpus, even though campaign STRUCTURE is LLM-authored. Keep the
actual generated script/message TEXT as a second data modality.

DEFEND: build two channels, late-fused. Channel one: a standard causal-feature tabular
detector (features computed only from each entity's PAST transactions — verify this by testing
that truncating a user's history doesn't change their earlier feature values). Channel two: a
classifier over embeddings of the generated text channel (chat/SMS/call-transcript). Report
detection efficacy per channel AND fused, so the walkthrough can show what text adds over
transactions alone.

LOOP: after each round, feed the Proposer agent BOTH the Critic's rejection reasons AND the
deployed detector's flag/no-flag outcome on its last campaign, and have it revise its OWN
prompting strategy in natural language (e.g. "your last three campaigns were flagged for rapid
new-payee-then-drain; try a slower escalation"). This is adaptation happening inside the LLM's
own reasoning, not a numeric parameter update — track evasion rate per round and specifically
note whether the STRATEGY the LLM converges on (not just the score) changes qualitatively.

BUILD ORDER: (1) recon environment and confirm LLM API access/budget; (2) reference-corpus
anchoring and calibration, same as any solution; (3) taxonomy with playbook prompts; (4) build
the lightweight world model first, standalone, and validate its fidelity; (5) build the
Proposer/Critic pipeline, dry-run it on 3-4 vectors and manually inspect outputs for
plausibility before scaling; (6) scale generation across the full taxonomy; (7) build the
dual-channel detector; (8) run the language-level adversarial loop; (9) web prototype (show a
live campaign transcript alongside its transaction trace and the detector's per-message risk
trajectory — this is your strongest, most visually distinctive demo screen); (10) .docx,
explicitly naming LLM API cost and latency as a feasibility trade-off since that's unique to
this approach.

HEADLINE RESULT TO PRODUCE: an evasion-rate-per-round curve, with a clear note on whether
language-level strategy changes (not parameter changes) are what drove any improvement.
```

---

## 3 — Adversarial Generative Modeling (GAN/Diffusion)

```
You are building a complete submission for the Mastercard Innovation Challenge @ GFF 2026 —
AI Defense Lab for Payment Security (Global Fintech Fest, Mumbai; submission deadline
31 August 2026).

THE BRIEF: build one closed-loop red-team/blue-team AI system across three pillars —
(1) IDENTIFY novel GenAI-powered payment-fraud attack vectors, grounded in how real payment
systems and fraud actually work; (2) GENERATE realistic simulations of those attacks at scale,
with high fidelity to real payment data and fraud patterns; (3) DEFEND with a detection/
mitigation model maximising precision/recall/F1/AUC on the simulated attacks while keeping
false positives on legitimate payments low. Treat the three as one feedback loop: attacks you
generate should stress-test the defense, and the defense's gaps should feed back into new
attacks.

REQUIRED DELIVERABLES: (a) a complete, reproducible, tested GitHub repository covering all
three pillars; (b) a solution-walkthrough Word document (.docx) covering the attacks
identified, how they're simulated, the detection/mitigation model with efficacy results, and
real-world feasibility; (c) a working web-based prototype with a presentable UI demonstrating
the closed loop live.

JUDGED ON: diversity of attacks identified, fidelity of simulation, detection efficacy,
novelty, real-world feasibility in live payments.

NON-NEGOTIABLE DISCIPLINE: measure every fidelity and efficacy claim against a stated target —
never assert a number without showing how it was produced. Report defects you find honestly,
including ones in your own earlier work. A detector's headline number must come from
evaluation on a genuinely unseen attack mechanism, never an in-distribution split. If you find
yourself reporting >0.95 AUC as a headline, that is a signal to build a harder holdout, not a
result to celebrate.

RESPONSIBLE-SCOPE BOUNDARY: everything synthetic and sandboxed. No working exploit tooling
against any live rail, PSP or issuer. No deepfake audio/video or forged document image
generation — model these as observable signal distributions instead (liveness scores, latency,
retry counts). No real personal data or real account/card numbers. State this boundary
explicitly in your repo.

DATA ANCHORING: you have no Kaggle credentials. Anchor on IBM's TabFormer credit-card
transaction dataset (24.4M rows, github.com/IBM/TabFormer) — it sits behind Git LFS on a
public repo, and public-repo LFS objects download without auth via the LFS batch API (POST to
https://github.com/IBM/TabFormer.git/info/lfs/objects/batch with the oid/size from the
pointer file at data/credit_card/transactions.tgz, then GET the returned href). State plainly
in your fidelity section that IBM documents this corpus as itself synthetic — it is a
non-circular EXTERNAL reference for distributional comparison, not ground truth.

===========================================================================
YOUR SPECIFIC MANDATE: ADVERSARIAL GENERATIVE MODELING
===========================================================================

Your hypothesis: deep generative models (GANs, diffusion) are the "obvious" answer to
synthetic fraud generation that most teams will reach for — test that assumption honestly
against a much simpler baseline, and pair it with an UNSUPERVISED detector, since the premise
of this challenge (novel, unseen fraud) is a natural fit for anomaly detection rather than
supervised classification.

IDENTIFY: weight your taxonomy toward synthetic-identity fraud — generated KYC documents,
face-morph liveness attacks, agent-nurtured bust-out identities — the family most naturally
about generating a convincing FAKE ARTIFACT rather than a behavioural sequence.

GENERATE: train a conditional tabular generative model — CTGAN, TVAE, or a tabular diffusion
model (e.g. implement or adapt TabDDPM) — on the reference corpus, conditioned on an
attack-vector embedding so one model can emit any taxonomy vector on demand. CRITICALLY: build
a full fidelity-measurement harness FIRST, before trusting any generator output — marginal
tests (KS distance, Jensen-Shannon divergence on amount/category/hour/channel), stylised-fact
tests (Benford first-digit conformance, merchant-popularity Zipf slope, card-not-present fraud
lift), and an ADVERSARIAL test (train a classifier to tell generated rows from reference rows;
report its AUC — 0.5 is indistinguishable, higher is worse). Test a naive/simple baseline
generator first to prove your harness actually catches bad synthesis (expect something like
AUC 0.95+ for naive sampling), THEN test your trained generative model against the same
harness and report the real number, whatever it is — do not tune the harness to flatter the
model.

DEFEND: your PRIMARY detector must be UNSUPERVISED — a deep autoencoder or VAE trained ONLY on
legitimate transactions, using reconstruction error plus latent-space distance as the anomaly
score, with NO fraud labels used in training at all. Build a supervised GBM on the same
generated fraud purely as a comparison baseline, to quantify what labels buy you over none.

LOOP: implement a TRUE minimax adversarial-training loop — the generator's loss function
should incorporate the (frozen-per-step) detector's output directly, i.e. real gradient-based
adversarial training, not a black-box parameter search. State explicitly in your writeup that
this is a WHITE-BOX threat model (the attacker has gradient access to a differentiable
generator) as opposed to a black-box query-only attacker, and that both threat models are
worth having in a portfolio — don't claim this one is more realistic than the other, just be
clear about which one it is.

BUILD ORDER: (1) recon and data anchoring; (2) build the fidelity harness completely and
validate it against a naive baseline BEFORE building your real generator — this order matters,
it's how you catch your own mistakes early; (3) build and train the conditional generative
model, iterate against the harness until the discriminator AUC is as low as you can reasonably
get it, documenting each fix you make and why; (4) build the taxonomy driving the conditioning;
(5) build the unsupervised detector and the supervised comparison baseline; (6) run the
minimax adversarial loop; (7) web prototype (a live "fidelity dashboard" comparing generated
vs. reference distributions is your strongest demo screen here); (8) .docx.

HEADLINE RESULT TO PRODUCE: your generator's discriminator AUC (report the naive baseline's
AUC alongside it, for contrast), and the unsupervised detector's recall on a held-out
mechanism it never saw, with zero fraud-label exposure during training.
```

---

## 4 — Sequence-Transformer Behavioral Modeling

```
You are building a complete submission for the Mastercard Innovation Challenge @ GFF 2026 —
AI Defense Lab for Payment Security (Global Fintech Fest, Mumbai; submission deadline
31 August 2026).

THE BRIEF: build one closed-loop red-team/blue-team AI system across three pillars —
(1) IDENTIFY novel GenAI-powered payment-fraud attack vectors, grounded in how real payment
systems and fraud actually work; (2) GENERATE realistic simulations of those attacks at scale,
with high fidelity to real payment data and fraud patterns; (3) DEFEND with a detection/
mitigation model maximising precision/recall/F1/AUC on the simulated attacks while keeping
false positives on legitimate payments low. Treat the three as one feedback loop: attacks you
generate should stress-test the defense, and the defense's gaps should feed back into new
attacks.

REQUIRED DELIVERABLES: (a) a complete, reproducible, tested GitHub repository covering all
three pillars; (b) a solution-walkthrough Word document (.docx) covering the attacks
identified, how they're simulated, the detection/mitigation model with efficacy results, and
real-world feasibility; (c) a working web-based prototype with a presentable UI demonstrating
the closed loop live.

JUDGED ON: diversity of attacks identified, fidelity of simulation, detection efficacy,
novelty, real-world feasibility in live payments.

NON-NEGOTIABLE DISCIPLINE: measure every fidelity and efficacy claim against a stated target —
never assert a number without showing how it was produced. Report defects you find honestly,
including ones in your own earlier work. A detector's headline number must come from
evaluation on a genuinely unseen attack mechanism, never an in-distribution split. If you find
yourself reporting >0.95 AUC as a headline, that is a signal to build a harder holdout, not a
result to celebrate.

RESPONSIBLE-SCOPE BOUNDARY: everything synthetic and sandboxed. No working exploit tooling
against any live rail, PSP or issuer. No deepfake audio/video or forged document image
generation — model these as observable signal distributions instead (liveness scores, latency,
retry counts). No real personal data or real account/card numbers. State this boundary
explicitly in your repo.

DATA ANCHORING: you have no Kaggle credentials. Anchor on IBM's TabFormer credit-card
transaction dataset (24.4M rows, github.com/IBM/TabFormer) — it sits behind Git LFS on a
public repo, and public-repo LFS objects download without auth via the LFS batch API (POST to
https://github.com/IBM/TabFormer.git/info/lfs/objects/batch with the oid/size from the
pointer file at data/credit_card/transactions.tgz, then GET the returned href). State plainly
in your fidelity section that IBM documents this corpus as itself synthetic — it is a
non-circular EXTERNAL reference for distributional comparison, not ground truth.

===========================================================================
YOUR SPECIFIC MANDATE: SEQUENCE-TRANSFORMER BEHAVIORAL MODELING
===========================================================================

Your hypothesis: a cardholder's spend is fundamentally a SEQUENCE, and hand-engineered
aggregate features (rolling means, velocity windows) are a lossy summary of that sequence.
Test whether a model that reads raw event sequences directly outperforms hand-engineered
features, and at what cost in latency/interpretability.

IDENTIFY: weight your taxonomy toward account-takeover and card-testing families — both are
fundamentally about a SEQUENCE of events deviating from an established pattern (the object
under attack is the trajectory, not any single transaction): SIM-swap pretexting, credential
stuffing, adaptive BIN enumeration, 3DS-exemption farming.

GENERATE: train a small autoregressive transformer (GPT-2 scale is enough; note that IBM's own
TabFormer paper, which you're anchoring your reference data on, ships exactly this recipe —
worth reading their methodology) directly on tokenized per-user transaction sequences from the
reference corpus. Use the trained model two ways: (a) to GENERATE realistic legitimate
sequences for your simulated population, in place of a hand-built statistical sampler; (b)
conditioned on a taxonomy vector's simulation profile, to author a "fraud continuation" — feed
it a real legitimate prefix and have it continue the sequence in a way that matches the
attack's mechanism. Measure this generator's own fidelity (perplexity on held-out real
sequences, plus a discriminator-AUC test identical in spirit to any tabular fidelity harness)
and compare it explicitly against what a simpler hand-built statistical sampler would achieve
— build the simpler baseline too, even briefly, so the comparison is real.

DEFEND: your PRIMARY detector is a sequence classifier — a small transformer encoder that
reads an entity's raw recent event history directly and outputs a per-event risk score. Build
a SEPARATE tabular GBM baseline using ~25-30 hand-engineered causal features (velocity counts,
novelty flags, baseline-deviation ratios — computed strictly from PAST events only, verified by
testing that truncating history doesn't change earlier rows' features). Run both on the
IDENTICAL held-out-mechanism split and report the gap, along with each model's inference
latency, since that's a real deployment trade-off worth being honest about.

LOOP: the adversarial attacker should mutate the SEQUENCE itself round over round — reorder
events, change dwell-time between them, interleave decoy legitimate-looking transactions —
via a small search over sequence-edit actions (a beam search or simple RL-lite policy is
enough), rather than mutating a fixed scalar parameter vector. Track evasion rate per round for
both the sequence-transformer detector and the tabular-feature detector, since a sequence-edit
attacker is exactly the kind that should be harder for feature-based detection to catch.

BUILD ORDER: (1) recon, data anchoring, calibration; (2) train the sequence-generation
transformer on the reference corpus, validate its fidelity before using it for anything else;
(3) taxonomy; (4) build the simple statistical-sampler baseline generator for comparison; (5)
inject fraud continuations via the transformer; (6) build both detectors (sequence-transformer
and feature-engineered GBM) side by side on identical splits; (7) run the sequence-edit
adversarial loop against both; (8) web prototype (visualize a raw event sequence with the
sequence model's per-token risk overlay — a genuinely different, distinctive demo screen); (9)
.docx, with an explicit latency/interpretability section comparing the two detector
architectures.

HEADLINE RESULT TO PRODUCE: sequence-transformer recall vs. feature-engineered-GBM recall on
the identical unseen-mechanism holdout, reported alongside each model's inference latency per
transaction.
```

---

## 5 — Zero-Prior-Label Anomaly-First Defense

```
You are building a complete submission for the Mastercard Innovation Challenge @ GFF 2026 —
AI Defense Lab for Payment Security (Global Fintech Fest, Mumbai; submission deadline
31 August 2026).

THE BRIEF: build one closed-loop red-team/blue-team AI system across three pillars —
(1) IDENTIFY novel GenAI-powered payment-fraud attack vectors, grounded in how real payment
systems and fraud actually work; (2) GENERATE realistic simulations of those attacks at scale,
with high fidelity to real payment data and fraud patterns; (3) DEFEND with a detection/
mitigation model maximising precision/recall/F1/AUC on the simulated attacks while keeping
false positives on legitimate payments low. Treat the three as one feedback loop: attacks you
generate should stress-test the defense, and the defense's gaps should feed back into new
attacks.

REQUIRED DELIVERABLES: (a) a complete, reproducible, tested GitHub repository covering all
three pillars; (b) a solution-walkthrough Word document (.docx) covering the attacks
identified, how they're simulated, the detection/mitigation model with efficacy results, and
real-world feasibility; (c) a working web-based prototype with a presentable UI demonstrating
the closed loop live.

JUDGED ON: diversity of attacks identified, fidelity of simulation, detection efficacy,
novelty, real-world feasibility in live payments.

NON-NEGOTIABLE DISCIPLINE: measure every fidelity and efficacy claim against a stated target —
never assert a number without showing how it was produced. Report defects you find honestly,
including ones in your own earlier work. A detector's headline number must come from
evaluation on a genuinely unseen attack mechanism, never an in-distribution split. If you find
yourself reporting >0.95 AUC as a headline, that is a signal to build a harder holdout, not a
result to celebrate.

RESPONSIBLE-SCOPE BOUNDARY: everything synthetic and sandboxed. No working exploit tooling
against any live rail, PSP or issuer. No deepfake audio/video or forged document image
generation — model these as observable signal distributions instead (liveness scores, latency,
retry counts). No real personal data or real account/card numbers. State this boundary
explicitly in your repo.

DATA ANCHORING: you have no Kaggle credentials. Anchor on IBM's TabFormer credit-card
transaction dataset (24.4M rows, github.com/IBM/TabFormer) — it sits behind Git LFS on a
public repo, and public-repo LFS objects download without auth via the LFS batch API (POST to
https://github.com/IBM/TabFormer.git/info/lfs/objects/batch with the oid/size from the
pointer file at data/credit_card/transactions.tgz, then GET the returned href). State plainly
in your fidelity section that IBM documents this corpus as itself synthetic — it is a
non-circular EXTERNAL reference for distributional comparison, not ground truth.

===========================================================================
YOUR SPECIFIC MANDATE: ZERO-PRIOR-LABEL ANOMALY-FIRST DEFENSE
===========================================================================

Your hypothesis: the entire premise of this challenge is that the fraud is NOVEL — so build a
detector on the working assumption that it will see ZERO labelled examples of any given attack
before encountering it live, and make that the primary product, not a secondary ablation next
to a supervised model.

IDENTIFY: build your full taxonomy as usual, but explicitly rate each vector's maturity
(observed / emerging / projected — i.e. already seen in the wild, early real-world reports, or
purely projected from existing capability). Weight some vectors toward attacks-on-the-defense-
itself (an attacker probing your detector's decision boundary, an attacker trying to poison
your training labels) — these are the vectors most philosophically aligned with "assume nothing
is labelled."

GENERATE: build your world simulator and attack injection as usual (agent-based population,
calibrated to the reference corpus, taxonomy-driven campaign injection — reuse standard
techniques here, since your differentiation is entirely in DEFEND). The one thing to build with
extra care: a ZERO-EXPOSURE split utility that guarantees your projected-maturity vectors never
touch the detector's training data even indirectly (not just "this family is held out" — check
that no shared low-level parameter or entity leaks information about held-out vectors into
training).

DEFEND: your ONLY primary detection channel is UNSUPERVISED. Build an ensemble of at least
three methods: an Isolation Forest, a deep autoencoder or Deep SVDD trained purely on
legitimate transactions (reconstruction error / distance-to-normal-manifold as the anomaly
score), and a conformal-prediction layer on top that gives you a STATISTICALLY CALIBRATED
anomaly threshold (with an actual coverage guarantee you can verify empirically) rather than an
arbitrary quantile cutoff. Build a supervised GBM SEPARATELY, trained only on observed and
emerging vectors, and NEVER let it touch projected-maturity vectors — keep it purely as a
labelled-data comparison point, not your headline model.

LOOP: your adversarial loop's job is different from a typical retrain-on-evaded-fraud cycle —
each round, recalibrate your CONFORMAL THRESHOLD using updated quantiles as the attacker
searches for statistically unremarkable-looking campaigns, and report how many projected-
maturity attacks your unsupervised ensemble catches, across rounds, with literally zero label
exposure at any point.

BUILD ORDER: (1) recon, data anchoring, calibration; (2) taxonomy with maturity ratings; (3)
world simulator and attack injection (standard); (4) build the zero-exposure split utility and
verify it rigorously — write a test that would fail if information leaked; (5) build the
unsupervised ensemble, calibrate the conformal layer, verify its coverage guarantee holds
empirically on a validation set; (6) build the supervised baseline for comparison only; (7) run
the threshold-recalibration adversarial loop; (8) web prototype (show the calibrated confidence
interval / anomaly score distribution live, not just a binary flag — this is your most
distinctive screen, since most fraud dashboards don't show calibrated uncertainty); (9) .docx.

HEADLINE RESULT TO PRODUCE: recall on the projected-maturity, zero-label-exposure slice for
your unsupervised ensemble vs. the supervised baseline (expect the supervised model to do worse
here — that's the finding, not a failure), plus empirical verification that your conformal
threshold's coverage guarantee actually holds.
```

---

## 6 — Federated Consortium Defense

```
You are building a complete submission for the Mastercard Innovation Challenge @ GFF 2026 —
AI Defense Lab for Payment Security (Global Fintech Fest, Mumbai; submission deadline
31 August 2026).

THE BRIEF: build one closed-loop red-team/blue-team AI system across three pillars —
(1) IDENTIFY novel GenAI-powered payment-fraud attack vectors, grounded in how real payment
systems and fraud actually work; (2) GENERATE realistic simulations of those attacks at scale,
with high fidelity to real payment data and fraud patterns; (3) DEFEND with a detection/
mitigation model maximising precision/recall/F1/AUC on the simulated attacks while keeping
false positives on legitimate payments low. Treat the three as one feedback loop: attacks you
generate should stress-test the defense, and the defense's gaps should feed back into new
attacks.

REQUIRED DELIVERABLES: (a) a complete, reproducible, tested GitHub repository covering all
three pillars; (b) a solution-walkthrough Word document (.docx) covering the attacks
identified, how they're simulated, the detection/mitigation model with efficacy results, and
real-world feasibility; (c) a working web-based prototype with a presentable UI demonstrating
the closed loop live.

JUDGED ON: diversity of attacks identified, fidelity of simulation, detection efficacy,
novelty, real-world feasibility in live payments.

NON-NEGOTIABLE DISCIPLINE: measure every fidelity and efficacy claim against a stated target —
never assert a number without showing how it was produced. Report defects you find honestly,
including ones in your own earlier work. A detector's headline number must come from
evaluation on a genuinely unseen attack mechanism, never an in-distribution split. If you find
yourself reporting >0.95 AUC as a headline, that is a signal to build a harder holdout, not a
result to celebrate.

RESPONSIBLE-SCOPE BOUNDARY: everything synthetic and sandboxed. No working exploit tooling
against any live rail, PSP or issuer. No deepfake audio/video or forged document image
generation — model these as observable signal distributions instead (liveness scores, latency,
retry counts). No real personal data or real account/card numbers. State this boundary
explicitly in your repo.

DATA ANCHORING: you have no Kaggle credentials. Anchor on IBM's TabFormer credit-card
transaction dataset (24.4M rows, github.com/IBM/TabFormer) — it sits behind Git LFS on a
public repo, and public-repo LFS objects download without auth via the LFS batch API (POST to
https://github.com/IBM/TabFormer.git/info/lfs/objects/batch with the oid/size from the
pointer file at data/credit_card/transactions.tgz, then GET the returned href). State plainly
in your fidelity section that IBM documents this corpus as itself synthetic — it is a
non-circular EXTERNAL reference for distributional comparison, not ground truth.

===========================================================================
YOUR SPECIFIC MANDATE: FEDERATED CONSORTIUM DEFENSE
===========================================================================

Your hypothesis: several real attack mechanisms (mule-account layering, distributed BIN
testing) explicitly exploit the fact that any SINGLE institution only sees a fragment of the
attack — but a detector trained as if one issuer sees the whole picture is testing an
unrealistic scenario. Model the blind spot honestly, and quantify what a privacy-preserving
consortium recovers.

IDENTIFY: weight your taxonomy toward mule-account layering networks and distributed/adaptive
card-testing (BIN enumeration spread across many acquirers specifically to stay under any one
institution's velocity threshold) — both are naturally cross-institution attacks.

GENERATE: this is the core engineering lift of your solution — extend your world simulator to
a MULTI-INSTITUTION setting. Partition your simulated cardholders and merchants across N
distinct synthetic institutions (issuers/acquirers/PSPs). Make your mule-layering and
BIN-testing campaigns deliberately SPAN institutions: a layering chain that hops accounts
across 3 different simulated banks; a testing campaign that spreads its probes across 5
simulated acquirers specifically to stay under any single institution's own velocity threshold.
This needs to be a genuine structural extension of your world model — an attacker whose
observable footprint at any ONE institution looks unremarkable, but whose footprint across ALL
institutions is obviously coordinated.

DEFEND: implement actual federated learning across your simulated institutions — local model
training per institution with only model updates (gradients or weight deltas), never raw
transaction data, shared to a central aggregator (use the Flower framework, or hand-roll a
simple FedAvg loop — either is fine, just be explicit about what's actually shared). Build and
report THREE models side by side on the identical cross-institution attack test set: (a) each
institution's LOCAL-ONLY model, trained on just its own data — this is the realistic status
quo; (b) the FEDERATED global model; (c) an unrealistic FULLY-CENTRALIZED oracle model with
complete data pooling, included specifically as an upper-bound reference. Report all three, so
the walkthrough can honestly state how much of the centralized model's advantage federation
actually recovers — do not just report the federated number in isolation.

LOOP: run your adversarial curriculum TWICE against the identical attacker — once against the
local-only defenders, once against the federated defender — and report the EVASION-CEILING GAP
between the two as your headline comparative result.

BUILD ORDER: (1) recon, data anchoring, calibration; (2) taxonomy; (3) build the
multi-institution world extension and validate that legitimate traffic still looks realistic
per-institution; (4) inject cross-institution campaigns, verify their per-institution footprint
genuinely looks unremarkable in isolation (this is the property that makes federation matter —
test it explicitly); (5) build the local-only, federated, and centralized-oracle models; (6)
run the adversarial loop against local-only and federated separately; (7) web prototype (a
multi-institution view showing the same mule chain as invisible from any single institution's
dashboard but visible from the federated one is your strongest, most on-thesis demo screen);
(8) .docx, with a strong feasibility section — federated learning is literally how real card
networks and bank consortiums discuss fraud-signal sharing today, so this section should be
your most concrete and compelling.

HEADLINE RESULT TO PRODUCE: the (local-only / federated / centralized-oracle) recall table on
the cross-institution attack subset, and the with-vs-without-federation evasion-ceiling gap
from the adversarial loop.
```

---

## 7 — Causal & Counterfactual Detection

```
You are building a complete submission for the Mastercard Innovation Challenge @ GFF 2026 —
AI Defense Lab for Payment Security (Global Fintech Fest, Mumbai; submission deadline
31 August 2026).

THE BRIEF: build one closed-loop red-team/blue-team AI system across three pillars —
(1) IDENTIFY novel GenAI-powered payment-fraud attack vectors, grounded in how real payment
systems and fraud actually work; (2) GENERATE realistic simulations of those attacks at scale,
with high fidelity to real payment data and fraud patterns; (3) DEFEND with a detection/
mitigation model maximising precision/recall/F1/AUC on the simulated attacks while keeping
false positives on legitimate payments low. Treat the three as one feedback loop: attacks you
generate should stress-test the defense, and the defense's gaps should feed back into new
attacks.

REQUIRED DELIVERABLES: (a) a complete, reproducible, tested GitHub repository covering all
three pillars; (b) a solution-walkthrough Word document (.docx) covering the attacks
identified, how they're simulated, the detection/mitigation model with efficacy results, and
real-world feasibility; (c) a working web-based prototype with a presentable UI demonstrating
the closed loop live.

JUDGED ON: diversity of attacks identified, fidelity of simulation, detection efficacy,
novelty, real-world feasibility in live payments.

NON-NEGOTIABLE DISCIPLINE: measure every fidelity and efficacy claim against a stated target —
never assert a number without showing how it was produced. Report defects you find honestly,
including ones in your own earlier work. A detector's headline number must come from
evaluation on a genuinely unseen attack mechanism, never an in-distribution split. If you find
yourself reporting >0.95 AUC as a headline, that is a signal to build a harder holdout, not a
result to celebrate.

RESPONSIBLE-SCOPE BOUNDARY: everything synthetic and sandboxed. No working exploit tooling
against any live rail, PSP or issuer. No deepfake audio/video or forged document image
generation — model these as observable signal distributions instead (liveness scores, latency,
retry counts). No real personal data or real account/card numbers. State this boundary
explicitly in your repo.

DATA ANCHORING: you have no Kaggle credentials. Anchor on IBM's TabFormer credit-card
transaction dataset (24.4M rows, github.com/IBM/TabFormer) — it sits behind Git LFS on a
public repo, and public-repo LFS objects download without auth via the LFS batch API (POST to
https://github.com/IBM/TabFormer.git/info/lfs/objects/batch with the oid/size from the
pointer file at data/credit_card/transactions.tgz, then GET the returned href). State plainly
in your fidelity section that IBM documents this corpus as itself synthetic — it is a
non-circular EXTERNAL reference for distributional comparison, not ground truth.

===========================================================================
YOUR SPECIFIC MANDATE: CAUSAL & COUNTERFACTUAL DETECTION
===========================================================================

Your hypothesis: a detector that flags "first-time high-value payee" or "sudden spend
escalation" will ALSO flag someone's first rent payment after moving city, or a small
business's genuine growth ramp. Test explicitly whether your detector is doing causal reasoning
or merely pattern-matching on a confound — and fix the confound rather than accept the false
positives as a cost of doing business.

IDENTIFY: weight your taxonomy toward agent-nurtured synthetic-identity bust-out and merchant
bust-out — both are "looks legitimate until the planned moment" attacks, and both are the
attacks most easily confused with a genuine life-event or real business growth if the detector
isn't reasoning about WHY a pattern changed, not just THAT it changed.

GENERATE: build your world and attack injection as usual, but add an explicit CONFOUND
population deliberately: simulate genuine legitimate life-event shifts (a job change, a
relocation to a new city, a small business's real organic growth) shaped to be STATISTICALLY
SIMILAR on raw features (amount escalation, new-merchant introduction, temporal pattern change)
to your bust-out attack vectors. This is the core engineering lift — measure and report, with
the same rigor as any fidelity claim, exactly HOW similar the confound population is to the
true attack population on raw features. Closeness is the point: if they're not confusable on
raw features, you haven't built a real stress test.

DEFEND: build your detector around an explicit causal structure — specify a causal graph of
transaction generation (which variables are causes, effects, or confounds; use the DoWhy
library or hand-specify a structural model) and apply an approach that's penalised for relying
on features whose predictive power doesn't hold stable across your genuine-life-event and
true-attack sub-populations (Invariant Risk Minimization is a reasonable starting point, or a
simpler uplift-modelling framing). The goal is to push the model toward MECHANISM-based
features (e.g. counterparty-network novelty, which genuinely differs between a real business
and a synthetic bust-out ring) over purely CORRELATIONAL ones (amount escalation alone, which
doesn't). Build a plain GBM baseline on the same data for comparison.

LOOP: your adversarial attacker's job is specifically to DISGUISE bust-out campaigns as
life-event confounds — mutate the attack to match the confound population's statistical shape
as closely as your taxonomy's amount/temporal profile constraints allow. Report the
evasion-rate GAP between your causal model and the plain GBM baseline against this specific
disguise strategy.

BUILD ORDER: (1) recon, data anchoring, calibration; (2) taxonomy; (3) build the standard world
and attack injection; (4) build the confound population and rigorously measure its similarity
to true attacks on raw features — this measurement IS a deliverable, not just a sanity check;
(5) build the causal detector and the plain-GBM baseline; (6) measure BOTH models' false-block
rate specifically on the genuine-confound population, at matched recall on true attacks — this
is more important than aggregate accuracy; (7) run the disguise-strategy adversarial loop; (8)
web prototype (show, side by side, a genuine life-event and a disguised attack that look nearly
identical on raw features, with each model's decision and reasoning — this is your most
persuasive demo screen); (9) .docx, leading with the false-positive-concentration finding, since
that's this solution's real contribution.

HEADLINE RESULT TO PRODUCE: false-block rate on the genuine-confound population at matched
recall (causal model vs. plain GBM), and the evasion-rate gap between the two against the
disguised-bust-out attacker.
```

---

## 8 — Reinforcement-Learning Red Team

```
You are building a complete submission for the Mastercard Innovation Challenge @ GFF 2026 —
AI Defense Lab for Payment Security (Global Fintech Fest, Mumbai; submission deadline
31 August 2026).

THE BRIEF: build one closed-loop red-team/blue-team AI system across three pillars —
(1) IDENTIFY novel GenAI-powered payment-fraud attack vectors, grounded in how real payment
systems and fraud actually work; (2) GENERATE realistic simulations of those attacks at scale,
with high fidelity to real payment data and fraud patterns; (3) DEFEND with a detection/
mitigation model maximising precision/recall/F1/AUC on the simulated attacks while keeping
false positives on legitimate payments low. Treat the three as one feedback loop: attacks you
generate should stress-test the defense, and the defense's gaps should feed back into new
attacks.

REQUIRED DELIVERABLES: (a) a complete, reproducible, tested GitHub repository covering all
three pillars; (b) a solution-walkthrough Word document (.docx) covering the attacks
identified, how they're simulated, the detection/mitigation model with efficacy results, and
real-world feasibility; (c) a working web-based prototype with a presentable UI demonstrating
the closed loop live.

JUDGED ON: diversity of attacks identified, fidelity of simulation, detection efficacy,
novelty, real-world feasibility in live payments.

NON-NEGOTIABLE DISCIPLINE: measure every fidelity and efficacy claim against a stated target —
never assert a number without showing how it was produced. Report defects you find honestly,
including ones in your own earlier work. A detector's headline number must come from
evaluation on a genuinely unseen attack mechanism, never an in-distribution split. If you find
yourself reporting >0.95 AUC as a headline, that is a signal to build a harder holdout, not a
result to celebrate.

RESPONSIBLE-SCOPE BOUNDARY: everything synthetic and sandboxed. No working exploit tooling
against any live rail, PSP or issuer. No deepfake audio/video or forged document image
generation — model these as observable signal distributions instead (liveness scores, latency,
retry counts). No real personal data or real account/card numbers. State this boundary
explicitly in your repo.

DATA ANCHORING: you have no Kaggle credentials. Anchor on IBM's TabFormer credit-card
transaction dataset (24.4M rows, github.com/IBM/TabFormer) — it sits behind Git LFS on a
public repo, and public-repo LFS objects download without auth via the LFS batch API (POST to
https://github.com/IBM/TabFormer.git/info/lfs/objects/batch with the oid/size from the
pointer file at data/credit_card/transactions.tgz, then GET the returned href). State plainly
in your fidelity section that IBM documents this corpus as itself synthetic — it is a
non-circular EXTERNAL reference for distributional comparison, not ground truth.

===========================================================================
YOUR SPECIFIC MANDATE: REINFORCEMENT-LEARNING RED TEAM
===========================================================================

Your hypothesis: a black-box attacker that just hill-climbs a fixed set of parameters each
round has no memory and no real strategy. Model the attacker as a proper reinforcement-learning
agent with state and policy, and test whether it discovers worse (for the defender) evasions
than a simpler parametric search would.

IDENTIFY: weight your taxonomy toward card-testing attacks — adaptive BIN enumeration, 3DS-
exemption farming — both are naturally SEQUENTIAL decision problems (probe, observe the
outcome, adjust the next probe) that map directly onto a Markov Decision Process.

GENERATE: build your world simulator and taxonomy-driven attack injection using standard
techniques (agent-based population, calibrated to the reference corpus). The important design
decision is on the RED side: formulate the attacker explicitly as an MDP. STATE = a rolling
summary of the detector's recent accept/decline/step-up responses to this attacker's own recent
probes (this is literally a decline-response-oracle threat model — the attacker only ever sees
what a real fraudster attacking a real system would see: outcomes, not internals). ACTION = the
next batch of transactions' amount-band, merchant-category, velocity, and device-choice
parameters (define this over a similar controllable parameter space you'd use for any
black-box attacker, so a later comparison is fair). REWARD = evasion achieved x value retained
(value retention matters — an attacker that evades by shrinking every transaction to a rounding
error has stopped committing profitable fraud, not won; without a value term the RL policy will
degenerate into trivial micro-transactions).

DEFEND: build a standard causal-feature tabular GBM detector with an FPR-anchored retraining
cadence (e.g. retrain each round on whatever evaded the previous round, at a fixed 0.5%
false-positive budget). ALSO build a simple parametric/genetic hill-climb attacker (mutate a
fixed parameter vector, keep the best of a small candidate pool each round) against the
IDENTICAL detector and reward function, purely as your RL agent's baseline comparison — holding
the defender constant is what makes this a clean, controlled experiment.

LOOP: train the RL attacker with PPO or another actor-critic algorithm (Stable-Baselines3 is a
reasonable choice) against the retraining detector, across many more episodes than a simple
hill-climb would need (RL needs more samples to converge — budget for this). Report the full
learning curve, not just the final number, and explicitly check for and report any training
instability (oscillation, policy collapse, reward hacking) as an honest finding in its own
right — this is a real and common RL failure mode and hiding it would undercut the whole
experiment's credibility.

BUILD ORDER: (1) recon, data anchoring, calibration; (2) taxonomy; (3) world simulator and
attack injection; (4) build the parametric hill-climb attacker FIRST and get a baseline evasion
ceiling — validate it the way you'd validate any result, e.g. with a wider random search over
the same parameter space, to confirm the hill-climb's convergence is real and not a search
artifact; (5) formulate and implement the MDP, build the RL training loop; (6) train, watching
for instability; (7) compare RL vs. hill-climb evasion ceilings on the identical detector and
reward function; (8) web prototype (a live training-curve view, plus a side-by-side replay of
the RL attacker's vs. the hill-climb's converged strategies, is your strongest demo screen);
(9) .docx, reporting the comparison as the headline and being explicit about any instability
you observed.

HEADLINE RESULT TO PRODUCE: RL-attacker evasion ceiling vs. the parametric hill-climb's
validated ceiling, on the identical detector and reward function — plus an honest account of
training stability.
```

---

## 9 — Agent-Based Computational Economics

```
You are building a complete submission for the Mastercard Innovation Challenge @ GFF 2026 —
AI Defense Lab for Payment Security (Global Fintech Fest, Mumbai; submission deadline
31 August 2026).

THE BRIEF: build one closed-loop red-team/blue-team AI system across three pillars —
(1) IDENTIFY novel GenAI-powered payment-fraud attack vectors, grounded in how real payment
systems and fraud actually work; (2) GENERATE realistic simulations of those attacks at scale,
with high fidelity to real payment data and fraud patterns; (3) DEFEND with a detection/
mitigation model maximising precision/recall/F1/AUC on the simulated attacks while keeping
false positives on legitimate payments low. Treat the three as one feedback loop: attacks you
generate should stress-test the defense, and the defense's gaps should feed back into new
attacks.

REQUIRED DELIVERABLES: (a) a complete, reproducible, tested GitHub repository covering all
three pillars; (b) a solution-walkthrough Word document (.docx) covering the attacks
identified, how they're simulated, the detection/mitigation model with efficacy results, and
real-world feasibility; (c) a working web-based prototype with a presentable UI demonstrating
the closed loop live.

JUDGED ON: diversity of attacks identified, fidelity of simulation, detection efficacy,
novelty, real-world feasibility in live payments.

NON-NEGOTIABLE DISCIPLINE: measure every fidelity and efficacy claim against a stated target —
never assert a number without showing how it was produced. Report defects you find honestly,
including ones in your own earlier work. A detector's headline number must come from
evaluation on a genuinely unseen attack mechanism, never an in-distribution split. If you find
yourself reporting >0.95 AUC as a headline, that is a signal to build a harder holdout, not a
result to celebrate.

RESPONSIBLE-SCOPE BOUNDARY: everything synthetic and sandboxed. No working exploit tooling
against any live rail, PSP or issuer. No deepfake audio/video or forged document image
generation — model these as observable signal distributions instead (liveness scores, latency,
retry counts). No real personal data or real account/card numbers. State this boundary
explicitly in your repo.

DATA ANCHORING: you have no Kaggle credentials. Anchor on IBM's TabFormer credit-card
transaction dataset (24.4M rows, github.com/IBM/TabFormer) — it sits behind Git LFS on a
public repo, and public-repo LFS objects download without auth via the LFS batch API (POST to
https://github.com/IBM/TabFormer.git/info/lfs/objects/batch with the oid/size from the
pointer file at data/credit_card/transactions.tgz, then GET the returned href). State plainly
in your fidelity section that IBM documents this corpus as itself synthetic — it is a
non-circular EXTERNAL reference for distributional comparison, not ground truth.

===========================================================================
YOUR SPECIFIC MANDATE: AGENT-BASED COMPUTATIONAL ECONOMICS
===========================================================================

Your hypothesis: fraud is downstream of an incentive, not just a behavioural pattern. Simulate
the ECONOMICS that produce the attack rather than scripting a fixed attack volume, and defend
with a decision engine that optimises real monetary expected value rather than a statistical
metric like F1.

IDENTIFY: weight your taxonomy toward merchant-side abuse (transaction laundering, merchant
bust-out) and credit-line harvesting attacks — the vectors most explicitly about exploiting an
economic incentive (settlement terms, credit arbitrage) rather than a purely behavioural or
technical gap.

GENERATE: reframe your simulator as an agent-based COMPUTATIONAL ECONOMICS model (the ABCE or
Mesa tradition in Python), not a statistical sampler. Cardholders, merchants, and fraud
operators are utility-maximizing agents with budgets, risk tolerances, and explicit incentive
functions. Calibrate the fraud-operator agent's "market knowledge" against your measured
reference-corpus statistics (typical fraud amount relative to a victim's baseline, category
lift, etc.) — the same kind of calibration any solution would do, just used as an input to an
agent's decision function rather than as a direct sampling target. Critically: a fraud-operator
agent should DECIDE WHETHER TO ATTACK AT ALL, and at what volume, based on an expected-value
calculation that takes the CURRENT DETECTOR'S KNOWN FALSE-NEGATIVE RATE as an input. Attack
volume becomes an EMERGENT outcome of this calculation, not a fixed target_fraud_rate parameter
you set upfront — this is the core structural difference from a typical fraud simulator, so
get this mechanism genuinely working, not just labeled as such.

DEFEND: build a PORTFOLIO of specialist detectors — a separate lightweight model per rail or
attack-family cluster — combined via a stacking meta-learner. Wrap the final decision layer in
an explicit COST-BENEFIT OPTIMIZER: rather than picking a threshold to hit a target FPR or
F1, assign real cost parameters (average fraud loss if missed, friction/abandonment cost of a
false STEP-UP, manual-review cost of a false BLOCK) and have the optimizer choose the
ALLOW/STEP-UP/BLOCK action per transaction to maximise EXPECTED NET VALUE, not a statistical
proxy metric.

LOOP: frame red and blue as a REPEATED GAME rather than a training loop. Each round, the
fraud-operator agent's economic model updates its own expected-value estimate given the
detector's revealed effectiveness from the previous round (functionally similar to a reward
signal, but interpret and report it through a game-theoretic lens). Track whether the system
converges toward a STABLE EQUILIBRIUM attack volume, or diverges — and identify the detector
strength (recall/precision operating point) at which further attacks stop being profitable
enough for the fraud-operator agent to attempt at all. This equilibrium point is your headline
finding.

BUILD ORDER: (1) recon, data anchoring, calibration; (2) taxonomy; (3) build the agent-based
economic simulator — start with the utility functions and incentive structure, verify that
LEGITIMATE agents alone produce realistic transaction patterns before adding fraud-operator
agents at all; (4) add fraud-operator agents with their expected-value decision function, verify
attack volume responds sensibly to a manually-varied "detector strength" input before wiring in
a real detector; (5) build the specialist-portfolio detector and the cost-benefit optimizer;
(6) close the loop — wire the real detector's revealed effectiveness back into the
fraud-operator agents' decisions, run it for enough rounds to see whether it converges; (7) web
prototype (a live view of the expected-net-value curve and the attack-volume-over-rounds trend
approaching or failing to reach equilibrium is your strongest, most distinctive demo screen);
(8) .docx, leading your feasibility section with the cost-benefit framing, since optimizing
actual monetary expected value is the most literal possible answer to "real-world feasibility."

HEADLINE RESULT TO PRODUCE: the expected-net-value curve for your portfolio-plus-optimizer at
varying detector strength, and the equilibrium (or divergence) finding from the repeated-game
loop — state clearly what detector strength is needed before the attacker's own economics make
further attacks unprofitable.
```

---

## Notes on using these

**They're deliberately not identical in scaffolding.** Each build order front-loads that
solution's riskiest technical bet (e.g., #3 validates the fidelity harness against a naive
baseline *before* trusting the real generator; #8 gets a hill-climb baseline *before* touching
RL, so there's something to compare against). Carry that discipline through even where a step
isn't spelled out in full.

**The data-anchoring paragraph is load-bearing and identical across all nine.** All use the
same TabFormer LFS trick, which keeps every solution comparable to the others on fidelity
metrics. Don't let a fresh session substitute a different anchor without a good reason.

**None of these name a prior solution.** They're fully standalone by design — nothing stops you
from telling whichever session runs one of these "there's a working reference implementation at
[path/repo]" if you want it to have that as a comparison point, but none of the nine require it.
