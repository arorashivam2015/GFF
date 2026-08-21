# Measured Findings

A running log of results that were *measured*, with the defect each one
exposed. Source material for the solution walkthrough. Every number here is
reproducible from the scripts in `scripts/`.

---

## 1. Data anchoring

No public corpus of real payment authorisations is redistributable. We anchor on
**IBM's TabFormer credit-card dataset** (24.4M transactions, Padhi et al.,
ICASSP 2021), obtained without credentials via the public Git-LFS endpoint,
SHA-256 verified.

**IBM documents this corpus as synthetic.** It is used as an *externally
authored reference*, not as real data. Its value is that (a) we did not write
it, so comparing against it is not circular, and (b) it is published, so a third
party can reproduce the comparison.

Fidelity claims are therefore split into three tiers, each labelled:

| Tier | Basis | Strength |
|---|---|---|
| Stylised facts | Benford conformance, Zipf merchant concentration, CNP fraud lift — documented properties of *real* payment systems | Strongest |
| Reference divergence | KS / JSD against TabFormer | Medium — target is synthetic |
| Published aggregates | NPCI/RBI monthly figures for UPI rails | Marginals only |

Reference corpus properties: fraud rate **0.122%**, CNP fraud lift **11.3×**,
amounts log-normal μ=3.222 σ=1.376, merchant Zipf α=**1.873** (top 1% of
merchants = 80% of volume), Benford MAD **0.92pp**, median inter-transaction gap
**3.77h**.

---

## 2. Legitimate-population fidelity

Discriminator AUC (a classifier trained to separate generated rows from
reference rows; 0.5 = indistinguishable):

| Generator | AUC |
|---|---|
| Naive uniform | 0.977 |
| Marginals perfectly matched | 0.847 |
| Agent-based world | **0.665** |

### Finding 2.1 — Matching marginals is not fidelity

The marginals-matched generator achieved JSD ≈ 0.0000 on hour, MCC and channel
and KS 0.003 on amount, and remained separable at **0.847**. All the
separability lived in *joint* structure: amount independent of category, hour
independent of channel, merchant uncorrelated with category.

This is the failure mode a fit-a-GAN-and-ship submission cannot see, because it
only ever inspects marginals.

### Finding 2.2 — The reference is not log-normal

Fitting per-category log-normals reproduced log-standard-deviation to within
0.01 while overshooting p99 by **2–8×** (category 5499: $830 generated vs $106
reference). Reference log-amounts carry skew **−0.71** with a truncated upper
tail.

Replaced with empirical inverse-CDF sampling through a Gaussian copula, which
preserves the exact marginal *and* per-user spend persistence. Also added
round-amount snapping: the reference is **11.0%** whole-unit amounts, smooth
draws are 1.0%, and that discrepancy alone is detectable.

### Finding 2.3 — Zipf α is partly a sampling artifact

The identical generator measures α=**1.60** at 50 transactions per merchant and
α=**1.95** at 221. The reference sits at 243 transactions per merchant.

This briefly corrupted our own harness, which was scoring concentration on the
300k discriminator subsample and reporting 1.63 for a world that actually sits
at 1.94. **Merchant concentration cannot be validated on a subsample.**

---

## 3. Attack fidelity

The target is not "hard to detect" — it is "as detectable as reference fraud."
We train the same simple model on the reference corpus's own labels and on ours,
then compare separability signatures feature by feature.

### Finding 3.1 — Invented attack rules were measurably wrong

A first injection pass, built from plausible-sounding rules, scored ROC-AUC
**0.9886** with every one of 42 vectors at the ~100th score percentile.
Measuring the reference corpus's 28,619 labelled frauds showed exactly where:

| Property | Reference | First pass |
|---|---|---|
| Online share | 61.0% | 92.8% |
| Top-1 MCC share | 16.9% (98 MCCs) | 74.7% (one MCC) |
| Amount vs victim median | 2.4× (p50) | ~4.4× |
| **Exceeds victim's own max** | **0.74%** | **~40%** |

Real fraud hides *beneath* the victim's historical ceiling. Drawing fraud
amounts as a multiple of the victim's maximum is an artifact a detector learns
instead of the attack.

### Finding 3.2 — After calibration, signatures match

| Feature | Reference | Ours | Gap |
|---|---|---|---|
| ROC-AUC | 0.9660 | 0.9650 | −0.0010 |
| PR-AUC | 0.7809 | 0.7740 | −0.0068 |
| amount | 0.7291 | 0.7236 | −0.0055 |
| mcc | 0.8679 | 0.8357 | −0.0321 |
| channel | 0.7815 | 0.7495 | −0.0320 |
| dow | 0.5577 | 0.5117 | −0.0460 |

Verdict: **MATCHED**. Note the corollary — raw-feature fraud detection *is* easy
in this corpus, for reference and generated data alike. ROC-AUC is therefore
useless as a headline metric here.

---

## 4. The loop caught a simulator defect

First detector run returned **PR-AUC 1.0000, recall 100%**. Not a result — a
leak. Top feature by a 7× margin: `d_distinct_users_prior`.

Cause: every legitimate device in the world belonged to exactly one user
forever, while every attack device was shared across victims. Measured:

- legit devices: **max 1 user**, 0.00% shared
- attack devices: median 6 users, max 43
- `u_first_device`: 0.007 legit vs 0.704 fraud

"Device seen with more than one user" was a perfect fraud oracle — a property of
the simulator, not of payments.

Fixed on both sides. Legitimate world now has household sharing, handset churn
with activation dates, and public terminals (**4.7%** of devices serve >1 user;
**2.0%** of legitimate transactions are first-use-of-device). Attacks now
execute **38%** of events from the victim's own device, matching session
hijack, malware and on-device social engineering.

This is the closed loop doing its job: the defence exposed a fidelity defect
that no amount of inspecting the attack generator would have revealed.

---

## 5. Open limitations

- **The reference is synthetic.** Stated everywhere it matters. IEEE-CIS would
  strengthen every claim in §2 and §3 and drops in via `CalibrationProfile`.
- **UPI rails are not transaction-level calibrated.** No such corpus is public.
  Transfer-rail vectors are injected with transfer-like semantics into a
  card-calibrated world; only the marginal priors in `calibration.upi_prior()`
  are published-aggregate anchored, and they need refreshing before submission.
- **Leave-one-family-out may be weaker than it looks.** All 42 vectors are
  rendered by one generic engine, so holding out a *family label* holds out
  parameter combinations rather than mechanisms. See `scripts/ablation.py` for
  the harder holdout over generative axes.
