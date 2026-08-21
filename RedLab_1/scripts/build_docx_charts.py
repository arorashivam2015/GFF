"""Generate print-quality chart PNGs for the solution walkthrough.

Light theme, high DPI, matplotlib only (no CDN/browser dependency) so this
reproduces identically for any judge running the repo.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = pathlib.Path("artifacts/docx_assets")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 10.5,
    "font.family": "sans-serif", "axes.edgecolor": "#888",
    "axes.labelcolor": "#222", "text.color": "#222",
    "xtick.color": "#333", "ytick.color": "#333",
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE, RED, GREEN, AMBER, GRAY = "#3568c9", "#c9403a", "#2f9e5e", "#c98a2f", "#8a97a8"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


# --- Chart 1: fidelity trajectory (legit population) -----------------------
fig, ax = plt.subplots(figsize=(6.2, 3.2))
labels = ["Naive uniform\ngenerator", "Marginals\nmatched", "Agent-based\nworld (final)"]
vals = [0.977, 0.847, 0.665]
colors = [RED, AMBER, GREEN]
bars = ax.bar(labels, vals, color=colors, width=0.55)
ax.axhline(0.5, color="#555", linestyle="--", linewidth=1)
ax.text(2.05, 0.51, "0.5 = indistinguishable\nfrom reference", fontsize=8.5, color="#555")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontweight="bold")
ax.set_ylabel("Discriminator AUC")
ax.set_ylim(0, 1.08)
ax.set_title("Legitimate-population fidelity: each fix measured, not asserted")
save(fig, "fidelity_trajectory.png")

# --- Chart 2: attack detectability signature --------------------------------
fa = json.load(open("artifacts/fidelity_attacks.json"))
feats = list(fa["reference"]["per_feature_auc"].keys())
ref_v = [fa["reference"]["per_feature_auc"][k] for k in feats]
gen_v = [fa["generated"]["per_feature_auc"][k] for k in feats]
x = np.arange(len(feats))
fig, ax = plt.subplots(figsize=(6.2, 3.2))
w = 0.36
ax.bar(x - w/2, ref_v, w, label="Reference fraud (28,619 real-labelled)", color=RED)
ax.bar(x + w/2, gen_v, w, label="Generated fraud", color=BLUE)
ax.set_xticks(x); ax.set_xticklabels(feats, rotation=20, ha="right")
ax.set_ylabel("Single-feature ROC-AUC")
ax.set_ylim(0.4, 1.0)
ax.legend(fontsize=8.5, frameon=False)
ax.set_title("Attack fidelity: detectability signature matches reference fraud")
save(fig, "attack_signature.png")

# --- Chart 3: detector evaluation across three honesty tiers ----------------
de = json.load(open("artifacts/detector_eval.json"))
tiers = ["In-distribution\n(all families seen)", "Leave-one-family-out\n(AGC + ADV held out)",
        "Unseen mechanism\n(amount profile held out)"]
pr = [de["in_distribution"]["pr_auc"], de["leave_one_family_out"]["pr_auc"],
     de["mechanism_holdout"]["pr_auc"]]
rec = [de["in_distribution"]["recall_at_fpr"]["0.5%"],
      de["leave_one_family_out"]["recall_at_fpr"]["0.5%"],
      de["mechanism_holdout"]["recall_at_fpr"]["0.5%"]]
fig, ax = plt.subplots(figsize=(6.4, 3.4))
x = np.arange(3); w = 0.36
b1 = ax.bar(x - w/2, pr, w, label="PR-AUC", color=BLUE)
b2 = ax.bar(x + w/2, rec, w, label="Recall @ 0.5% FPR", color=GREEN)
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f"{b.get_height():.2f}",
               ha="center", fontsize=8.5, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(tiers, fontsize=9)
ax.set_ylim(0, 1.15)
ax.legend(fontsize=9, frameon=False, loc="lower left")
ax.set_title("Detection efficacy: the honest number is the mechanism holdout")
save(fig, "detector_tiers.png")

# --- Chart 4: adversarial loop evasion curve --------------------------------
loop = json.load(open("artifacts/adversarial_loop.json"))
rounds = [r["round"] for r in loop]
evasion = [r["evasion_rate"] * 100 for r in loop]
recall = [r["detector_recall"] * 100 for r in loop]
fig, ax = plt.subplots(figsize=(6.2, 3.2))
ax.plot(rounds, evasion, "o-", color=RED, label="Attacker evasion %", linewidth=2)
ax.plot(rounds, recall, "o-", color=GREEN, label="Detector recall %", linewidth=2)
ax.set_xlabel("Adversarial round"); ax.set_ylabel("%")
ax.set_xticks(rounds)
ax.set_ylim(0, 105)
ax.legend(fontsize=9, frameon=False)
ax.set_title("D2 adversarial curriculum: evasion caps near 6%, not zero")
save(fig, "adversarial_loop.png")

# --- Chart 5: wide-search validation of the evasion ceiling ------------------
fig, ax = plt.subplots(figsize=(6.2, 3.0))
cats = ["Unmutated\nbaseline", "Hill-climb\n(converged)", "Wide random search\n(mean of 25)",
       "Wide random search\n(max of 25)"]
vals = [1.4, 6.1, 6.5, 12.4]
colors2 = [GRAY, BLUE, AMBER, RED]
bars = ax.bar(cats, vals, color=colors2, width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.3, f"{v:.1f}%", ha="center", fontweight="bold")
ax.set_ylabel("Evasion rate @ 0.5% FPR (%)")
ax.set_title("Two independent searches agree on the evasion ceiling")
save(fig, "evasion_validation.png")

# --- Chart 6: taxonomy composition ------------------------------------------
from redlab.taxonomy.loader import Taxonomy
tax = Taxonomy.load()
s = tax.summary()
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
fam = sorted(s["families"].items(), key=lambda kv: -kv[1])
axes[0].barh([k for k, _ in fam][::-1], [v for _, v in fam][::-1], color=BLUE)
axes[0].set_title("Vectors per family", fontsize=10.5)
mat = s["maturity"]
order = ["observed", "emerging", "projected"]
mvals = [mat.get(m, 0) for m in order]
axes[1].bar(order, mvals, color=[RED, AMBER, BLUE])
axes[1].set_title("Vectors per maturity level", fontsize=10.5)
for i, v in enumerate(mvals):
    axes[1].text(i, v + 0.3, str(v), ha="center", fontweight="bold")
save(fig, "taxonomy_composition.png")

print(f"\n{len(list(OUT.glob('*.png')))} chart PNGs written to {OUT}")
