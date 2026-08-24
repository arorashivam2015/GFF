"""Print-quality charts, built only from artifacts actually measured during
this build."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

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
    print("wrote", name)


# --- Chart 1: fidelity - naive vs VAE discriminator AUC ---------------------
naive = json.loads(pathlib.Path("artifacts/fidelity_naive.json").read_text())
vae = json.loads(pathlib.Path("artifacts/fidelity_vae.json").read_text())
d_naive = next(m for m in naive["metrics"] if m["name"] == "discriminator_auc")["value"]
d_vae = next(m for m in vae["metrics"] if m["name"] == "discriminator_auc")["value"]

fig, ax = plt.subplots(figsize=(5.6, 3.0))
bars = ax.bar(["Naive baseline\n(independent marginals)", "Trained conditional VAE"],
              [d_naive, d_vae], color=[RED, BLUE], width=0.5)
ax.axhline(0.5, color="#555", linestyle="--", linewidth=1)
ax.text(1.3, 0.52, "0.5 = indistinguishable", fontsize=8.5, color="#555")
for b, v in zip(bars, [d_naive, d_vae]):
    ax.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.3f}", ha="center", fontweight="bold")
ax.set_ylabel("Discriminator AUC")
ax.set_ylim(0, 1.05)
ax.set_title("Fidelity: does a trained generator beat the naive baseline?")
save(fig, "fidelity_comparison.png")

# --- Chart 2: per-metric fidelity comparison --------------------------------
common = ["amount_percentile_log_rmse", "hour_of_day_jsd", "mcc_mix_jsd",
         "channel_mix_jsd", "benford_mad_pp"]
naive_v = [next(m["value"] for m in naive["metrics"] if m["name"] == k) for k in common]
vae_v = [next(m["value"] for m in vae["metrics"] if m["name"] == k) for k in common]
x = np.arange(len(common))
fig, ax = plt.subplots(figsize=(6.4, 3.2))
w = 0.36
ax.bar(x - w/2, naive_v, w, label="Naive", color=RED)
ax.bar(x + w/2, vae_v, w, label="VAE", color=BLUE)
ax.set_xticks(x)
ax.set_xticklabels([c.replace("_", "\n") for c in common], fontsize=8)
ax.set_ylabel("value (lower = closer to reference)")
ax.legend(fontsize=9, frameon=False)
ax.set_title("Per-metric fidelity: naive vs. trained VAE")
save(fig, "fidelity_per_metric.png")

# --- Chart 3: detection efficacy, AE vs GBM ---------------------------------
ae = json.loads(pathlib.Path("artifacts/defend_ae_eval.json").read_text())["autoencoder"]
gbm = json.loads(pathlib.Path("artifacts/defend_gbm_eval.json").read_text())["supervised"]
fig, ax = plt.subplots(figsize=(6.0, 3.2))
labels = ["ROC-AUC", "PR-AUC", "Recall @ 0.5% FPR"]
ae_v = [ae["roc_auc"], ae["pr_auc"], ae["recall_at_0.5fpr"]]
gbm_v = [gbm["roc_auc"], gbm["pr_auc"], gbm["recall_at_0.5fpr"]]
x = np.arange(3)
w = 0.36
b1 = ax.bar(x - w/2, ae_v, w, label="Autoencoder (zero labels)", color=AMBER)
b2 = ax.bar(x + w/2, gbm_v, w, label="Supervised GBM", color=GREEN)
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f"{b.get_height():.2f}",
               ha="center", fontsize=8.5, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.15)
ax.legend(fontsize=9, frameon=False, loc="upper left")
ax.set_title("Detection on never-trained-on (projected) mechanisms")
save(fig, "defend_comparison.png")

# --- Chart 4: evasion before/after, per vector ------------------------------
loop = json.loads(pathlib.Path("artifacts/loop_eval.json").read_text())
vids = [r["vector_id"] for r in loop]
before = [r["evasion_before"] * 100 for r in loop]
after = [r["evasion_after"] * 100 for r in loop]
x = np.arange(len(vids))
fig, ax = plt.subplots(figsize=(6.6, 3.2))
w = 0.36
ax.bar(x - w/2, before, w, label="Before white-box fine-tuning", color=AMBER)
ax.bar(x + w/2, after, w, label="After white-box fine-tuning", color=RED)
ax.set_xticks(x)
ax.set_xticklabels(vids, rotation=20, fontsize=8.5)
ax.set_ylabel("Evasion rate (%)")
ax.set_ylim(0, 105)
ax.legend(fontsize=9, frameon=False, loc="lower right")
ax.set_title("White-box evasion: gradient access closes most of the remaining gap")
save(fig, "loop_evasion.png")

print(f"\n{len(list(OUT.glob('*.png')))} charts written")
