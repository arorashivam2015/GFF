"""Print-quality charts for the RedLab_2 walkthrough, built only from
artifacts that were actually measured during this build."""
import pathlib, sys
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
    print("wrote", name)


# --- taxonomy: network_role composition -------------------------------------
from redlab.taxonomy.loader import Taxonomy
tax = Taxonomy.load()
from collections import Counter
roles = Counter(v.network_role.value for v in tax)
order = ["fan_in", "fan_out", "collection_point", "originator", "layering_hop"]
vals = [roles.get(r, 0) for r in order]
fig, ax = plt.subplots(figsize=(6.2, 3.0))
bars = ax.bar([r.replace("_", "\n") for r in order], vals, color=BLUE, width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.05, str(v), ha="center", fontweight="bold")
ax.set_ylabel("Tagged vectors")
ax.set_title(f"Graph-motif coverage: {sum(vals)} of {len(tax)} vectors carry a network_role")
save(fig, "network_role_composition.png")

# --- graph fidelity: measured vs target, where defined -----------------------
d = json.load(open("artifacts/graph_fidelity_legit.json"))
metrics = [m for m in d["metrics"] if m["target"] is not None]
labels = [m["name"].replace("_", "\n") for m in metrics]
measured = [m["value"] for m in metrics]
targets = [m["target"] for m in metrics]
verdicts = [m["verdict"] for m in metrics]
colors = {"pass": GREEN, "warn": AMBER, "fail": RED}
x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(7.6, 3.6))
# Normalise each metric to its own target=1.0 so wildly different scales
# (degree counts vs clustering coefficients) share one axis honestly.
ratio = [m_/t_ if t_ else 0 for m_, t_ in zip(measured, targets)]
bars = ax.bar(x, ratio, color=[colors[v] for v in verdicts], width=0.6)
ax.axhline(1.0, color="#555", linestyle="--", linewidth=1)
ax.text(len(labels)-0.5, 1.03, "target", fontsize=8.5, color="#555", ha="right")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("measured / target")
ax.set_title("Graph-structure fidelity: legitimate population vs. reference corpus")
save(fig, "graph_fidelity.png")

print(f"\n{len(list(OUT.glob('*.png')))} charts written")
