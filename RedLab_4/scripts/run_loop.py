"""Run the white-box evasion loop against the trained (frozen) autoencoder,
for each of the five synthetic-identity vectors this solution is weighted
toward."""
import json
import pathlib
import pickle
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from redlab.defend.anomaly_ae import Autoencoder
from redlab.loop.adversarial import run_evasion
from redlab.sim.cvae import ConditionalVAE
from redlab.taxonomy.loader import Taxonomy
from redlab.taxonomy.schema import Family

tax = Taxonomy.load()
vectors = [v.id for v in tax.by_family(Family.SYNTHETIC_IDENTITY)]
print(f"running evasion for: {vectors}", flush=True)

enc = pickle.load(open("artifacts/model/vae_encoding.pkl", "rb"))
generator = ConditionalVAE(n_mcc=len(enc.idx_to_mcc), n_channel=len(enc.idx_to_chan),
                          n_conditions=len(enc.idx_to_cond))
generator.load_state_dict(torch.load("artifacts/model/cvae.pt"))
generator.eval()

detector = Autoencoder(in_dim=66)
detector.load_state_dict(torch.load("artifacts/model/anomaly_ae.pt"))
detector.eval()

defend_eval = json.loads(pathlib.Path("artifacts/defend_ae_eval.json").read_text())
threshold = defend_eval["autoencoder"]["threshold"]
print(f"detector threshold: {threshold:.4f}", flush=True)

results = []
for vid in vectors:
    r = run_evasion(generator, detector, enc, condition=vid, threshold=threshold,
                    n_samples=500, steps=200, seed=0)
    print(f"  {vid}: evasion {r.evasion_rate_before*100:5.1f}% -> "
         f"{r.evasion_rate_after*100:5.1f}%  (drift={r.mean_drift:.3f})", flush=True)
    results.append({"vector_id": vid, "evasion_before": r.evasion_rate_before,
                    "evasion_after": r.evasion_rate_after, "mean_drift": r.mean_drift,
                    "rounds": r.rounds})

json.dump(results, open("artifacts/loop_eval.json", "w"), indent=1)
print("-> artifacts/loop_eval.json", flush=True)
