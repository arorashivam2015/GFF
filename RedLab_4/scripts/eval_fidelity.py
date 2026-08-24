"""Score the naive baseline and the trained VAE against the identical
fidelity harness, and save both reports for direct comparison."""
import json
import pathlib
import pickle
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

from redlab.sim.calibration import CalibrationProfile
from redlab.sim.cvae import ConditionalVAE, sample
from redlab.sim.fidelity import evaluate
from redlab.sim.naive_generator import generate_naive

profile = CalibrationProfile.model_validate(
    json.loads(pathlib.Path("data/processed/reference_profile.json").read_text()))
ref = pd.read_parquet("data/interim/reference_sample.parquet")
ref_c = pd.DataFrame({
    "amount": ref["Amount"].str.replace("$", "", regex=False).astype(float),
    "hour": ref["Time"].str.slice(0, 2).astype(int),
    "mcc": ref["MCC"].astype(str),
    "channel": ref["Use Chip"].astype(str),
    "is_fraud": (ref["Is Fraud?"] == "Yes").astype(int),
})
ref_c = ref_c[ref_c.amount > 0].reset_index(drop=True)

print("=" * 70)
print("NAIVE BASELINE (independent marginal sampling)")
print("=" * 70)
naive = generate_naive(profile, n=100_000, seed=0)
rep_naive = evaluate(naive, profile, ref=ref_c,
                     discriminator_features=["amount", "hour", "mcc", "channel"])
print(rep_naive.render())

print("\n" + "=" * 70)
print("TRAINED CONDITIONAL VAE")
print("=" * 70)
enc = pickle.load(open("artifacts/model/vae_encoding.pkl", "rb"))
model = ConditionalVAE(n_mcc=len(enc.idx_to_mcc), n_channel=len(enc.idx_to_chan),
                       n_conditions=len(enc.idx_to_cond))
model.load_state_dict(torch.load("artifacts/model/cvae.pt"))
model.eval()
vae_legit = sample(model, enc, "LEGIT", n=100_000, seed=1)
rep_vae = evaluate(vae_legit, profile, ref=ref_c,
                   discriminator_features=["amount", "hour", "mcc", "channel"])
print(rep_vae.render())

print("\n" + "=" * 70)
print("HEADLINE COMPARISON")
print("=" * 70)
print(f"  naive baseline discriminator AUC: {rep_naive.discriminator_auc:.4f}")
print(f"  trained VAE discriminator AUC:    {rep_vae.discriminator_auc:.4f}")

pathlib.Path("artifacts").mkdir(exist_ok=True)
json.dump({
    "naive": rep_naive.model_dump(mode="json"),
    "vae": rep_vae.model_dump(mode="json"),
}, open("artifacts/fidelity_comparison.json", "w"), indent=1)
print("\n-> artifacts/fidelity_comparison.json")
