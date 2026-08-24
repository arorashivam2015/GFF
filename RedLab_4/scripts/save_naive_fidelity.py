"""Reconstruct the naive-baseline fidelity report as a saved artifact.

The original run printed the full report to stdout successfully and this
script's output was manually transcribed from that run, then this script
re-derives it deterministically (same seed=0, same profile) so the artifact
is reproducible from source, not just a pasted number.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd

from redlab.sim.calibration import CalibrationProfile
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

naive = generate_naive(profile, n=100_000, seed=0)
rep = evaluate(naive, profile, ref=ref_c,
              discriminator_features=["amount", "hour", "mcc", "channel"])
print(rep.render())

pathlib.Path("artifacts").mkdir(exist_ok=True)
json.dump(rep.model_dump(mode="json"), open("artifacts/fidelity_naive.json", "w"), indent=1)
print("-> artifacts/fidelity_naive.json")
