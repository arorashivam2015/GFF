"""Evaluate the world simulator against the reference corpus."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import json, sys, time, numpy as np, pandas as pd
from redlab.sim.calibration import CalibrationProfile
from redlab.sim.fidelity import evaluate
from redlab.sim.world import build_world, WorldConfig

prof = CalibrationProfile.model_validate(json.load(open("data/processed/reference_profile.json")))
ref = pd.read_parquet("data/interim/reference_sample.parquet")
ref_c = pd.DataFrame({
    "amount":  ref["Amount"].str.replace("$","",regex=False).astype(float),
    "hour":    ref["Time"].str.slice(0,2).astype(int),
    "mcc":     ref["MCC"].astype(str),
    "channel": ref["Use Chip"].astype(str),
    "merchant_id": ref["Merchant Name"].astype(str),
    "is_fraud": (ref["Is Fraud?"]=="Yes").astype(int)})
ref_c = ref_c[ref_c.amount>0].reset_index(drop=True)

seed = int(sys.argv[1]) if len(sys.argv)>1 else 42
t0=time.time()
w = build_world(cfg=WorldConfig(seed=seed))
df = w.generate()
print(f"generated {len(df):,} txns in {time.time()-t0:.1f}s "
      f"({df.user_id.nunique():,} users, {df.merchant_id.nunique():,} merchants, "
      f"{len(df)/df.merchant_id.nunique():.0f} txn/merchant)")
df.to_parquet("data/processed/world_legit.parquet", index=False)

gen = df[df.amount>0]
rep = evaluate(gen, prof, ref=ref_c,
               discriminator_features=["amount","hour","mcc","channel"])
print(rep.render())
json.dump(rep.model_dump(mode="json"), open("artifacts/fidelity_legit.json","w"), indent=1)
