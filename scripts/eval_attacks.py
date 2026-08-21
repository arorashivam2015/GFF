"""Score attack fidelity against the reference corpus's own fraud."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import json, pandas as pd
from redlab.sim.attack_fidelity import compare

ref = pd.read_parquet("data/interim/reference_sample.parquet")
R = pd.DataFrame({
 "amount": ref["Amount"].str.replace("$","",regex=False).astype(float),
 "hour":   ref["Time"].str.slice(0,2).astype(int),
 "dow":    pd.to_datetime(dict(year=ref.Year,month=ref.Month,day=ref.Day),
                          errors="coerce").dt.dayofweek,
 "mcc":    ref["MCC"].astype(str),
 "channel":ref["Use Chip"].astype(str),
 "is_fraud":(ref["Is Fraud?"]=="Yes").astype(int)}).dropna()

G = pd.read_parquet("data/processed/world_attacked.parquet")
rep = compare(G, R)
print(rep.render())
json.dump(rep.model_dump(mode="json"), open("artifacts/fidelity_attacks.json","w"), indent=1)
