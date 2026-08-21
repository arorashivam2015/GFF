"""Train the detector and report the honest evaluation."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import json
import pandas as pd

from redlab.defend.detect import (Detector, leave_one_family_out,
                                  mechanism_holdout, temporal_split)
from redlab.taxonomy.loader import Taxonomy

F = pd.read_parquet("data/processed/features.parquet")
print(f"loaded {len(F):,} rows, {int(F.is_fraud.sum()):,} fraud")

print("\n" + "="*78 + "\nIN-DISTRIBUTION (temporal split, all families seen in training)\n" + "="*78)
tr, te = temporal_split(F)
d = Detector().fit(tr)
r_in = d.evaluate(te, "in-distribution", per_vector=True)
print(r_in.render())
print("\n  top features by gain:")
for k, v in d.importances(10).items():
    print(f"    {k:26s} {v:12.0f}")

print("\n" + "="*78 + "\nLEAVE-ONE-FAMILY-OUT (agentic_commerce + anti_defense never seen)\n" + "="*78)
tr2, te2 = leave_one_family_out(F, ["AGC", "ADV"])
print(f"  train fraud {int(tr2.is_fraud.sum()):,} | test fraud {int(te2.is_fraud.sum()):,}")
d2 = Detector().fit(tr2)
r_out = d2.evaluate(te2, "unseen families (AGC + ADV)", per_vector=True)
print(r_out.render())

print("\n" + "="*78 + "\nMECHANISM HOLDOUT (unseen generative axis - the honest test)\n" + "="*78)
tax = Taxonomy.load()
tr3, te3 = mechanism_holdout(F, tax, "amount_profile", ["micro_probe", "drain"])
print(f"  train fraud {int(tr3.is_fraud.sum()):,} | test fraud {int(te3.is_fraud.sum()):,}")
d3 = Detector().fit(tr3)
r_mech = d3.evaluate(te3, "unseen amount mechanisms (micro_probe + drain)")
print(r_mech.render())

print("\n" + "="*78)
print(f"GENERALISATION GAP  PR-AUC {r_in.pr_auc:.4f} -> {r_out.pr_auc:.4f} "
      f"({r_out.pr_auc - r_in.pr_auc:+.4f})")
print(f"                    recall@0.5%FPR {r_in.recall_at_fpr['0.5%']*100:.1f}% -> "
      f"{r_out.recall_at_fpr['0.5%']*100:.1f}%")
print("="*78)

print(f"  unseen MECHANISM   PR-AUC {r_mech.pr_auc:.4f}  "
      f"recall@0.5%FPR {r_mech.recall_at_fpr['0.5%']*100:.1f}%   <-- headline")
json.dump({"in_distribution": r_in.model_dump(mode="json"),
           "leave_one_family_out": r_out.model_dump(mode="json"),
           "mechanism_holdout": r_mech.model_dump(mode="json")},
          open("artifacts/detector_eval.json", "w"), indent=1)
print("\n-> artifacts/detector_eval.json")
