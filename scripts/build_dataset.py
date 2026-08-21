"""End-to-end data build: world -> attacks -> causal features.

    python3 scripts/build_dataset.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import time
import pandas as pd

from redlab.defend.features import build_features, feature_names
from redlab.sim.attacks import AttackConfig, inject_attacks
from redlab.sim.world import WorldConfig, build_world

t0 = time.time()
world = build_world(cfg=WorldConfig(seed=42))
legit = world.generate()
print(f"[{time.time()-t0:5.0f}s] world      {len(legit):>10,} legit txns")

combined, campaigns = inject_attacks(world, legit, cfg=AttackConfig(seed=1234))
print(f"[{time.time()-t0:5.0f}s] attacks    {int(combined.is_fraud.sum()):>10,} fraud "
      f"({100*combined.is_fraud.mean():.3f}%), {len(campaigns)} campaigns, "
      f"{combined.attack_id.nunique()}/42 vectors")

feats = build_features(combined)
print(f"[{time.time()-t0:5.0f}s] features   {len(feature_names(feats)):>10} causal features")

combined.to_parquet("data/processed/world_attacked.parquet", index=False)
feats.to_parquet("data/processed/features.parquet", index=False)
print(f"[{time.time()-t0:5.0f}s] wrote data/processed/{{world_attacked,features}}.parquet")
