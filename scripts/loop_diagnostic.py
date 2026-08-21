"""Wide-net check: does a broader search find more evasion than the hill-climb did?"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from redlab.loop.adversarial import AttackGenome, _inject, _score_attack
from redlab.defend.detect import Detector, temporal_split
from redlab.defend.features import build_features
from redlab.sim.world import build_world, WorldConfig

world = build_world(cfg=WorldConfig(n_cardholders=2500, n_merchants=5000, days=120, seed=42))
legit = world.generate()

base_frame = _inject(world, legit, AttackGenome(), 7)
base_value = float(base_frame.loc[base_frame.is_fraud==1,"amount"].abs().sum())
feats = build_features(base_frame)
tr, te = temporal_split(feats)
det = Detector(n_estimators=250).fit(tr)
ev0, rec0, prec0 = _score_attack(det, base_frame)
print(f"baseline: evasion {ev0*100:.1f}%  recall {rec0*100:.1f}%\n")

rng = np.random.default_rng(99)
results = []
N = 25
for i in range(N):
    g = AttackGenome(
        amount_shift=rng.uniform(-0.4, 0.4),
        amount_width=np.exp(rng.uniform(np.log(0.4), np.log(1.6))),
        reuse_delta=rng.uniform(-0.4, 0.4),
        victim_device_share=rng.uniform(0.0, 0.95),
        events_scale=np.exp(rng.uniform(np.log(0.3), np.log(1.6))),
        burst_spread=np.exp(rng.uniform(np.log(0.5), np.log(3.0))),
    )
    frame = _inject(world, legit, g, 7000 + i)
    ev, rec, prec = _score_attack(det, frame)
    value = float(frame.loc[frame.is_fraud==1,"amount"].abs().sum())
    retention = value / base_value
    fitness = ev * min(retention, 1.5)
    results.append((fitness, ev, retention, g))

results.sort(key=lambda t: -t[0])
print(f"{'rank':<6}{'evasion':>9}{'value':>8}{'fitness':>9}")
for i,(fit,ev,ret,g) in enumerate(results[:8]):
    print(f"{i:<6}{ev*100:>8.1f}%{ret*100:>7.1f}%{fit:>9.3f}")
print(f"\nmax evasion found across {N} random genomes: {max(r[1] for r in results)*100:.1f}%")
print(f"mean: {np.mean([r[1] for r in results])*100:.1f}%  (hill-climb converged to 6.1%)")
print(f"\nbest genome: {results[0][3].as_dict()}")
