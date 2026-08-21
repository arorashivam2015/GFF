"""D2 adversarial curriculum: red team adapts, blue team retrains."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import json
import time

from redlab.loop.adversarial import run_curriculum
from redlab.sim.world import WorldConfig, build_world

t0 = time.time()
world = build_world(cfg=WorldConfig(n_cardholders=2500, n_merchants=5000,
                                    days=120, seed=42))
legit = world.generate()
print(f"world: {len(legit):,} legit txns  ({time.time()-t0:.0f}s)\n")
print("ADVERSARIAL CURRICULUM  (evasion measured at a 0.5% false-positive budget)")

results = run_curriculum(world, legit, rounds=5, candidates=4, seed=7)

print(f"\n{'round':>6}{'evasion':>10}{'value':>9}{'fitness':>10}{'recall':>9}")
for r in results:
    print(f"{r.round:>6}{r.evasion_rate*100:>9.1f}%{r.value_retention*100:>8.1f}%"
          f"{r.fitness:>10.3f}{r.detector_recall*100:>8.1f}%")

first, last = results[0], results[-1]
print(f"\nevasion {first.evasion_rate*100:.1f}% -> {last.evasion_rate*100:.1f}%"
      f"   over {len(results)} rounds")
print(f"attacker value retention at the end: {last.value_retention*100:.1f}%")
print(f"\nfinal genome: {json.dumps({k: round(v,3) for k,v in last.genome.items()})}")
json.dump([r.model_dump(mode="json") for r in results],
          open("artifacts/adversarial_loop.json", "w"), indent=1)
print(f"\n-> artifacts/adversarial_loop.json  ({time.time()-t0:.0f}s total)")
