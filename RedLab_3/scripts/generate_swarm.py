"""Generate a small, deliberately scoped set of LLM-authored campaign
scripts - five vectors, one call each, not the full taxonomy. Kept small on
purpose: this solution's point is demonstrating the mechanism (language-
level generation feeding a detector), not maximising volume."""
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from redlab.sim.llm_swarm import generate_swarm
from redlab.taxonomy.loader import Taxonomy

SELECTED = ["PF-SE-001", "PF-SE-002", "PF-SE-003", "PF-AGC-001", "PF-AGC-002"]

tax = Taxonomy.load()
vectors = [tax[vid] for vid in SELECTED]

t0 = time.time()
campaigns = generate_swarm(vectors)
print(f"\ngenerated {len(campaigns)} campaigns in {time.time()-t0:.0f}s")

out = [{"vector_id": c.vector_id, "vector_name": c.vector_name, "text": c.text}
      for c in campaigns]
pathlib.Path("data/processed").mkdir(parents=True, exist_ok=True)
json.dump(out, open("data/processed/generated_campaigns.json", "w"), indent=2)
print("-> data/processed/generated_campaigns.json")

for c in campaigns:
    print(f"\n--- {c.vector_id} ({c.vector_name}) ---")
    print(c.text)
