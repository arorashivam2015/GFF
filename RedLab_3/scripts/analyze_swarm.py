"""Analyse what the swarm run actually produced. With one successful
generation out of five attempts, a trained text classifier would be
meaningless - this reports the refusal pattern itself, which is the real
finding, plus a small structural comparison of the one generated artefact
against the four refusals."""
import json
import pathlib

campaigns = json.load(open("data/processed/generated_campaigns.json"))

refused = [c for c in campaigns if c["text"].strip().lower().startswith(
    ("i'm not going to", "i won't"))]
generated = [c for c in campaigns if c not in refused]

print(f"attempted: {len(campaigns)}  |  refused: {len(refused)}  |  generated: {len(generated)}")
print()
for c in refused:
    print(f"REFUSED  {c['vector_id']:12s} {c['vector_name']}")
    # Pull out the model's own suggested alternative, where it offered one.
    lines = [l.strip("- ") for l in c["text"].split("\n") if l.strip().startswith("-")]
    if lines:
        print(f"         suggested alternative(s): {lines[0]}")
print()
for c in generated:
    print(f"GENERATED  {c['vector_id']:12s} {c['vector_name']}")
    print(f"           target: {'human' if 'AGC' not in c['vector_id'] else 'AI agent, not a human'}")
    print(f"           length: {len(c['text'])} chars")

summary = {
    "n_attempted": len(campaigns),
    "n_refused": len(refused),
    "n_generated": len(generated),
    "refused_vectors": [c["vector_id"] for c in refused],
    "generated_vectors": [c["vector_id"] for c in generated],
}
pathlib.Path("artifacts").mkdir(exist_ok=True)
json.dump(summary, open("artifacts/swarm_summary.json", "w"), indent=1)
print(f"\n-> artifacts/swarm_summary.json")
