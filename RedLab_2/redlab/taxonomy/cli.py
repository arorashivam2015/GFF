"""Taxonomy CLI.

    python -m redlab.taxonomy.cli stats
    python -m redlab.taxonomy.cli matrix
    python -m redlab.taxonomy.cli top -n 12
    python -m redlab.taxonomy.cli show PF-AGC-001
    python -m redlab.taxonomy.cli split --holdout agentic_commerce anti_defense
"""

import argparse
import json
import sys
from typing import List

from .loader import Taxonomy, TaxonomyError
from .schema import COVERAGE_AXES, Family

BAR = "-" * 72


def _hdr(title: str) -> None:
    print(f"\n{title}\n{BAR}")


def cmd_stats(tax: Taxonomy, args: argparse.Namespace) -> None:
    s = tax.summary()
    _hdr(f"TAXONOMY: {s['total_vectors']} vectors")
    print(f"mean novelty score: {s['mean_novelty']}/5\n")

    print("by family")
    for fam, n in sorted(s["families"].items(), key=lambda kv: -kv[1]):
        print(f"  {fam:22s} {'#' * n} {n}")

    print("\nby maturity")
    for mat in ("observed", "emerging", "projected"):
        n = s["maturity"].get(mat, 0)
        print(f"  {mat:22s} {'#' * n} {n}")

    _hdr("AXIS COVERAGE (vectors touching each value)")
    for axis, counter in tax.axis_coverage().items():
        print(f"\n{axis}")
        for value, n in counter.most_common():
            print(f"  {value:34s} {'#' * n} {n}")

    gaps = s["gaps"]
    _hdr("UNCOVERED AXIS VALUES (ideation backlog)")
    if not gaps:
        print("  none - every enum value has at least one vector")
    else:
        for axis, missing in gaps.items():
            print(f"  {axis:18s} {', '.join(missing)}")


def cmd_matrix(tax: Taxonomy, args: argparse.Namespace) -> None:
    matrix = tax.family_rail_matrix()
    rails: List[str] = sorted({r for row in matrix.values() for r in row})
    width = max(len(f) for f in matrix) + 2

    _hdr("FAMILY x RAIL OCCUPANCY")
    print(" " * width + "".join(f"{r[:9]:>11s}" for r in rails))
    for fam in sorted(matrix):
        cells = "".join(
            f"{(str(matrix[fam].get(r)) if matrix[fam].get(r) else '.'):>11s}" for r in rails
        )
        print(f"{fam:{width}s}{cells}")
    print(f"\n{len(matrix)} families x {len(rails)} rails; "
          f"{sum(len(v) for v in matrix.values())} occupied cells")


def cmd_top(tax: Taxonomy, args: argparse.Namespace) -> None:
    _hdr(f"TOP {args.n} BY PRIORITY (simulation build order)")
    print(f"{'rank':<5}{'id':<12}{'prio':>6}  {'maturity':<10}{'name'}")
    for i, v in enumerate(tax.top(args.n), 1):
        print(f"{i:<5}{v.id:<12}{v.priority:>6.2f}  {v.maturity.value:<10}{v.name}")


def cmd_show(tax: Taxonomy, args: argparse.Namespace) -> None:
    try:
        v = tax[args.vector_id]
    except KeyError:
        sys.exit(f"unknown vector id: {args.vector_id}")
    if args.json:
        print(json.dumps(v.model_dump(mode="json"), indent=2))
        return
    _hdr(f"{v.id}  {v.name}")
    print(f"family      {v.family.value}")
    print(f"maturity    {v.maturity.value}")
    print(f"priority    {v.priority}  "
          f"(novelty {v.scores.novelty}, impact {v.scores.impact}, "
          f"scale {v.scores.scalability}, difficulty {v.scores.detection_difficulty})")
    print(f"\n{v.summary.strip()}\n")
    print(f"rails       {', '.join(r.value for r in v.rails)}")
    print(f"stages      {', '.join(s.value for s in v.stages)}")
    print(f"genai       {', '.join(g.value for g in v.genai_uplift)}")
    print(f"signals     {', '.join(s.value for s in v.signals)}")
    sim = v.simulation
    print(f"\nsimulation  entities={', '.join(sim.entities)}")
    print(f"            amount={sim.amount_profile.value} shape={sim.temporal_shape.value}")
    print(f"            {sim.duration_days[0]}-{sim.duration_days[1]} days, "
          f"{sim.events_per_campaign[0]}-{sim.events_per_campaign[1]} events, "
          f"reuse={sim.entity_reuse}")
    print("\ndetection hypotheses")
    for h in v.detection_hypotheses:
        print(f"  [{h.channel.value}] {h.feature}")
        print(f"      {h.rationale.strip()}")
    if v.mitigations:
        print("\nmitigations")
        for m in v.mitigations:
            print(f"  - {m}")


def cmd_split(tax: Taxonomy, args: argparse.Namespace) -> None:
    families = [Family(f) for f in args.holdout]
    train, holdout = tax.holdout_split(families)
    _hdr("LEAVE-ONE-FAMILY-OUT SPLIT")
    print(f"train    {len(train):3d} vectors")
    print(f"holdout  {len(holdout):3d} vectors  ({', '.join(args.holdout)})")
    print("\nholdout vectors (detector must never see these in training):")
    for v in holdout:
        print(f"  {v.id:<12} {v.name}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="redlab-taxonomy", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("stats", help="corpus summary, axis coverage and gaps")
    sub.add_parser("matrix", help="family x rail occupancy matrix")

    p_top = sub.add_parser("top", help="highest-priority vectors")
    p_top.add_argument("-n", type=int, default=10)

    p_show = sub.add_parser("show", help="full detail for one vector")
    p_show.add_argument("vector_id")
    p_show.add_argument("--json", action="store_true")

    p_split = sub.add_parser("split", help="leave-one-family-out train/holdout split")
    p_split.add_argument("--holdout", nargs="+", required=True,
                         choices=[f.value for f in Family])

    args = p.parse_args(argv)
    try:
        tax = Taxonomy.load()
    except TaxonomyError as exc:
        sys.exit(f"ERROR: {exc}")

    {"stats": cmd_stats, "matrix": cmd_matrix, "top": cmd_top,
     "show": cmd_show, "split": cmd_split}[args.cmd](tax, args)


if __name__ == "__main__":
    main()
