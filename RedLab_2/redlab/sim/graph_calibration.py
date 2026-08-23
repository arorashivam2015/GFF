"""Bipartite graph-structure targets, extracted from the reference corpus.

RedLab_1's conditional_profile.json carries summary statistics (Zipf alpha,
per-user distinct-merchant percentiles) sufficient to calibrate a tabular
generator. It does not carry the actual bipartite DEGREE DISTRIBUTIONS this
solution's graph-fidelity harness needs to validate against - the shape of
"how many distinct users does each merchant see," not just its Zipf slope.

Streamed once over the 24.4M-row reference corpus (~2000 distinct users
total, so per-entity adjacency sets stay small in memory throughout).
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict


class GraphCalibrationProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provenance: str
    n_users: int
    n_merchants: int
    n_edges: int
    user_degree_quantiles: Dict[str, float]     # distinct merchants per user
    merchant_degree_quantiles: Dict[str, float]  # distinct users per merchant
    mean_clustering_sample: float
    largest_component_share: float
    reciprocal_edge_share: float  # share of user-merchant pairs with >1 interaction

    @classmethod
    def load(cls, path: str) -> "GraphCalibrationProfile":
        return cls.model_validate(json.loads(Path(path).read_text()))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.model_dump(mode="json"), indent=1))


PCTS = [5, 10, 25, 50, 75, 90, 95, 99]


def extract(csv: Path, chunksize: int = 2_000_000, sample_merchants_for_clustering: int = 4000
           ) -> GraphCalibrationProfile:
    user_adj: Dict[int, set] = defaultdict(set)
    merch_adj: Dict[str, set] = defaultdict(set)
    pair_count: Dict[tuple, int] = defaultdict(int)
    n_rows = 0

    for ch in pd.read_csv(csv, usecols=["User", "Merchant Name"], chunksize=chunksize,
                          dtype={"Merchant Name": str, "User": "int32"}):
        n_rows += len(ch)
        for u, m in zip(ch["User"].to_numpy(), ch["Merchant Name"].to_numpy()):
            user_adj[u].add(m)
            merch_adj[m].add(u)

    # Pair-repeat share, sampled (full pair-count table over 24M rows is
    # unnecessary; a 2M-row sample is enough to estimate the repeat rate).
    sample = pd.read_csv(csv, usecols=["User", "Merchant Name"], nrows=2_000_000,
                         dtype={"Merchant Name": str, "User": "int32"})
    vc = sample.groupby(["User", "Merchant Name"]).size()
    reciprocal_share = float((vc > 1).mean())

    import networkx as nx
    rng = np.random.default_rng(0)
    sampled_merchants = rng.choice(list(merch_adj.keys()),
                                   size=min(sample_merchants_for_clustering, len(merch_adj)),
                                   replace=False)
    G = nx.Graph()
    for m in sampled_merchants:
        for u in merch_adj[m]:
            G.add_edge(f"U{u}", f"M{m}")
    # Clustering on a bipartite graph is 0 by definition (no triangles) unless
    # projected; project onto the merchant side via shared-user co-occurrence
    # for a meaningful clustering signal.
    from networkx.algorithms import bipartite
    merch_nodes = [n for n in G.nodes if n.startswith("M")]
    if len(merch_nodes) > 2:
        proj = bipartite.weighted_projected_graph(G, merch_nodes)
        clustering = float(np.mean(list(nx.clustering(proj).values()))) if proj.number_of_edges() else 0.0
        largest_cc = len(max(nx.connected_components(proj), key=len)) / max(proj.number_of_nodes(), 1)
    else:
        clustering, largest_cc = 0.0, 0.0

    udeg = np.array([len(v) for v in user_adj.values()], dtype=float)
    mdeg = np.array([len(v) for v in merch_adj.values()], dtype=float)

    return GraphCalibrationProfile(
        provenance="TabFormer reference corpus bipartite user-merchant graph "
                   "(synthetic; see calibration.py honesty note)",
        n_users=len(user_adj), n_merchants=len(merch_adj), n_edges=n_rows,
        user_degree_quantiles={f"p{p}": float(np.percentile(udeg, p)) for p in PCTS},
        merchant_degree_quantiles={f"p{p}": float(np.percentile(mdeg, p)) for p in PCTS},
        mean_clustering_sample=clustering,
        largest_component_share=largest_cc,
        reciprocal_edge_share=reciprocal_share,
    )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/raw/card_transaction.v1.csv")
    ap.add_argument("--out", default="data/processed/graph_calibration_profile.json")
    a = ap.parse_args()
    p = extract(Path(a.csv))
    p.save(a.out)
    print(f"wrote {a.out}")
    print(f"  users={p.n_users}  merchants={p.n_merchants}  edges={p.n_edges:,}")
    print(f"  user degree   p50={p.user_degree_quantiles['p50']:.0f} "
         f"p90={p.user_degree_quantiles['p90']:.0f} p99={p.user_degree_quantiles['p99']:.0f}")
    print(f"  merchant degree p50={p.merchant_degree_quantiles['p50']:.0f} "
         f"p90={p.merchant_degree_quantiles['p90']:.0f} p99={p.merchant_degree_quantiles['p99']:.0f}")
    print(f"  merchant-projection clustering={p.mean_clustering_sample:.4f}  "
         f"largest-component-share={p.largest_component_share:.3f}")
    print(f"  repeat-visit pair share={p.reciprocal_edge_share:.3f}")
