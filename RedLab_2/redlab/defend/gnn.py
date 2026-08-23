"""The real bet: a temporal-ish Graph Neural Network scoring edges directly.

Design, and why it's built this way:

  CAUSALITY: node embeddings are computed by message-passing over the TRAIN
  graph only, then FROZEN. Test edges are scored using those frozen
  embeddings - no test-period structure ever reaches the embeddings used to
  score test edges. This is the graph analogue of RedLab_1's causal-feature
  discipline (features computed only from an entity's past).

  COLD START IS REAL, NOT HIDDEN: every attack campaign in this solution
  mints fresh mule-account IDs (see graph_attacks.py). Many test-period mule
  nodes therefore never appear in the train graph at all. Rather than
  quietly failing on them, unseen nodes get a single trainable "unknown
  node" embedding vector - the model's learned prior for "a node I have
  never seen." Whether that prior is informative, or whether cold-start
  nodes are simply hard, is exactly the kind of thing the evaluation in
  train_gnn.py should surface honestly, not hide behind an averaged metric.

  OPERATIONAL ASYMMETRY (worth stating in the walkthrough): the tabular
  graph-proxy baseline in graph_features.py updates online, per-transaction,
  with zero retraining. This GNN requires a batch graph-construction step
  before it can score anything past its training graph's frontier. That
  latency/freshness trade-off is real and belongs in the feasibility section,
  not just the accuracy comparison.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

NODE_TYPES = ["user", "merchant", "mule", "attacker"]
TYPE_IDX = {t: i for i, t in enumerate(NODE_TYPES)}


def build_node_index(df: pd.DataFrame) -> Tuple[Dict[str, int], np.ndarray]:
    """Integer index over every node id in the FULL edge set (train+test),
    so test-only nodes have a slot to look up even though they were never
    embedded. Node TYPE is a static attribute, safe to know upfront: it
    describes what kind of entity a node is, not what it did over time."""
    ids = pd.unique(pd.concat([df.src_id, df.dst_id], ignore_index=True))
    idx = {v: i for i, v in enumerate(ids)}

    type_of: Dict[str, str] = {}
    for col_id, col_type in [("src_id", "src_type"), ("dst_id", "dst_type")]:
        for i, t in zip(df[col_id].to_numpy(), df[col_type].to_numpy()):
            type_of.setdefault(i, t)
    types = np.array([TYPE_IDX.get(type_of.get(v, "user"), 0) for v in ids], dtype=np.int64)
    return idx, types


class GraphSAGEEncoder(nn.Module):
    def __init__(self, n_nodes: int, n_types: int, hidden: int = 32, out: int = 24):
        super().__init__()
        self.type_emb = nn.Embedding(n_types, 8)
        self.deg_proj = nn.Linear(1, 8)
        self.conv1 = SAGEConv(16, hidden)
        self.conv2 = SAGEConv(hidden, out)
        self.unknown = nn.Parameter(torch.randn(out) * 0.1)

    def input_features(self, node_types: torch.Tensor, log_degree: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.type_emb(node_types), self.deg_proj(log_degree.unsqueeze(-1))], dim=-1)

    def forward(self, node_types: torch.Tensor, log_degree: torch.Tensor,
               edge_index: torch.Tensor) -> torch.Tensor:
        x = self.input_features(node_types, log_degree)
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class EdgeScorer(nn.Module):
    def __init__(self, emb_dim: int = 24, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 2 + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1))

    def forward(self, e_src: torch.Tensor, e_dst: torch.Tensor,
               log_amount: torch.Tensor) -> torch.Tensor:
        x = torch.cat([e_src, e_dst, log_amount.unsqueeze(-1)], dim=-1)
        return self.net(x).squeeze(-1)


@dataclass
class GraphDetector:
    hidden: int = 32
    emb_dim: int = 24
    epochs: int = 12
    lr: float = 0.01
    seed: int = 0
    encoder: nn.Module = field(default=None, init=False)
    scorer: nn.Module = field(default=None, init=False)
    node_idx: Dict[str, int] = field(default=None, init=False)
    known_train_nodes: set = field(default=None, init=False)
    node_types_t: torch.Tensor = field(default=None, init=False)
    log_degree_t: torch.Tensor = field(default=None, init=False)
    train_edge_index: torch.Tensor = field(default=None, init=False)

    def _prep_edges(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Vectorised lookup via pandas .map (returns NaN for unseen ids,
        # filled with -1) rather than a per-row Python comprehension.
        src = torch.tensor(df.src_id.map(self.node_idx).fillna(-1).to_numpy(dtype=np.int64),
                           dtype=torch.long)
        dst = torch.tensor(df.dst_id.map(self.node_idx).fillna(-1).to_numpy(dtype=np.int64),
                           dtype=torch.long)
        log_amt = torch.tensor(np.log1p(df.amount.clip(lower=0).to_numpy()), dtype=torch.float32)
        return src, dst, log_amt

    def fit(self, full_df: pd.DataFrame, train_df: pd.DataFrame) -> "GraphDetector":
        torch.manual_seed(self.seed)
        self.node_idx, node_types = build_node_index(full_df)
        n_nodes = len(self.node_idx)

        # Degree computed from TRAIN edges only - a structural feature, but
        # one only known up to the training frontier. Vectorised via
        # np.add.at rather than a per-row Python loop.
        deg = np.zeros(n_nodes, dtype=np.float32)
        src_i = train_df.src_id.map(self.node_idx).to_numpy()
        dst_i = train_df.dst_id.map(self.node_idx).to_numpy()
        np.add.at(deg, src_i, 1)
        np.add.at(deg, dst_i, 1)
        self.log_degree_t = torch.tensor(np.log1p(deg), dtype=torch.float32)
        self.node_types_t = torch.tensor(node_types, dtype=torch.long)

        self.known_train_nodes = set(train_df.src_id) | set(train_df.dst_id)
        src, dst, _ = self._prep_edges(train_df)
        mask = (src >= 0) & (dst >= 0)
        self.train_edge_index = torch.stack([src[mask], dst[mask]], dim=0)
        # Undirected message passing: fraud rings are informative in both directions.
        self.train_edge_index = torch.cat(
            [self.train_edge_index, self.train_edge_index.flip(0)], dim=1)

        self.encoder = GraphSAGEEncoder(n_nodes, len(NODE_TYPES), self.hidden, self.emb_dim)
        self.scorer = EdgeScorer(self.emb_dim)
        opt = torch.optim.Adam(list(self.encoder.parameters()) + list(self.scorer.parameters()),
                               lr=self.lr)

        tr_src, tr_dst, tr_amt = self._prep_edges(train_df)
        y = torch.tensor(train_df.is_fraud.to_numpy(), dtype=torch.float32)
        valid = (tr_src >= 0) & (tr_dst >= 0)
        tr_src, tr_dst, tr_amt, y = tr_src[valid], tr_dst[valid], tr_amt[valid], y[valid]
        pos_weight = torch.tensor((y == 0).sum() / max((y == 1).sum(), 1))

        for epoch in range(self.epochs):
            opt.zero_grad()
            emb = self.encoder(self.node_types_t, self.log_degree_t, self.train_edge_index)
            unk = self.encoder.unknown
            e_src = torch.where((tr_src >= 0).unsqueeze(-1), emb[tr_src.clamp(min=0)], unk)
            e_dst = torch.where((tr_dst >= 0).unsqueeze(-1), emb[tr_dst.clamp(min=0)], unk)
            logits = self.scorer(e_src, e_dst, tr_amt)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            loss.backward()
            opt.step()
        return self

    @torch.no_grad()
    def score(self, df: pd.DataFrame) -> np.ndarray:
        self.encoder.eval()
        emb = self.encoder(self.node_types_t, self.log_degree_t, self.train_edge_index)
        unk = self.encoder.unknown
        src, dst, amt = self._prep_edges(df)
        e_src = torch.where((src >= 0).unsqueeze(-1), emb[src.clamp(min=0)], unk)
        e_dst = torch.where((dst >= 0).unsqueeze(-1), emb[dst.clamp(min=0)], unk)
        logits = self.scorer(e_src, e_dst, amt)
        return torch.sigmoid(logits).numpy()

    def cold_start_mask(self, df: pd.DataFrame) -> np.ndarray:
        """True where NEITHER endpoint was seen during training - the honest
        cold-start slice, reported separately rather than averaged away."""
        known = self.known_train_nodes
        src_known = df.src_id.isin(known).to_numpy()
        dst_known = df.dst_id.isin(known).to_numpy()
        return ~(src_known | dst_known)
