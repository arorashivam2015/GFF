"""The cheap baseline: graph-derived FEATURES in a tabular model.

This is deliberately the same trick RedLab_1 used - distinct-counterparty
counts as a proxy for network structure - reimplemented here as the
comparison point for the real GNN in gnn.py. Every feature is causal (past
edges only), using the vectorised first-occurrence trick RedLab_1 validated:
a (src, dst) pair contributes to a running distinct-count exactly once, at
its first occurrence, so cumulative distinct-count is the cumsum of
first-occurrence flags, shifted to exclude the current edge.
"""

import numpy as np
import pandas as pd

WINDOW_SECONDS = 24 * 3600


def _prior_distinct(df: pd.DataFrame, key: str, target: str) -> np.ndarray:
    is_first = (df.groupby([key, target]).cumcount() == 0).astype(np.int32)
    cum = is_first.groupby(df[key]).cumsum()
    return (cum - is_first).to_numpy(dtype=np.int32)


def _prior_count(df: pd.DataFrame, key: str) -> np.ndarray:
    return df.groupby(key).cumcount().to_numpy(dtype=np.int32)


def _seconds_since_prior(df: pd.DataFrame, key: str) -> np.ndarray:
    return (df.groupby(key)["timestamp"].diff().dt.total_seconds()).to_numpy()


def _window_count(df: pd.DataFrame, key: str, seconds: int) -> np.ndarray:
    out = np.zeros(len(df), dtype=np.int32)
    ts = df["timestamp"].astype("int64").to_numpy() // 10**9
    for _, idx in df.groupby(key, sort=False).indices.items():
        t = ts[idx]
        order = np.argsort(t, kind="stable")
        t_sorted = t[order]
        left = np.searchsorted(t_sorted, t_sorted - seconds, side="left")
        counts = np.arange(len(t_sorted)) - left
        res = np.empty(len(t_sorted), dtype=np.int32)
        res[order] = counts
        out[idx] = res
    return out


def build_graph_proxy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Causal, hand-engineered graph-proxy features - the cheap alternative
    to a real GNN this solution benchmarks against."""
    d = df.sort_values("timestamp", kind="stable").reset_index(drop=True).copy()
    f = pd.DataFrame(index=d.index)

    f["amount"] = d["amount"]
    f["log_amount"] = np.log1p(d["amount"].clip(lower=0))

    # Fan-out proxy: how many distinct counterparties has this src touched.
    f["src_distinct_dst_prior"] = _prior_distinct(d, "src_id", "dst_id")
    f["src_prior_n"] = _prior_count(d, "src_id")
    f["src_secs_since_last"] = _seconds_since_prior(d, "src_id")
    f["src_edges_24h"] = _window_count(d, "src_id", WINDOW_SECONDS)

    # Fan-in proxy: how many distinct counterparties has this dst received from.
    f["dst_distinct_src_prior"] = _prior_distinct(d, "dst_id", "src_id")
    f["dst_prior_n"] = _prior_count(d, "dst_id")
    f["dst_secs_since_last"] = _seconds_since_prior(d, "dst_id")
    f["dst_edges_24h"] = _window_count(d, "dst_id", WINDOW_SECONDS)

    f["src_first_seen"] = (f["src_prior_n"] == 0).astype(np.int8)
    f["dst_first_seen"] = (f["dst_prior_n"] == 0).astype(np.int8)
    f["src_type"] = d["src_type"].astype("category")
    f["dst_type"] = d["dst_type"].astype("category")

    meta = d[["txn_id", "timestamp", "src_id", "dst_id", "is_fraud", "attack_id",
              "network_role", "motif_id"]].copy()
    return pd.concat([meta, f], axis=1)


def feature_names(frame: pd.DataFrame) -> list:
    drop = {"txn_id", "timestamp", "src_id", "dst_id", "is_fraud", "attack_id",
           "network_role", "motif_id"}
    return [c for c in frame.columns if c not in drop]
