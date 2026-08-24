"""Causal feature engineering for the detector.

EVERY feature here is computed from a transaction's PAST ONLY. This is the
single most important property in the module: a fraud model trained on features
that peek at the current or future rows scores beautifully offline and fails in
production, and the failure is invisible unless causality is enforced by
construction.

Concretely, "the user's average amount" must mean the average of their PRIOR
transactions, excluding this one. Computing it with a plain groupby mean leaks
the current amount into its own feature, and leaks fraud from later in a
campaign backwards into earlier rows of the same campaign.

Feature families, chosen against the detection_hypotheses declared in the
taxonomy:

  BASELINE DEVIATION  amount relative to the user's own prior behaviour. The
                      taxonomy's amount profiles are defined relative to victim
                      baselines, so absolute amount is deliberately weak.
  VELOCITY            counts and sums over trailing time windows. Targets the
                      burst and slow-drip temporal shapes.
  NOVELTY             first-time merchant / category / device / geography for
                      this user. Targets acquisition and execution stages.
  ENTITY SHARING      how many distinct users a device or merchant touches.
                      This is the cheap proxy for the graph signal that mule
                      networks and card-testing rings produce.
"""

from typing import List

import numpy as np
import pandas as pd

WINDOWS = {"1h": 3600, "24h": 86400, "7d": 604800}


def _prior_count(df: pd.DataFrame, key: str) -> pd.Series:
    return df.groupby(key).cumcount()


def _prior_mean(df: pd.DataFrame, key: str, value: str) -> pd.Series:
    """Expanding mean EXCLUDING the current row."""
    csum = df.groupby(key)[value].cumsum() - df[value]
    cnt = df.groupby(key).cumcount()
    return csum / cnt.where(cnt > 0)


def _prior_std(df: pd.DataFrame, key: str, value: str) -> pd.Series:
    sq = df[value] ** 2
    csum = df.groupby(key)[value].cumsum() - df[value]
    csum2 = sq.groupby(df[key]).cumsum() - sq
    cnt = df.groupby(key).cumcount()
    n = cnt.where(cnt > 1)
    var = (csum2 / n) - (csum / n) ** 2
    return np.sqrt(var.clip(lower=0))


def _prior_max(df: pd.DataFrame, key: str, value: str) -> pd.Series:
    shifted = df.groupby(key)[value].shift(1)
    return shifted.groupby(df[key]).cummax()


def _seconds_since_prior(df: pd.DataFrame, key: str) -> pd.Series:
    return df.groupby(key)["timestamp"].diff().dt.total_seconds()


def _window_counts(df: pd.DataFrame, key: str, seconds: int) -> np.ndarray:
    """Transactions by the same entity in the trailing window, excluding self.

    Implemented per group with searchsorted rather than a rolling window,
    because pandas time-based rolling on a non-monotonic-per-group frame is
    both slower here and easy to get subtly wrong at group boundaries.
    """
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


def _prior_distinct(df: pd.DataFrame, key: str, target: str) -> np.ndarray:
    """Number of distinct `target` values the entity had seen BEFORE this row.

    Vectorised: a (key, target) pair contributes to the distinct count exactly
    once, at its first occurrence, so the running distinct count is the
    cumulative sum of first-occurrence flags within the key - shifted to
    exclude the current row. The equivalent per-row Python loop cost 5+ minutes
    on 1.6M rows, which made the adversarial loop impractical.
    """
    is_first = (df.groupby([key, target]).cumcount() == 0).astype(np.int32)
    cum = is_first.groupby(df[key]).cumsum()
    return (cum - is_first).to_numpy(dtype=np.int32)


def _is_first_time(df: pd.DataFrame, keys: List[str]) -> np.ndarray:
    """1 when this (entity, target) pair has not occurred before."""
    return (df.groupby(keys).cumcount() == 0).to_numpy().astype(np.int8)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the modelling frame. Input must carry the world schema."""
    d = df.sort_values("timestamp", kind="stable").reset_index(drop=True).copy()
    d["abs_amount"] = d["amount"].abs()
    f = pd.DataFrame(index=d.index)

    # --- raw context ---------------------------------------------------
    f["amount"] = d["amount"]
    f["abs_amount"] = d["abs_amount"]
    f["is_refund"] = (d["amount"] < 0).astype(np.int8)
    f["hour"] = d["hour"]
    f["dow"] = d["dow"]
    f["is_online"] = d["channel"].eq("Online Transaction").astype(np.int8)
    f["mcc"] = d["mcc"].astype("category")
    f["channel"] = d["channel"].astype("category")

    # --- baseline deviation (user) --------------------------------------
    u_mean = _prior_mean(d, "user_id", "abs_amount")
    u_std = _prior_std(d, "user_id", "abs_amount")
    u_max = _prior_max(d, "user_id", "abs_amount")
    f["u_prior_n"] = _prior_count(d, "user_id")
    f["u_amt_over_mean"] = d["abs_amount"] / u_mean.replace(0, np.nan)
    f["u_amt_over_max"] = d["abs_amount"] / u_max.replace(0, np.nan)
    f["u_amt_z"] = (d["abs_amount"] - u_mean) / u_std.replace(0, np.nan)
    f["u_exceeds_prior_max"] = (d["abs_amount"] > u_max).astype(np.int8)

    # --- velocity --------------------------------------------------------
    f["u_secs_since_last"] = _seconds_since_prior(d, "user_id")
    for name, secs in WINDOWS.items():
        f[f"u_txn_{name}"] = _window_counts(d, "user_id", secs)
    f["m_txn_1h"] = _window_counts(d, "merchant_id", WINDOWS["1h"])
    f["d_txn_24h"] = _window_counts(d, "device_id", WINDOWS["24h"])

    # --- novelty ---------------------------------------------------------
    f["u_first_merchant"] = _is_first_time(d, ["user_id", "merchant_id"])
    f["u_first_mcc"] = _is_first_time(d, ["user_id", "mcc"])
    f["u_first_device"] = _is_first_time(d, ["user_id", "device_id"])
    f["u_first_state"] = _is_first_time(d, ["user_id", "state"])
    f["u_distinct_merch_prior"] = _prior_distinct(d, "user_id", "merchant_id")
    f["u_merch_share"] = (f["u_distinct_merch_prior"] /
                          f["u_prior_n"].replace(0, np.nan))

    # --- entity sharing (graph proxy) -------------------------------------
    # A device touching many distinct users is the cheapest mule/farm signal
    # available without building the full graph.
    f["d_distinct_users_prior"] = _prior_distinct(d, "device_id", "user_id")
    f["m_distinct_users_prior"] = _prior_distinct(d, "merchant_id", "user_id")
    f["m_prior_n"] = _prior_count(d, "merchant_id")
    f["d_prior_n"] = _prior_count(d, "device_id")

    # --- hour deviation ---------------------------------------------------
    hr = d["hour"].astype(float)
    u_hr_mean = _prior_mean(d.assign(_h=hr), "user_id", "_h")
    f["u_hour_dev"] = (hr - u_hr_mean).abs()

    meta = d[["txn_id", "timestamp", "user_id", "merchant_id", "device_id",
              "is_fraud", "attack_id"]].copy()
    return pd.concat([meta, f], axis=1)


FEATURE_COLUMNS = None  # populated on first build; see feature_names()


def feature_names(frame: pd.DataFrame) -> List[str]:
    drop = {"txn_id", "timestamp", "user_id", "merchant_id", "device_id",
            "is_fraud", "attack_id"}
    return [c for c in frame.columns if c not in drop]
