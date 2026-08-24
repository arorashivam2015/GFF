"""Partition the population across synthetic institutions.

RedLab_1's own taxonomy already states, in PF-IND-005's and PF-CT-001's own
detection hypotheses, that these two vectors work BECAUSE any one issuer or
acquirer sees only a fragment of the attack ("multi-hop flow conservation
... visible only network-side"; "network-level BIN velocity monitoring" as
the stated mitigation). This module makes that fragmentation literal: every
cardholder is assigned to one issuer, every merchant to one acquirer, via a
deterministic hash - no change to the world simulator itself, since spread
across institutions is a property of WHO owns which entity, not of how
transactions are generated.
"""

import hashlib
from typing import Dict

import numpy as np
import pandas as pd

INSTITUTION_NAMES = ["Alpha Bank", "Beta Trust", "Gamma Financial", "Delta Credit Union"]


def _hash_bucket(ids: pd.Series, n_buckets: int, salt: str) -> np.ndarray:
    """Deterministic, salted hash assignment - stable across reruns, and
    issuer vs. acquirer assignment uses a different salt so a user's bank
    and a merchant's acquirer are independent, as they are in reality."""
    def h(x: str) -> int:
        return int(hashlib.md5(f"{salt}:{x}".encode()).hexdigest(), 16) % n_buckets
    return ids.astype(str).map(h).to_numpy()


def assign_institutions(df: pd.DataFrame, n_institutions: int = 4) -> pd.DataFrame:
    out = df.copy()
    out["issuer"] = _hash_bucket(out["user_id"], n_institutions, "issuer")
    out["acquirer"] = _hash_bucket(out["merchant_id"], n_institutions, "acquirer")
    return out


def cross_institution_spread(df: pd.DataFrame, attack_id: str) -> Dict[str, float]:
    """How many distinct issuers/acquirers does this vector's fraud touch -
    the number the whole solution's premise depends on being > 1."""
    sub = df[df.attack_id == attack_id]
    return {
        "n_issuers": int(sub.issuer.nunique()),
        "n_acquirers": int(sub.acquirer.nunique()),
        "n_events": int(len(sub)),
    }
