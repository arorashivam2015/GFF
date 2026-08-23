"""The split this solution's headline number depends on.

Not a family-label holdout: RedLab_1 measured that a family holdout costs a
supervised detector only ~0.02 PR-AUC, because a family label bundles
parameter combinations rendered by one shared generic engine, not distinct
mechanisms. This split instead uses the taxonomy's own maturity rating -
observed / emerging / projected - and removes every PROJECTED vector from
training entirely, at the row level, not just by relabelling. Projected
vectors are attacks that have not been observed in the wild yet; scoring a
detector against them with zero training exposure is the most honest
available proxy for "does this catch fraud that doesn't exist yet."
"""

from typing import Tuple

import pandas as pd

from ..taxonomy.loader import Taxonomy
from ..taxonomy.schema import Maturity


def zero_exposure_split(frame: pd.DataFrame, taxonomy: Taxonomy,
                        test_frac: float = 0.3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    projected_ids = {v.id for v in taxonomy if v.maturity == Maturity.PROJECTED}
    is_projected = frame["attack_id"].isin(projected_ids)

    cut = frame["timestamp"].quantile(1 - test_frac)
    train = frame[(frame.timestamp <= cut) & ~(frame.is_fraud.eq(1) & is_projected)]
    test = frame[(frame.timestamp > cut) &
                (frame.is_fraud.eq(0) | (frame.is_fraud.eq(1) & is_projected))]
    return train, test


def verify_no_leakage(train: pd.DataFrame, taxonomy: Taxonomy) -> None:
    """Raises if any projected-vector fraud reached the training frame."""
    projected_ids = {v.id for v in taxonomy if v.maturity == Maturity.PROJECTED}
    leaked = train[train.is_fraud.eq(1) & train.attack_id.isin(projected_ids)]
    if len(leaked):
        raise AssertionError(
            f"{len(leaked)} projected-vector rows leaked into training: "
            f"{sorted(leaked.attack_id.unique())}")
