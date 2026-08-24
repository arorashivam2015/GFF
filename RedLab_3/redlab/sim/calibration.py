"""Calibration targets for the payment-world simulator.

FIDELITY HONESTY NOTE
---------------------
No public corpus of real payment authorisations exists that we can redistribute
or anchor against without licensing. This project therefore calibrates against
two things, and labels each precisely:

  1. REFERENCE CORPUS - IBM's TabFormer credit-card transaction dataset
     (Padhi et al., ICASSP 2021). 24.4M records. It is *itself synthetic*,
     authored by IBM. Using it is NOT equivalent to anchoring on real data.
     Its value is that it is (a) externally authored, so comparing our output
     to it is not circular, and (b) a published benchmark, so the comparison
     is reproducible by a third party.

  2. STYLISED FACTS - empirical regularities documented for *real* payment
     data: Benford conformance of amounts, Zipf-like merchant concentration,
     log-normal ticket sizes, circadian and weekly rhythm, and the large
     card-not-present fraud lift. Conformance to these is meaningful
     independent of any reference corpus, because they are properties of real
     systems reported in the literature.

  3. PUBLISHED AGGREGATES - for UPI rails, where no transaction-level corpus
     exists at all, targets come from NPCI/RBI published monthly aggregates.
     These constrain marginals only, never joint structure, and are marked as
     such in `UPI_PRIOR.confidence`.

Any fidelity number this project reports must carry the label of which of these
it was measured against. Overclaiming here is the fastest way to lose a
payments audience.
"""

from enum import Enum
from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class Confidence(str, Enum):
    REFERENCE_CORPUS = "reference_corpus"      # fitted to TabFormer (synthetic)
    STYLISED_FACT = "stylised_fact"            # from literature on real data
    PUBLISHED_AGGREGATE = "published_aggregate"  # NPCI/RBI marginals only
    ASSUMPTION = "assumption"                  # our judgement, stated openly


class AmountTargets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    percentiles: Dict[str, float] = Field(..., description="pXX -> value")
    lognormal_mu: float
    lognormal_sigma: float
    mean: float
    benford_first_digit: List[float] = Field(..., min_length=9, max_length=9)
    refund_share: float = Field(..., ge=0, le=1, description="share of negative amounts")
    zero_share: float = Field(..., ge=0, le=1)


class MerchantTargets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_merchants: int
    top1pct_volume_share: float = Field(..., ge=0, le=1)
    zipf_alpha: float = Field(..., description="rank-frequency log-log slope")
    mcc_mix: Dict[str, float]
    mcc_fraud_rate: Dict[str, float]


class TemporalTargets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hour_of_day: List[float] = Field(..., min_length=24, max_length=24)
    day_of_week: List[float] = Field(..., min_length=7, max_length=7)
    inter_txn_hours: Dict[str, float] = Field(..., description="pXX -> hours between txns")


class CalibrationProfile(BaseModel):
    """Everything the simulator needs to match a target population."""

    model_config = ConfigDict(extra="forbid")

    name: str
    provenance: str
    confidence: Confidence
    n_rows: int
    n_users: int
    fraud_rate: float
    amount: AmountTargets
    merchant: MerchantTargets
    temporal: TemporalTargets
    channel_mix: Dict[str, float]
    channel_fraud_rate: Dict[str, float]
    error_mix: Dict[str, float]
    per_user_txn_percentiles: Dict[str, float]
    notes: Optional[str] = None

    @property
    def cnp_fraud_lift(self) -> Optional[float]:
        """Ratio of card-not-present to card-present fraud rate.

        One of the most robust real-world regularities in card fraud, and a
        target the simulator must reproduce or its attack mix is wrong.
        """
        online = next((v for k, v in self.channel_fraud_rate.items()
                       if "online" in k.lower()), None)
        present = [v for k, v in self.channel_fraud_rate.items()
                   if "swipe" in k.lower() or "chip" in k.lower()]
        if online is None or not present or not np.mean(present):
            return None
        return float(online / np.mean(present))


# --------------------------------------------------------------------------
# UPI prior - marginals only, from published aggregates
# --------------------------------------------------------------------------

UPI_PRIOR_NOTE = """
UPI has no public transaction-level corpus. These targets constrain marginal
distributions only - they say nothing about joint structure, per-user
behaviour, or graph topology, all of which the simulator must supply from
mechanism rather than from data. Values are order-of-magnitude anchors derived
from NPCI monthly statistics and RBI payment-system reporting, and MUST be
refreshed against the current published figures before the walkthrough is
finalised. They are deliberately coarse: false precision here would be worse
than none.
"""


def upi_prior() -> Dict[str, object]:
    """Coarse marginal anchors for UPI rails.

    Returned as a plain dict rather than a CalibrationProfile because we
    genuinely do not have the joint structure a full profile implies, and
    pretending otherwise would misrepresent the evidence.
    """
    return {
        "confidence": Confidence.PUBLISHED_AGGREGATE,
        "note": UPI_PRIOR_NOTE.strip(),
        "p2p_share_of_count": 0.45,
        "p2m_share_of_count": 0.55,
        "p2m_median_amount_inr": 250.0,
        "p2p_median_amount_inr": 800.0,
        "amount_heavy_tail": True,
        "peak_hours_local": [11, 12, 13, 19, 20, 21],
        "requires_refresh": True,
    }


# --------------------------------------------------------------------------
# Extraction from the reference corpus
# --------------------------------------------------------------------------


def fit_zipf_alpha(counts: np.ndarray) -> float:
    """Slope of log(frequency) against log(rank). Real merchant popularity is
    strongly Zipf-like; a simulator that samples merchants uniformly produces
    a flat slope and is trivially separable from real data."""
    c = np.sort(np.asarray(counts, dtype=float))[::-1]
    c = c[c > 0]
    if len(c) < 10:
        return float("nan")
    ranks = np.arange(1, len(c) + 1)
    slope, _ = np.polyfit(np.log(ranks), np.log(c), 1)
    return float(-slope)


def benford_expected() -> np.ndarray:
    d = np.arange(1, 10)
    return np.log10(1 + 1 / d)


def benford_mad(observed: np.ndarray) -> float:
    """Mean absolute deviation from Benford, in percentage points.

    Nigrini's conventional bands for first-digit MAD:
      < 0.6pp close conformity | 0.6-1.2 acceptable | > 1.5 nonconformity.
    """
    obs = np.asarray(observed, dtype=float)
    obs = obs / obs.sum()
    return float(np.mean(np.abs(obs - benford_expected())) * 100)
