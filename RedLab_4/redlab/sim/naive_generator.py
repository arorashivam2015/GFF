"""The naive baseline: sample each marginal independently, with no joint
structure at all. This exists to prove the fidelity harness actually catches
bad synthesis before any trained model is allowed near it - RedLab_1's own
history recorded this exact generator scoring discriminator AUC 0.977
(1.0 = trivially separable). If the harness doesn't reproduce a comparably
damning number here, the harness itself is broken, not just the generator.
"""

import numpy as np
import pandas as pd

from .calibration import CalibrationProfile


def generate_naive(profile: CalibrationProfile, n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    amount = rng.lognormal(profile.amount.lognormal_mu, profile.amount.lognormal_sigma, n)

    mcc_keys = list(profile.merchant.mcc_mix)
    mcc_w = np.array(list(profile.merchant.mcc_mix.values()))
    mcc_w = mcc_w / mcc_w.sum()
    mcc = rng.choice(mcc_keys, size=n, p=mcc_w)

    hour_w = np.array(profile.temporal.hour_of_day)
    hour_w = hour_w / hour_w.sum()
    hour = rng.choice(24, size=n, p=hour_w)

    chan_keys = list(profile.channel_mix)
    chan_w = np.array(list(profile.channel_mix.values()))
    chan_w = chan_w / chan_w.sum()
    channel = rng.choice(chan_keys, size=n, p=chan_w)

    return pd.DataFrame({
        "amount": amount, "mcc": mcc, "hour": hour, "channel": channel,
        "is_fraud": np.zeros(n, dtype=int),
    })
