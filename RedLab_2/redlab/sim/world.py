"""Agent-based payment world.

Emits a legitimate population with persistent entities. Attacks are injected on
top of this world (see redlab.sim.attacks) rather than generated standalone, so
that fraud inherits a real background: the same merchants, the same devices,
the same behavioural baselines it has to hide inside.

The design target is joint structure. Harness validation showed a
marginals-only generator scoring discriminator AUC 0.847 despite near-perfect
marginals, because it had none of the dependencies that define real payment
data. This world builds them explicitly:

  merchant -> mcc        99.3% of merchants trade under exactly one MCC
  mcc      -> amount     per-category log-normal, 7x spread across categories
  mcc      -> hour       category-specific circadian shape
  mcc      -> channel    some categories are wholly online, most wholly present
  user     -> merchant   top-5 merchants carry ~46% of a user's transactions
  user     -> geography  ~83% of spend in the user's home state
  user     -> amount     a persistent per-user spend scale

Amounts are drawn by inverse-CDF sampling from the per-category empirical
distribution, not from a fitted log-normal. Reference log-amounts carry skew
-0.71 with a truncated upper tail; a log-normal fit reproduces the variance
almost exactly (per-MCC log-sd within 0.01) while overshooting p99 by 2-8x.

Per-user spend persistence is preserved through a Gaussian copula rather than a
multiplicative factor. A latent normal z = sqrt(rho)*z_user + sqrt(1-rho)*eps is
standard normal by construction, so u = Phi(z) is uniform and Q_mcc(u) restores
the empirical marginal exactly, while a user's z_user keeps them consistently
toward the top or bottom of every category they shop in.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from .conditionals import ConditionalProfile

USER_RHO = 0.25            # share of latent amount variance that is persistent per user
N_REGULARS = 8             # size of a cardholder's habitual merchant set
DEVICES_PER_USER = (1, 3)


@dataclass
class WorldConfig:
    """Defaults reproduce the reference corpus's sampling density.

    Measured merchant Zipf alpha is partly an artifact of how many
    transactions each merchant receives, not only of the generating exponent:
    at ~50 txns/merchant the measured slope is 1.60, at ~221 it is 1.95, using
    an identical generator. The reference sits at 243 txns/merchant. Shrinking
    the world therefore breaks the concentration target even though nothing
    about the generator changed - merchant concentration cannot be validated
    on a small sample.
    """

    n_cardholders: int = 3000
    n_merchants: int = 100000
    days: int = 240
    # n_merchants:n_cardholders matches the reference corpus's own ratio
    # (50.2:1, from graph_calibration_profile.json) - not RedLab_1's 2:1,
    # which was tuned for tabular Zipf-alpha fidelity, not bipartite degree.
    start_date: str = "2025-01-01"
    seed: int = 42
    loyalty_target: float = 0.46      # share of txns at a user's regulars
    home_state_share: float = 0.83
    cold_outreach_share: float = 0.55
    # Share of EXPLORATION visits that sample a merchant uniformly within its
    # category rather than by popularity. Reference-corpus merchant degree
    # (distinct users per merchant) is long-tailed to an extreme RedLab_1's
    # popularity-weighted exploration never reaches: even after correcting
    # the merchant:user population ratio, realised merchant degree stayed
    # far above target because exploration keeps re-selecting already-popular
    # merchants. This adds genuine "tries an obscure local merchant" behaviour.
    refund_share: float = 0.051
    zipf_alpha: float = 2.60          # within-category popularity exponent
    error_rate: float = 0.016

    # Device realism. An earlier version gave every device exactly one owner
    # for all time, which made "device seen with more than one user" a perfect
    # fraud oracle: the detector reached PR-AUC 1.000 by reading a property of
    # the simulator rather than of payments. Real device graphs are messier -
    # households share hardware, people replace phones, public terminals exist.
    household_user_share: float = 0.22    # users belonging to a shared household
    household_txn_share: float = 0.05     # txns made on the household device
    public_txn_share: float = 0.012       # txns on a shared/public terminal
    n_public_devices: int = 60


@dataclass
class PaymentWorld:
    """A persistent population of cardholders, merchants and devices."""

    profile: ConditionalProfile
    cfg: WorldConfig = field(default_factory=WorldConfig)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.cfg.seed)
        self._build_merchants()
        self._build_cardholders()

    # -- entity construction ----------------------------------------------

    def _build_merchants(self) -> None:
        cfg, rng = self.cfg, self.rng
        n = cfg.n_merchants
        keys = self.profile.mcc_keys()
        weights = self.profile.mcc_weights()

        # Merchant COUNT per category is damped so that small categories still
        # get a viable merchant population. Popularity is renormalised below so
        # that damping does not leak into the transaction mix - conflating the
        # two was measured to over-represent rare categories by up to 13x.
        damped = weights ** 0.7
        damped = damped / damped.sum()
        self.m_mcc = rng.choice(keys, size=n, p=damped)

        states = list(self.profile.state_mix)
        s_w = np.array(list(self.profile.state_mix.values()), dtype=float)
        s_w = s_w / s_w.sum()
        self.m_state = rng.choice(states, size=n, p=s_w)

        # Popularity is built PER CATEGORY and normalised to that category's
        # target transaction share. This makes P(merchant in mcc k) exactly the
        # reference share while keeping Zipf-like concentration within each
        # category, decoupling merchant count from transaction volume.
        share = {k: w for k, w in zip(keys, weights)}
        self.m_pop = np.zeros(n, dtype=float)
        for k in keys:
            idx = np.where(self.m_mcc == k)[0]
            if len(idx) == 0:
                continue
            rank = rng.permutation(len(idx)) + 1
            w = rank.astype(float) ** (-cfg.zipf_alpha)
            self.m_pop[idx] = w / w.sum() * share[k]
        self.m_pop /= self.m_pop.sum()

        self.m_id = np.array([f"M{i:06d}" for i in range(n)])

        # Per-state merchant pools, for home-locality sampling.
        self._state_pool: Dict[str, np.ndarray] = {}
        for st in np.unique(self.m_state):
            self._state_pool[st] = np.where(self.m_state == st)[0]

        # Per-category and per-(category, state) pools. Exploration selects a
        # category first and a merchant second, which makes the category mix
        # exact by construction; selecting a merchant first let the random
        # merchant->state assignment distort it, since Zipf concentration puts
        # most of a category's weight in a handful of merchants whose state is
        # arbitrary.
        self._mcc_pool: Dict[str, np.ndarray] = {}
        self._mcc_state_pool: Dict[tuple, np.ndarray] = {}
        for k in np.unique(self.m_mcc):
            idx = np.where(self.m_mcc == k)[0]
            self._mcc_pool[k] = idx
            for st in np.unique(self.m_state[idx]):
                self._mcc_state_pool[(k, st)] = idx[self.m_state[idx] == st]

    def _build_cardholders(self) -> None:
        cfg, rng = self.cfg, self.rng
        n = cfg.n_cardholders

        self.u_id = np.array([f"U{i:05d}" for i in range(n)])
        states = list(self.profile.state_mix)
        s_w = np.array(list(self.profile.state_mix.values()), dtype=float)
        s_w = s_w / s_w.sum()
        # Cardholders live in physical states; ONLINE is a merchant-side token.
        phys = [s for s in states if s != self.profile.online_state_token]
        p_w = np.array([self.profile.state_mix[s] for s in phys], dtype=float)
        p_w /= p_w.sum()
        self.u_home = rng.choice(phys, size=n, p=p_w)

        # Persistent per-user position in the latent amount copula.
        self.u_spend = rng.standard_normal(n)

        # Activity: log-normal around the reference median of 2.8 txns/active-day.
        med = self.profile.users.txns_per_day.get("p50", 2.8)
        self.u_rate = rng.lognormal(np.log(med), 0.55, size=n).clip(0.2, 25.0)

        # Habitual merchants, drawn with home-state bias so a user's regulars
        # are geographically coherent rather than scattered nationwide.
        self.u_regulars = np.zeros((n, N_REGULARS), dtype=np.int64)
        for i in range(n):
            pool = self._state_pool.get(self.u_home[i])
            if pool is None or len(pool) < N_REGULARS:
                pool = np.arange(self.cfg.n_merchants)
            w = self.m_pop[pool] / self.m_pop[pool].sum()
            k = min(N_REGULARS, len(pool))
            self.u_regulars[i, :k] = rng.choice(pool, size=k, replace=False, p=w)
            if k < N_REGULARS:
                self.u_regulars[i, k:] = self.u_regulars[i, 0]

        # Visit frequency within a user's regulars follows those merchants'
        # popularity. Choosing uniformly among them was measured to flatten the
        # category mix (mcc single-feature discriminator AUC 0.61), because a
        # user visited their occasional restaurant as often as their grocery.
        reg_w = self.m_pop[self.u_regulars]
        self.u_reg_cdf = np.cumsum(reg_w / reg_w.sum(axis=1, keepdims=True), axis=1)

        # --- devices -----------------------------------------------------
        # Three sources of legitimate device/user overlap, so that device
        # sharing is a graded signal rather than a fraud oracle:
        #   1. multiple personal devices, ACTIVATED over time (handset churn)
        #   2. household devices shared by several users
        #   3. public terminals used occasionally by anyone
        lo, hi = DEVICES_PER_USER
        self.u_ndev = rng.integers(lo, hi + 1, size=n)
        self._max_dev = hi

        dev_rows, start_rows = [], []
        for i in range(n):
            k = int(self.u_ndev[i])
            ids = [f"D{i:05d}_{j}" for j in range(k)]
            # First device is present from day zero; later ones appear partway
            # through, which makes "first use of this device" a normal event.
            starts = [0] + sorted(rng.integers(1, max(cfg.days, 2), size=k - 1).tolist())
            pad = hi - k
            dev_rows.append(ids + [ids[-1]] * pad)
            start_rows.append(starts + [10 ** 6] * pad)
        self.u_dev_matrix = np.array(dev_rows, dtype=object)
        self.u_dev_start = np.array(start_rows, dtype=np.int64)

        # Households: a share of users share one device with 1-3 others.
        self.u_household = np.full(n, "", dtype=object)
        pool = rng.permutation(n)[: int(n * cfg.household_user_share)]
        h = cursor = 0
        while cursor < len(pool) - 1:
            size = int(min(rng.integers(2, 5), len(pool) - cursor))
            if size < 2:
                break
            for m in pool[cursor:cursor + size]:
                self.u_household[m] = f"H{h:05d}"
            cursor += size
            h += 1

        self.public_devices = np.array(
            [f"P{j:04d}" for j in range(cfg.n_public_devices)], dtype=object)

    # -- generation --------------------------------------------------------

    def generate(self, days: Optional[int] = None) -> pd.DataFrame:
        """Emit the legitimate transaction population."""
        cfg, rng = self.cfg, self.rng
        days = days or cfg.days
        n_users = cfg.n_cardholders

        # Active days per user, then transactions per active day.
        active = rng.binomial(days, np.clip(self.u_rate / self.u_rate.max() * 0.6 + 0.25, 0, 1))
        counts = rng.poisson(np.repeat(self.u_rate, np.maximum(active, 0)))
        counts = counts[counts > 0]
        user_idx = np.repeat(np.arange(n_users), np.maximum(active, 0))[
            np.repeat(np.arange(len(counts)), 1)
        ][: len(counts)]
        user_of_txn = np.repeat(user_idx, counts)
        day_of_txn = rng.integers(0, days, size=len(user_of_txn))
        n = len(user_of_txn)
        if n == 0:
            raise RuntimeError("world generated no transactions; check config")

        # --- merchant choice: regulars vs exploration --------------------
        use_regular = rng.random(n) < cfg.loyalty_target
        merch = np.empty(n, dtype=np.int64)

        reg_idx = user_of_txn[use_regular]
        draws = rng.random(len(reg_idx))
        r_slot = (self.u_reg_cdf[reg_idx] < draws[:, None]).sum(axis=1).clip(0, N_REGULARS - 1)
        merch[use_regular] = self.u_regulars[reg_idx, r_slot]

        # Exploration: draw the category from the reference mix, then a
        # merchant inside it, preferring the user's home state.
        expl = np.where(~use_regular)[0]
        if len(expl):
            keys = self.profile.mcc_keys()
            weights = self.profile.mcc_weights()
            want_mcc = rng.choice(keys, size=len(expl), p=weights)
            at_home = rng.random(len(expl)) < cfg.home_state_share
            homes = self.u_home[user_of_txn[expl]]

            is_cold = rng.random(len(expl)) < cfg.cold_outreach_share
            for k in np.unique(want_mcc):
                sel = np.where(want_mcc == k)[0]
                gpool = self._mcc_pool[k]
                gw = self.m_pop[gpool] / self.m_pop[gpool].sum()
                picked = rng.choice(gpool, size=len(sel), p=gw)

                cold_sel = sel[is_cold[sel]]
                if len(cold_sel) and len(gpool) > 1:
                    picked[np.searchsorted(sel, cold_sel)] = rng.choice(
                        gpool, size=len(cold_sel))  # uniform, not popularity-weighted

                # Redirect the home-state share to a local merchant in the
                # same category where one exists.
                for st in np.unique(homes[sel]):
                    lp = self._mcc_state_pool.get((k, st))
                    if lp is None or not len(lp):
                        continue
                    local = sel[(homes[sel] == st) & at_home[sel]]
                    if not len(local):
                        continue
                    lw = self.m_pop[lp] / self.m_pop[lp].sum()
                    picked[np.searchsorted(sel, local)] = rng.choice(
                        lp, size=len(local), p=lw)
                merch[expl[sel]] = picked

        mcc = self.m_mcc[merch]

        # --- MCC-conditional amount, hour and channel --------------------
        amount = np.empty(n, dtype=float)
        hour = np.empty(n, dtype=np.int16)
        channel = np.empty(n, dtype=object)

        for k in np.unique(mcc):
            m = self.profile.mccs[k]
            sel = np.where(mcc == k)[0]

            # Gaussian copula -> uniform -> empirical inverse CDF.
            z = (np.sqrt(USER_RHO) * self.u_spend[user_of_txn[sel]]
                 + np.sqrt(1.0 - USER_RHO) * rng.standard_normal(len(sel)))
            u = norm.cdf(z)
            q = np.asarray(m.amount_quantiles, dtype=float)
            a = np.interp(u, np.linspace(0.0, 1.0, len(q)), q)

            # Real amounts cluster on whole units (reference: 11.0% at .00);
            # smooth draws sit at 1.0% and are separable on that alone.
            snap = rng.random(len(sel)) < m.round_share
            a = np.where(snap, np.maximum(np.round(a), 1.0), a)
            amount[sel] = a

            hp = np.array(m.hour_dist, dtype=float)
            hour[sel] = rng.choice(24, size=len(sel), p=hp / hp.sum())

            ck = list(m.channel_dist)
            cw = np.array(list(m.channel_dist.values()), dtype=float)
            channel[sel] = rng.choice(ck, size=len(sel), p=cw / cw.sum())

        # --- refunds, errors, device, geography --------------------------
        is_refund = rng.random(n) < cfg.refund_share
        amount = np.where(is_refund, -amount, amount)

        err_keys = ["(none)", "Insufficient Balance,", "Bad PIN,",
                    "Technical Glitch,", "Bad Card Number,", "Bad CVV,"]
        err_p = np.array([1 - cfg.error_rate, 0.0100, 0.0024, 0.0020, 0.0006, 0.0004])
        err_p[0] = 1 - err_p[1:].sum()
        error = rng.choice(err_keys, size=n, p=err_p)

        # Personal device, restricted to those already activated on that day.
        starts = self.u_dev_start[user_of_txn]
        active = (starts <= day_of_txn[:, None]).astype(float)
        active[:, 0] = 1.0
        cdf = np.cumsum(active, axis=1)
        cdf = cdf / cdf[:, -1:]
        slot = (cdf < rng.random(n)[:, None]).sum(axis=1).clip(0, self._max_dev - 1)
        device = self.u_dev_matrix[user_of_txn, slot].astype(object)

        # Household device, where the user belongs to one.
        house = self.u_household[user_of_txn]
        use_house = (rng.random(n) < cfg.household_txn_share) & (house != "")
        device[use_house] = house[use_house]

        # Public terminal.
        use_pub = rng.random(n) < cfg.public_txn_share
        if use_pub.any():
            device[use_pub] = rng.choice(self.public_devices, int(use_pub.sum()))

        state = self.m_state[merch]
        online = np.char.find(channel.astype(str), "Online") >= 0
        state = np.where(online, self.profile.online_state_token, state)

        ts = (pd.Timestamp(cfg.start_date)
              + pd.to_timedelta(day_of_txn, unit="D")
              + pd.to_timedelta(hour.astype(int), unit="h")
              + pd.to_timedelta(rng.integers(0, 60, n), unit="m"))

        df = pd.DataFrame({
            "txn_id": [f"T{i:09d}" for i in range(n)],
            "timestamp": ts,
            "hour": hour.astype(int),
            "dow": ts.dayofweek.astype(int),
            "user_id": self.u_id[user_of_txn],
            "device_id": device,
            "merchant_id": self.m_id[merch],
            "mcc": mcc,
            "channel": channel.astype(str),
            "amount": amount,
            "state": state,
            "error": error,
            "is_fraud": 0,
            "attack_id": None,
        }).sort_values("timestamp", ignore_index=True)

        return df


def build_world(conditional_path: str = "data/processed/conditional_profile.json",
                cfg: Optional[WorldConfig] = None) -> PaymentWorld:
    return PaymentWorld(ConditionalProfile.load(conditional_path), cfg or WorldConfig())
