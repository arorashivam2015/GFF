"""Attack fidelity: does synthetic fraud carry the same *detectability
signature* as reference fraud?

Fidelity of the legitimate population is measured by how hard it is to tell
apart from the reference (see fidelity.py). That test does not work for fraud,
because fraud is supposed to be different from legitimate traffic. The question
for attacks is not "is it separable" but "is it separable in the same way, and
to the same degree, as real fraud".

So we train the same simple detector twice - once on the reference corpus with
its own labels, once on ours - and compare the resulting separability profiles
feature by feature. A generator whose fraud is too easy (invented giveaways) or
too hard (fraud that never deviates) shows up as a signed gap.

This matters because chasing "hard to detect" is the wrong objective and
produces its own artefacts. An early injection pass here scored ROC-AUC 0.989;
the reference scores 0.966 on the identical feature set, so the real defect was
2.3 points, concentrated in three measurable places, not the wholesale failure
the raw number suggested.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

RAW_FEATURES = ["amount", "hour", "dow", "mcc", "channel"]
CATEGORICAL = ["mcc", "channel"]


class SeparabilityProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    prevalence: float
    roc_auc: float
    pr_auc: float
    per_feature_auc: Dict[str, float]


class AttackFidelityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference: SeparabilityProfile
    generated: SeparabilityProfile

    @property
    def roc_gap(self) -> float:
        return self.generated.roc_auc - self.reference.roc_auc

    @property
    def feature_gaps(self) -> Dict[str, float]:
        return {f: self.generated.per_feature_auc[f] - self.reference.per_feature_auc[f]
                for f in self.reference.per_feature_auc}

    @property
    def max_abs_feature_gap(self) -> float:
        return max(abs(v) for v in self.feature_gaps.values())

    def verdict(self) -> str:
        g = abs(self.roc_gap)
        m = self.max_abs_feature_gap
        if g < 0.02 and m < 0.06:
            return "MATCHED - fraud is as detectable as reference fraud, feature by feature"
        if g < 0.05 and m < 0.12:
            return "CLOSE - minor signature differences"
        direction = "too easy" if self.roc_gap > 0 else "too hard"
        return f"DIVERGENT - generated fraud is {direction}"

    def render(self) -> str:
        out = ["", "=" * 78, "ATTACK FIDELITY: detectability signature vs reference fraud",
               "=" * 78,
               f"{'':14s}{'reference':>12s}{'generated':>12s}{'gap':>10s}",
               f"{'prevalence':14s}{self.reference.prevalence*100:>11.3f}%"
               f"{self.generated.prevalence*100:>11.3f}%{'':>10s}",
               f"{'ROC-AUC':14s}{self.reference.roc_auc:>12.4f}"
               f"{self.generated.roc_auc:>12.4f}{self.roc_gap:>+10.4f}",
               f"{'PR-AUC':14s}{self.reference.pr_auc:>12.4f}"
               f"{self.generated.pr_auc:>12.4f}"
               f"{self.generated.pr_auc - self.reference.pr_auc:>+10.4f}",
               "", "per-feature separability"]
        for f, gap in sorted(self.feature_gaps.items(), key=lambda kv: -abs(kv[1])):
            flag = "  <-- " if abs(gap) > 0.06 else ""
            out.append(f"  {f:12s}{self.reference.per_feature_auc[f]:>12.4f}"
                       f"{self.generated.per_feature_auc[f]:>12.4f}{gap:>+10.4f}{flag}")
        out += ["", f"VERDICT: {self.verdict()}", "=" * 78]
        return "\n".join(out)


def _profile(df: pd.DataFrame, label: str, features: List[str], seed: int = 0
             ) -> SeparabilityProfile:
    import lightgbm as lgb

    X = df[features].copy()
    for c in features:
        if c in CATEGORICAL or X[c].dtype == object:
            X[c] = X[c].astype("category")
    y = df["is_fraud"].to_numpy()

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y)
    clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                             random_state=seed, verbosity=-1).fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]

    per_feature = {}
    for c in features:
        s = lgb.LGBMClassifier(n_estimators=150, num_leaves=31, random_state=seed,
                               verbosity=-1).fit(Xtr[[c]], ytr)
        per_feature[c] = float(roc_auc_score(yte, s.predict_proba(Xte[[c]])[:, 1]))

    return SeparabilityProfile(
        label=label, prevalence=float(y.mean()),
        roc_auc=float(roc_auc_score(yte, p)),
        pr_auc=float(average_precision_score(yte, p)),
        per_feature_auc=per_feature)


def compare(generated: pd.DataFrame, reference: pd.DataFrame,
            features: Optional[List[str]] = None, seed: int = 0
            ) -> AttackFidelityReport:
    """Compare detectability signatures at matched prevalence.

    Prevalence is matched by subsampling the generated legitimate population,
    because PR-AUC is prevalence-dependent and an unmatched comparison would be
    measuring class balance rather than attack realism.
    """
    features = features or RAW_FEATURES
    rng = np.random.default_rng(seed)

    ref_prev = float(reference["is_fraud"].mean())
    fr = generated[generated.is_fraud == 1]
    lg = generated[generated.is_fraud == 0]
    n_legit = min(len(lg), int(len(fr) / max(ref_prev, 1e-9)) - len(fr))
    gen_matched = pd.concat(
        [fr, lg.sample(max(n_legit, 1), random_state=seed)], ignore_index=True)

    return AttackFidelityReport(
        reference=_profile(reference, "reference", features, seed),
        generated=_profile(gen_matched, "generated", features, seed))
