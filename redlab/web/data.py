"""Data access for the web prototype.

Loads pre-computed artifacts (fidelity reports, detector eval, adversarial
loop results) and the persisted detector/console sample, so the app serves
instantly rather than retraining on request. All artifacts are produced by the
scripts in scripts/ and are reproducible by re-running them.
"""

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
ART = ROOT / "artifacts"


@lru_cache(maxsize=1)
def taxonomy():
    from ..taxonomy.loader import Taxonomy
    return Taxonomy.load()


@lru_cache(maxsize=1)
def fidelity_legit() -> dict:
    return json.loads((ART / "fidelity_legit.json").read_text())


@lru_cache(maxsize=1)
def fidelity_attacks() -> dict:
    return json.loads((ART / "fidelity_attacks.json").read_text())


@lru_cache(maxsize=1)
def detector_eval() -> dict:
    return json.loads((ART / "detector_eval.json").read_text())


@lru_cache(maxsize=1)
def adversarial_loop() -> list:
    return json.loads((ART / "adversarial_loop.json").read_text())


@lru_cache(maxsize=1)
def ablation() -> dict:
    return json.loads((ART / "ablation.json").read_text())


@lru_cache(maxsize=1)
def model_bundle() -> dict:
    return joblib.load(ART / "model" / "detector.joblib")


@lru_cache(maxsize=1)
def console_sample() -> pd.DataFrame:
    df = pd.read_parquet(ART / "model" / "console_sample.parquet")
    return df.sort_values("timestamp").reset_index(drop=True)


@lru_cache(maxsize=1)
def demo_window() -> tuple:
    """A representative day for the default console view.

    Fraud campaigns cluster tightly in time (see FINDINGS.md #6), so the
    literal chronological tail of a multi-month sample is either 100% fraud
    (if it lands inside a campaign burst) or 100% legitimate (if it doesn't) -
    neither is representative, and the latter makes the default view look
    broken for a live-monitoring demo. Anchoring on the busiest fraud day
    gives a default view that actually demonstrates the system working, while
    every other day remains reachable via the date filter.
    """
    df = console_sample()
    day = df.loc[df.is_fraud == 1, "timestamp"].dt.floor("D").value_counts().idxmax()
    return day, day + pd.Timedelta(days=1)
