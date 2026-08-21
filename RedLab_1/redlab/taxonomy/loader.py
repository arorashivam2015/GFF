"""Load, validate and report on the attack taxonomy.

The coverage report is not decoration: "diversity of attacks identified" is a
judged criterion, and coverage over the morphological cross-product is the
defensible way to evidence it. Gaps printed here are the ideation backlog.
"""

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from .schema import (
    COVERAGE_AXES,
    AttackVector,
    Family,
    Rail,
    VectorFile,
)

VECTOR_DIR = Path(__file__).parent / "vectors"


class TaxonomyError(ValueError):
    """Raised when the on-disk taxonomy is inconsistent."""


class Taxonomy:
    """An immutable, validated view over the attack-vector corpus."""

    def __init__(self, vectors: List[AttackVector]) -> None:
        self._vectors = sorted(vectors, key=lambda v: v.id)
        self._by_id: Dict[str, AttackVector] = {v.id: v for v in self._vectors}

    # -- construction ------------------------------------------------------

    @classmethod
    def load(cls, directory: Optional[Path] = None) -> "Taxonomy":
        directory = Path(directory) if directory else VECTOR_DIR
        paths = sorted(directory.glob("*.yaml"))
        if not paths:
            raise TaxonomyError(f"no vector files found in {directory}")

        vectors: List[AttackVector] = []
        seen: Dict[str, str] = {}
        errors: List[str] = []

        for path in paths:
            try:
                raw = yaml.safe_load(path.read_text())
                parsed = VectorFile.model_validate(raw)
            except Exception as exc:  # noqa: BLE001 - aggregate and report all
                errors.append(f"{path.name}: {exc}")
                continue

            for vector in parsed.vectors:
                if vector.id in seen:
                    errors.append(
                        f"{path.name}: duplicate id {vector.id} "
                        f"(first seen in {seen[vector.id]})"
                    )
                    continue
                seen[vector.id] = path.name
                vectors.append(vector)

        if errors:
            raise TaxonomyError(
                "taxonomy failed validation:\n  - " + "\n  - ".join(errors)
            )
        return cls(vectors)

    # -- access ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._vectors)

    def __iter__(self):
        return iter(self._vectors)

    def __getitem__(self, vector_id: str) -> AttackVector:
        return self._by_id[vector_id]

    @property
    def vectors(self) -> List[AttackVector]:
        return list(self._vectors)

    def by_family(self, family: Family) -> List[AttackVector]:
        return [v for v in self._vectors if v.family == family]

    def by_rail(self, rail: Rail) -> List[AttackVector]:
        return [v for v in self._vectors if rail in v.rails]

    def top(self, n: int = 10) -> List[AttackVector]:
        """Highest-priority vectors — the simulation build order."""
        return sorted(self._vectors, key=lambda v: (-v.priority, v.id))[:n]

    def holdout_split(
        self, holdout_families: List[Family]
    ) -> Tuple[List[AttackVector], List[AttackVector]]:
        """Leave-one-family-out split.

        This is the split that makes the DEFEND evaluation honest: the detector
        trains on `train` and is scored on `holdout`, which it has never seen.
        """
        holdout_set = set(holdout_families)
        train = [v for v in self._vectors if v.family not in holdout_set]
        holdout = [v for v in self._vectors if v.family in holdout_set]
        if not holdout:
            raise TaxonomyError(f"holdout families {holdout_families} matched no vectors")
        return train, holdout

    # -- reporting ---------------------------------------------------------

    def axis_coverage(self) -> Dict[str, Counter]:
        """How many vectors touch each value of each morphological axis."""
        out: Dict[str, Counter] = {}
        for axis in COVERAGE_AXES:
            counter: Counter = Counter()
            for vector in self._vectors:
                for value in getattr(vector, axis):
                    counter[value.value] += 1
            out[axis] = counter
        return out

    def gaps(self) -> Dict[str, List[str]]:
        """Enum values no vector covers — the ideation backlog."""
        coverage = self.axis_coverage()
        out: Dict[str, List[str]] = {}
        for axis, enum_cls in COVERAGE_AXES.items():
            missing = [e.value for e in enum_cls if coverage[axis][e.value] == 0]
            if missing:
                out[axis] = missing
        return out

    def family_rail_matrix(self) -> Dict[str, Dict[str, int]]:
        """Family x rail occupancy — the ATT&CK-style matrix backing the UI."""
        matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for vector in self._vectors:
            for rail in vector.rails:
                matrix[vector.family.value][rail.value] += 1
        return {k: dict(v) for k, v in matrix.items()}

    def summary(self) -> Dict[str, object]:
        maturity: Counter = Counter(v.maturity.value for v in self._vectors)
        family: Counter = Counter(v.family.value for v in self._vectors)
        return {
            "total_vectors": len(self._vectors),
            "families": dict(family),
            "maturity": dict(maturity),
            "mean_novelty": round(
                sum(v.scores.novelty for v in self._vectors) / max(len(self._vectors), 1), 2
            ),
            "gaps": self.gaps(),
        }
