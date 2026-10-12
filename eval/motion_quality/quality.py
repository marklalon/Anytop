"""
Motion quality of generated clips: physical plausibility and data-prior conformity.

Two dimensions, each in [0, 1] per clip:

- ``physics`` (``physics.py``): label-free -- rigid bones, smooth parts, no
  frame jumps -- against the spread of the same statistics in real clips of the
  same body plan;
- ``prior`` (``prior.py``): whether the limbs move together and at the tempo
  the way real clips of the action do, calibrated so the median real clip
  scores 1.

A clip's quality is the geometric mean of the two, so a clip loses for either
failing; a task's is the median over its clips. The two dimensions share one
corpus index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from data_loaders.truebones.offline_reference_dataset import resolve_sources
from data_loaders.truebones.truebones_utils.dataset_sources import species_lookup_map

from .physics import PhysicsScorer
from .prior import PriorScorer
from .reference_bank import ReferenceCorpus

DIMENSIONS = ("physics", "prior")
# Weakest sub-scores a report lists.
N_WORST = 3


@dataclass
class MotionQualityReport:
    object_type: str
    action_label: str
    subset: str
    score: float                                # median over clips of the clip quality
    dimensions: Dict[str, float]                # median over clips, per dimension
    terms: Dict[str, Dict[str, float]]          # median over clips, per dimension and term
    clips: List[Dict[str, object]]
    worst: List[Tuple[str, float]] = field(default_factory=list)

    @property
    def n_clips(self) -> int:
        return len(self.clips)

    def as_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            **{name: round(value, 4) for name, value in self.dimensions.items()},
            "terms": {dim: {k: round(v, 4) for k, v in terms.items()} for dim, terms in self.terms.items()},
            "worst": [[k, round(v, 4)] for k, v in self.worst],
            "clips": self.clips,
            "meta": {
                "object_type": self.object_type,
                "action_label": self.action_label,
                "subset": self.subset,
                "n_clips": self.n_clips,
            },
        }


class MotionQualityScorer:
    """Score a task's clips on both dimensions against one shared corpus."""

    def __init__(self, dataset_root=None):
        corpus = ReferenceCorpus(tuple(resolve_sources(dataset_root)))
        self.physics = PhysicsScorer(corpus=corpus)
        self.prior = PriorScorer(corpus=corpus)
        self._query_cond_lookup = dict(corpus.cond_lookup)

    def register_cond(self, cond_dict: dict) -> None:
        """Register query skeletons from a custom cond.npy (they are never references)."""
        self.physics.register_cond(cond_dict)
        self.prior.register_cond(cond_dict)
        self._query_cond_lookup.update({str(k): v for k, v in cond_dict.items()})

    def species_lookup(self) -> Dict[str, str]:
        """``{filename token: canonical key}`` over every species this scorer knows."""
        return species_lookup_map(self._query_cond_lookup)

    def evaluate(
        self,
        motions: List[np.ndarray],
        object_type: str,
        action_label: str,
        names: Optional[Sequence[str]] = None,
    ) -> MotionQualityReport:
        names = list(names) if names is not None else [str(i) for i in range(len(motions))]
        physics = self.physics.evaluate(motions, object_type, names=names)
        prior = self.prior.evaluate(motions, object_type, action_label, names=names)
        by_file = {row["file"]: row for row in prior.clips}

        clips = []
        for phys_row in physics.clips:
            prior_row = by_file.get(phys_row["file"])
            if prior_row is None:
                continue
            p, q = float(phys_row["score"]), float(prior_row["score"])
            clips.append({
                "file": phys_row["file"],
                "score": round(float(np.sqrt(max(p, 0.0) * max(q, 0.0))), 4),
                "physics": round(p, 4),
                "prior": round(q, 4),
                "terms": {"physics": phys_row["terms"], "prior": prior_row["terms"]},
                "sub": {**{f"physics/{k}": v for k, v in phys_row["sub"].items()},
                        **{f"prior/{k}": v for k, v in prior_row["sub"].items()}},
            })
        if not clips:
            raise ValueError("no clip was scored on both dimensions")

        sub_keys = sorted({k for c in clips for k in c["sub"]})
        sub_median = {k: float(np.median([c["sub"][k] for c in clips if k in c["sub"]])) for k in sub_keys}
        worst = sorted(((k, v) for k, v in sub_median.items() if v < 1.0), key=lambda kv: kv[1])[:N_WORST]
        return MotionQualityReport(
            object_type=physics.object_type,
            action_label=prior.action_label,
            subset=physics.subset,
            score=float(np.median([c["score"] for c in clips])),
            dimensions={dim: float(np.median([c[dim] for c in clips])) for dim in DIMENSIONS},
            terms={"physics": physics.terms, "prior": prior.terms},
            clips=clips,
            worst=worst,
        )
