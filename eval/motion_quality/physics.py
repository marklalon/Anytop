"""
Physical plausibility of generated clips, independent of the action label.

Each clip is measured on a few physical statistics: how much its bones stretch,
how smooth each body part moves and whether any frame jumps. A statistic is
scored against the spread of the same statistic over real clips, so "natural"
means "no worse than the training data", not "perfect": real clips stretch
bones a little and real insects twitch more than real elephants.

Tolerance layers
----------------
Every statistic's tolerance comes from real clips, each species weighted
equally, in two layers:

- ``global``: every species;
- ``subset``: the query's object_subset (its first species tag), shrunk toward
  the global layer when the subset holds few species.

The query's own clips are never a layer: the skeletons scored are new ones.

Smoothness takes the wider layer: how jerky a motion may be is a trait of the
body plan, and a query should not lose for twitching like an insect when it is
one. Rigidity takes the subset layer: the global tail is set by
squash-and-stretch packs, which must not excuse every other body plan. A statistic is scored by where it
falls in its layer: up to the median it scores 1, the score falls through the
anchors ``SCORE_AT_QUANTILE`` (log score linear in log value between them), and
past the last one it halves for every doubling (``TAIL_EXPONENT``). So a clip a little worse
than the typical real clip already loses a little, and one past the real tail
keeps losing the further out it is. Discontinuity uses the global layer only:
a frame jump is never a trait of a body plan.

Statistics
----------
- ``stretch_mean``: bone-length-weighted mean of |L/L0 - 1| over the clip.
- ``stretch_long``: the same over the longest third of the bones, where a
  stretch is visible (a stretched toe tip is not).
- ``jerk:<part>`` / ``snap:<part>``: RMS of the 3rd / 4th frame difference of a
  body part's root-relative positions over the RMS of its velocity --
  dimensionless, so a part that moves a lot is not penalised for it.
- ``flatness:<part>``: spectral flatness of the part's velocity: noise is
  flat, deliberate motion is peaked.
- ``spike``: the largest single-frame acceleration of the body over the
  median acceleration of its low-passed motion. The low-passed level is the
  deliberate motion's, which jitter barely moves, so a skipped frame, a popping
  limb and jitter all raise the ratio; a ratio over the clip's own raw
  acceleration would fall under jitter instead.

Body parts come from ``joint_parts.prefill_joint_parts``.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

from data_loaders.truebones.offline_reference_dataset import resolve_sources
from data_loaders.truebones.truebones_utils.dataset_sources import resolve_species_key
from data_loaders.truebones.truebones_utils.joint_parts import JOINT_PARTS, prefill_joint_parts
from data_loaders.truebones.truebones_utils.param_utils import FEATS_LEN

from .bone_length_drift import resolve_comparison_edges
from .reference_bank import ReferenceCorpus

# Bump when a statistic's definition changes: it keys the on-disk corpus cache.
PHYSICS_VERSION = 2

MIN_CLIP_FRAMES = 8
CH_POS = slice(0, 3)

# Per-frame motion (rest spans) below which a part counts as still: it keeps an
# idle part's ratios off a division by ~0.
SPEED_FLOOR = 2e-3
# Longest fraction of the bones ``stretch_long`` looks at.
LONG_BONE_FRACTION = 1.0 / 3.0
# Pseudo-species count a subset layer is shrunk toward the global layer with.
SUBSET_PRIOR_SPECIES = 5.0
# Score ~ value^-TAIL_EXPONENT past the last anchor: a fixed decay, so a layer's
# fat or thin tail does not decide how fast a value beyond it loses.
TAIL_EXPONENT = 1.0
# Smallest tolerance a statistic gets, whatever the corpus says: real clips of a
# rigid rig have zero stretch, which would make any stretch infinitely bad.
TOLERANCE_FLOOR: Dict[str, float] = {"stretch_mean": 0.005, "stretch_long": 0.005}
# Score at each quantile of the tolerance layer; at or under the first, 1.
SCORE_AT_QUANTILE: Tuple[Tuple[float, float], ...] = ((0.50, 1.0), (0.75, 0.9), (0.90, 0.75), (0.99, 0.45))
# Smallest ratio between consecutive quantile anchors: a layer whose tail is as
# tight as its body would otherwise drop the score on any excess.
MIN_ANCHOR_RATIO = 1.15
# Gaussian width (frames) of the low-pass that sets ``spike``'s denominator.
SPIKE_LOWPASS_SIGMA = 1.5

# Sub-scores are grouped into these terms; the dimension is their geometric mean.
TERMS = ("rigidity", "smoothness", "continuity")


def _stat_term(stat: str) -> str:
    if stat.startswith("stretch_"):
        return "rigidity"
    if stat == "spike":
        return "continuity"
    return "smoothness"


# ── Per-clip statistics ──────────────────────────────────────────────────────
def rest_positions(object_cond: Mapping[str, object]) -> np.ndarray:
    """World positions of the rest pose, from the cond's parents and offsets."""
    parents = np.asarray(object_cond["parents"], dtype=np.int64)
    offsets = np.asarray(object_cond["offsets"], dtype=np.float64)
    pos = np.zeros_like(offsets)
    for joint in range(len(parents)):
        parent = int(parents[joint])
        pos[joint] = offsets[joint] + (pos[parent] if parent >= 0 else 0.0)
    return pos


def rest_span(object_cond: Mapping[str, object]) -> float:
    """Largest extent of the rest pose -- the length unit statistics are measured in."""
    positions = rest_positions(object_cond)
    span = float(np.ptp(positions, axis=0).max()) if len(positions) else 0.0
    if not np.isfinite(span) or span <= 1e-6:
        raise ValueError(f"degenerate rest pose (span={span})")
    return span


class PhysicsProfile:
    """What the statistics need from one skeleton's cond entry."""

    def __init__(self, object_cond: Mapping[str, object]):
        parents = np.asarray(object_cond["parents"], dtype=np.int32)
        offsets = np.asarray(object_cond["offsets"], dtype=np.float64)
        self.n_joints = len(parents)
        self.span = rest_span(object_cond)
        self.translation_root_index = int(object_cond.get("translation_root_index", 0) or 0)
        self.bone_parent, self.bone_child = resolve_comparison_edges(parents, offsets)
        self.bone_rest = np.linalg.norm(offsets[self.bone_child], axis=-1) / self.span
        order = np.argsort(-self.bone_rest, kind="stable")
        self.long_bones = order[: max(1, int(np.ceil(len(order) * LONG_BONE_FRACTION)))]

        prefill = prefill_joint_parts(object_cond)
        names = [str(name) for name in object_cond["joints_names"]]
        part_of = [prefill.get(name, {}).get("part") for name in names]
        self.parts: Dict[str, np.ndarray] = {}
        for part in JOINT_PARTS:
            joints = [j for j, p in enumerate(part_of) if p == part and j != self.translation_root_index]
            if joints:
                self.parts[part] = np.asarray(joints, dtype=np.int64)
        self.body = np.asarray(
            [j for j, p in enumerate(part_of) if p in JOINT_PARTS and j != self.translation_root_index]
            or [j for j in range(self.n_joints) if j != self.translation_root_index]
            or [0],
            dtype=np.int64,
        )


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def _spectral_flatness(vel: np.ndarray) -> float:
    """Flatness of the power spectrum of ``vel`` (T, ...) along time, DC excluded,
    pooled over every channel."""
    t = vel.shape[0]
    if t < 4:
        return 0.0
    window = np.hanning(t).reshape((t,) + (1,) * (vel.ndim - 1))
    power = np.abs(np.fft.rfft((vel - vel.mean(axis=0)) * window, axis=0)) ** 2
    power = power[1:].reshape(power.shape[0] - 1, -1).sum(axis=1) + 1e-20
    return float(np.exp(np.mean(np.log(power))) / np.mean(power))


def clip_statistics(motion: np.ndarray, profile: PhysicsProfile) -> Dict[str, float]:
    """Physical statistics of one clip (lengths in rest spans, time in frames)."""
    if motion.ndim != 3 or motion.shape[-1] != FEATS_LEN:
        raise ValueError(f"expected (T, J, {FEATS_LEN}), got {motion.shape}")
    if motion.shape[1] != profile.n_joints:
        raise ValueError(f"motion has {motion.shape[1]} joints, skeleton has {profile.n_joints}")
    pos = np.asarray(motion[:, :, CH_POS], dtype=np.float64) / profile.span
    stats: Dict[str, float] = {}

    if profile.bone_rest.size:
        length = np.linalg.norm(pos[:, profile.bone_child] - pos[:, profile.bone_parent], axis=-1)
        stretch = np.abs(length / profile.bone_rest - 1.0).mean(axis=0)
        weight = profile.bone_rest / profile.bone_rest.sum()
        stats["stretch_mean"] = float(stretch @ weight)
        long_w = profile.bone_rest[profile.long_bones]
        stats["stretch_long"] = float(stretch[profile.long_bones] @ (long_w / long_w.sum()))

    tri = profile.translation_root_index
    local = pos - pos[:, tri:tri + 1]
    for part, joints in profile.parts.items():
        x = local[:, joints]
        d1 = np.diff(x, n=1, axis=0)
        speed = _rms(d1) + SPEED_FLOOR
        stats[f"jerk:{part}"] = _rms(np.diff(x, n=3, axis=0)) / speed
        stats[f"snap:{part}"] = _rms(np.diff(x, n=4, axis=0)) / speed
        stats[f"flatness:{part}"] = _spectral_flatness(d1)

    body = pos[:, profile.body]
    if body.shape[0] >= 3:
        accel = np.linalg.norm(np.diff(body, n=2, axis=0), axis=-1).mean(axis=1)
        smooth = gaussian_filter1d(body, SPIKE_LOWPASS_SIGMA, axis=0, mode="nearest")
        level = np.linalg.norm(np.diff(smooth, n=2, axis=0), axis=-1).mean(axis=1)
        stats["spike"] = float(accel.max() / (np.median(level) + SPEED_FLOOR * 0.1))
    return stats


# ── Corpus statistics ────────────────────────────────────────────────────────
def species_subset(object_cond: Mapping[str, object]) -> str:
    tags = object_cond.get("species_tags") or ()
    return str(tags[0]).strip().lower() if len(tags) else ""


class CorpusPhysics:
    """``clip_statistics`` of every corpus clip, cached per dataset source."""

    def __init__(self, corpus: ReferenceCorpus):
        self.corpus = corpus
        self.species_of: Dict[str, str] = {}
        for species, by_head in corpus._clips_by_species.items():
            for paths in by_head.values():
                for path in paths:
                    self.species_of[path] = species
        self.stats: Dict[str, Dict[str, float]] = {}
        self._profiles: Dict[str, PhysicsProfile] = {}
        for source in corpus.sources:
            self._load_source(source)

    def profile(self, species: str) -> PhysicsProfile:
        profile = self._profiles.get(species)
        if profile is None:
            profile = PhysicsProfile(self.corpus.cond_lookup[species])
            self._profiles[species] = profile
        return profile

    def _load_source(self, source) -> None:
        motion_dir = Path(source.motion_dir)
        paths = sorted(p for p in self.species_of if Path(p).parent == motion_dir)
        digest = hashlib.sha1(f"v{PHYSICS_VERSION}".encode())
        for path in paths:
            st = os.stat(path)
            digest.update(f"{Path(path).name}:{st.st_size}:{st.st_mtime_ns}:{self.species_of[path]}".encode())
        fingerprint = digest.hexdigest()
        cache_path = Path(source.root) / "cache" / f"physics_stats_v{PHYSICS_VERSION}.npz"

        if cache_path.is_file():
            try:
                with np.load(cache_path, allow_pickle=False) as cached:
                    if str(cached["fingerprint"]) == fingerprint:
                        keys = [str(k) for k in cached["keys"]]
                        values = cached["values"]
                        for i, name in enumerate(cached["names"]):
                            row = values[i]
                            self.stats[str(motion_dir / str(name))] = {
                                k: float(v) for k, v in zip(keys, row) if np.isfinite(v)
                            }
                        return
            except (OSError, KeyError, ValueError):
                pass

        print(f"[physics] measuring {len(paths)} clips of {source.namespace} ...")
        rows: Dict[str, Dict[str, float]] = {}
        for path in paths:
            motion = np.load(path)
            if motion.ndim != 3 or motion.shape[-1] != FEATS_LEN or motion.shape[0] < MIN_CLIP_FRAMES:
                continue
            try:
                rows[path] = clip_statistics(motion, self.profile(self.species_of[path]))
            except ValueError as exc:
                print(f"[physics]   skip {Path(path).name}: {exc}")
        self.stats.update(rows)
        keys = sorted({k for row in rows.values() for k in row})
        values = np.full((len(rows), len(keys)), np.nan)
        for i, row in enumerate(rows.values()):
            for j, key in enumerate(keys):
                values[i, j] = row.get(key, np.nan)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            fingerprint=np.asarray(fingerprint),
            names=np.asarray([Path(p).name for p in rows]),
            keys=np.asarray(keys),
            values=values,
        )


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray, qs: Sequence[float]) -> np.ndarray:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cdf = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(qs, cdf, v)


@dataclass
class Tolerance:
    anchors: np.ndarray          # values at the SCORE_AT_QUANTILE quantiles, increasing
    layer: str

    @classmethod
    def from_quantiles(cls, quantiles: Sequence[float], floor: float, layer: str) -> "Tolerance":
        anchors = []
        for q in quantiles:
            low = max(floor, anchors[-1] * MIN_ANCHOR_RATIO) if anchors else floor
            anchors.append(max(float(q), low))
        return cls(np.asarray(anchors), layer)

    def score(self, value: float) -> float:
        """Log score, linear in log value between the anchors, extrapolated past the last."""
        if value <= self.anchors[0]:
            return 1.0
        log_v = np.log(self.anchors)
        log_s = np.log([s for _, s in SCORE_AT_QUANTILE])
        x = np.log(value)
        if x <= log_v[-1]:
            return float(np.exp(np.interp(x, log_v, log_s)))
        return float(np.exp(log_s[-1] - TAIL_EXPONENT * (x - log_v[-1])))


# ── Report ───────────────────────────────────────────────────────────────────
@dataclass
class PhysicsReport:
    object_type: str
    subset: str
    clips: List[Dict[str, object]]               # per clip: score, terms, stats
    score: float                                 # median over clips
    terms: Dict[str, float]                      # median per term
    worst: List[Tuple[str, float]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "physics": round(self.score, 4),
            "terms": {k: round(v, 4) for k, v in self.terms.items()},
            "worst": [[k, round(v, 4)] for k, v in self.worst],
            "clips": self.clips,
            "meta": {"object_type": self.object_type, "subset": self.subset, "n_clips": len(self.clips)},
        }


class PhysicsScorer:
    """Score clips' physical plausibility against layered corpus tolerances."""

    def __init__(self, dataset_root=None, corpus: Optional[ReferenceCorpus] = None):
        self._corpus = corpus if corpus is not None else ReferenceCorpus(tuple(resolve_sources(dataset_root)))
        self._physics = CorpusPhysics(self._corpus)
        self._query_cond_lookup: Dict[str, Mapping[str, object]] = dict(self._corpus.cond_lookup)
        self._custom_cond_keys: set[str] = set()
        self._query_profiles: Dict[str, PhysicsProfile] = {}
        self._tolerances: Dict[Tuple[str, str], Dict[str, Tolerance]] = {}
        self._species_clips: Dict[str, List[str]] = {}
        for path, species in self._physics.species_of.items():
            if path in self._physics.stats:
                self._species_clips.setdefault(species, []).append(path)
        self._layer_cache: Dict[str, Dict[str, Tuple[float, float, int]]] = {}

    # ── Query skeletons ──────────────────────────────────────────────────────
    def register_cond(self, cond_dict: dict) -> None:
        """Register query skeletons from a custom cond.npy (they are never references)."""
        for key, entry in cond_dict.items():
            key_str = str(key)
            self._query_cond_lookup[key_str] = entry
            self._custom_cond_keys.add(key_str)
            self._query_profiles.pop(key_str, None)
            for cache_key in [k for k in self._tolerances if k[0] == key_str]:
                del self._tolerances[cache_key]

    def _resolve_object_type_key(self, object_type: str) -> str:
        resolved = resolve_species_key(self._query_cond_lookup, object_type)
        if resolved is None:
            raise KeyError(f"Unknown object_type {object_type!r} in cond.npy")
        return resolved

    def _query_profile(self, key: str) -> PhysicsProfile:
        profile = self._query_profiles.get(key)
        if profile is None:
            profile = PhysicsProfile(self._query_cond_lookup[key])
            self._query_profiles[key] = profile
        return profile

    # ── Tolerances ───────────────────────────────────────────────────────────
    def _layer(self, name: str, species: Sequence[str]) -> Dict[str, Tuple[np.ndarray, int]]:
        """``{stat: (quantiles at SCORE_AT_QUANTILE, n_species)}`` over ``species``'
        clips, each species weighted equally."""
        cached = self._layer_cache.get(name)
        if cached is not None:
            return cached
        values: Dict[str, List[float]] = {}
        weights: Dict[str, List[float]] = {}
        members: Dict[str, set] = {}
        for sp in species:
            clips = self._species_clips.get(sp, [])
            for path in clips:
                for stat, value in self._physics.stats[path].items():
                    values.setdefault(stat, []).append(value)
                    weights.setdefault(stat, []).append(1.0 / len(clips))
                    members.setdefault(stat, set()).add(sp)
        qs = [q for q, _ in SCORE_AT_QUANTILE]
        layer = {}
        for stat, v in values.items():
            layer[stat] = (_weighted_quantiles(np.asarray(v), np.asarray(weights[stat]), qs), len(members[stat]))
        self._layer_cache[name] = layer
        return layer

    def _tolerance_table(self, key: str) -> Dict[str, Tolerance]:
        subset = species_subset(self._query_cond_lookup[key])
        cache_key = (key, subset)
        cached = self._tolerances.get(cache_key)
        if cached is not None:
            return cached
        all_species = list(self._species_clips)
        global_layer = self._layer("global", all_species)
        subset_species = [s for s in all_species if species_subset(self._corpus.cond_lookup[s]) == subset]
        subset_layer = self._layer(f"subset:{subset}", subset_species) if subset_species else {}

        table: Dict[str, Tolerance] = {}
        for stat, (g, _) in global_layer.items():
            term = _stat_term(stat)
            candidates = [(g, "global")]
            if term != "continuity" and stat in subset_layer:
                sq, n = subset_layer[stat]
                lam = n / (n + SUBSET_PRIOR_SPECIES)
                candidates.append((lam * sq + (1 - lam) * g, "subset"))
            if term == "rigidity":
                quantiles, layer = candidates[-1]
            else:
                quantiles, layer = max(candidates, key=lambda c: c[0][-1])
            table[stat] = Tolerance.from_quantiles(quantiles, TOLERANCE_FLOOR.get(stat, 1e-6), layer)
        self._tolerances[cache_key] = table
        return table

    # ── Scoring ──────────────────────────────────────────────────────────────
    def score_clip(self, motion: np.ndarray, key: str) -> Dict[str, object]:
        table = self._tolerance_table(key)
        stats = clip_statistics(np.asarray(motion, dtype=np.float32), self._query_profile(key))
        sub: Dict[str, float] = {stat: table[stat].score(v) for stat, v in stats.items() if stat in table}
        terms: Dict[str, float] = {}
        rigidity = [sub[s] for s in ("stretch_mean", "stretch_long") if s in sub]
        if rigidity:
            terms["rigidity"] = float(min(rigidity))
        parts = sorted({s.split(":", 1)[1] for s in sub if ":" in s})
        if parts:
            per_part = [
                float(np.exp(np.mean(np.log([max(sub[f"{m}:{p}"], 1e-6) for m in ("jerk", "snap", "flatness")
                                              if f"{m}:{p}" in sub]))))
                for p in parts
            ]
            terms["smoothness"] = float(np.exp(np.mean(np.log(per_part))))
        if "spike" in sub:
            terms["continuity"] = sub["spike"]
        score = float(np.exp(np.mean(np.log([max(v, 1e-6) for v in terms.values()])))) if terms else 0.0
        return {
            "score": round(score, 4),
            "terms": {k: round(v, 4) for k, v in terms.items()},
            "sub": {k: round(v, 4) for k, v in sub.items()},
            "stats": {k: round(v, 5) for k, v in stats.items()},
        }

    def evaluate(self, motions: List[np.ndarray], object_type: str, names: Optional[Sequence[str]] = None) -> PhysicsReport:
        key = self._resolve_object_type_key(object_type)
        clips = []
        for i, motion in enumerate(motions):
            if motion.ndim != 3 or motion.shape[-1] != FEATS_LEN or motion.shape[0] < MIN_CLIP_FRAMES:
                continue
            row = self.score_clip(motion, key)
            row["file"] = names[i] if names else str(i)
            clips.append(row)
        if not clips:
            raise ValueError(f"no valid (T>={MIN_CLIP_FRAMES}, J, {FEATS_LEN}) clip to score")
        terms = {
            term: float(np.median([c["terms"][term] for c in clips if term in c["terms"]]))
            for term in TERMS if any(term in c["terms"] for c in clips)
        }
        sub_keys = sorted({k for c in clips for k in c["sub"]})
        sub_median = {k: float(np.median([c["sub"][k] for c in clips if k in c["sub"]])) for k in sub_keys}
        worst = sorted(((k, v) for k, v in sub_median.items() if v < 1.0), key=lambda kv: kv[1])[:3]
        return PhysicsReport(
            object_type=key,
            subset=species_subset(self._query_cond_lookup[key]),
            clips=clips,
            score=float(np.median([c["score"] for c in clips])),
            terms=terms,
            worst=worst,
        )
