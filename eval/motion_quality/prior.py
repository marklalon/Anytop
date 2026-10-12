"""
Data-prior conformity of generated clips: do the limbs move together, and at
the tempo, the way real clips of the action do?

Only features that track motion quality are scored -- the ones that move
toward the real distribution as a model trains: left/right symmetry, limb
coordination and part tempo. Amplitude, speed, posture and root travel are
left out on purpose: a generated clip tends to the average motion, so on those
it is *more* typical than a real one, and scoring them rewards the averaging.

Reference
---------
Corpus clips sharing a head word with the label (``reference_prior_words``),
never the query's own species: the skeletons scored are new ones. Modifiers
only weigh: a clip whose words all match weighs ``LABEL_MATCH_WEIGHTS[0]``, one
with the same head set but other modifiers ``[1]``, one sharing only some heads
``[2]``. Species weigh equally: a species' weight is its best clip weight,
split over its clips by their label weights. Two layers -- every species
(``global``) and those of the query's object_subset (``subset``) -- and a
feature takes the layer it deviates from least: the query may do the action
like its own body plan or like any other.

Scoring
-------
A feature's deviation is its robust z against the reference: distance from the
weighted median over the weighted MAD (floored by ``FEATURE_SCALE_FLOOR``),
counted on the bad side only for the one-sided features. The deviation is then
scored against how far real clips of the same action deviate: sampled corpus
clips of each head word are scored the same way, each against the reference of
its own label without its own species, and the quantiles of those deviations
are the anchors of ``physics.Tolerance``. The calibration is per head word:
an idle's legs need not move in step, a run's must, so how far a real idle
strays says nothing about how far a run may. A label of several heads takes
the widest of their anchors; a head with too few real clips falls back to all
heads pooled. A clip as typical as the median real clip scores 1; a
clip more typical than that gains nothing.

Features
--------
- ``asym:<part>``: |log| ratio of the left and right joints' RMS displacement
  in a sided body part (higher is worse).
- ``agree:<limbs>``: share of the limbs whose dominant frequency is the
  limbs' common one (lower is worse).
- ``lock:<limbs>``: mean pairwise phase-locking value of the limbs' signals
  (lower is worse).
- ``lr_phase:<limbs>``: mean relative phase (0 in step, 1 opposite) of the
  left/right limb pairs.
- ``slip:legs``: log of the root's horizontal speed over the legs' stride
  times their step frequency: travel the feet do not account for.
- ``tempo:<part>``: centroid frequency (cycles per frame) of the part's
  low-passed velocity spectrum. The low-pass keeps jitter -- the physics
  dimension's business -- out of the tempo.

``legs`` are the foot chains of ``joint_parts.prefill_foot_chains``, each
followed by its lowest joint's fore-aft swing; ``wings`` are the wing tips
(the farthest wing joint of each side), followed by their height.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert

from data_loaders.truebones.offline_reference_dataset import resolve_sources
from data_loaders.truebones.truebones_utils.dataset_sources import resolve_species_key
from data_loaders.truebones.truebones_utils.joint_parts import prefill_foot_chains
from data_loaders.truebones.truebones_utils.motion_labels import (
    head_words_in,
    load_motion_metadata,
    vocab_words_in,
)
from data_loaders.truebones.truebones_utils.param_utils import FEATS_LEN

from .physics import SCORE_AT_QUANTILE, PhysicsProfile, Tolerance, rest_positions, species_subset
from .reference_bank import ReferenceCorpus

# Bump when a feature's definition changes: it keys the on-disk corpus cache.
PRIOR_VERSION = 3

MIN_CLIP_FRAMES = 8
CH_POS = slice(0, 3)
# Per-frame motion (rest spans) below which a part counts as still.
SPEED_FLOOR = 2e-3
# Frames trimmed from each end of a phase signal: the Hilbert phase is
# unreliable there.
PHASE_EDGE_FRAMES = 3
# Gaussian width (frames) of the low-pass the tempo is measured on.
TEMPO_LOWPASS_SIGMA = 1.5

# Label match tiers: every word equal / same head set / some head shared.
LABEL_MATCH_WEIGHTS = (1.0, 0.6, 0.25)
# A layer needs this many species and clips having a feature to score it.
MIN_LAYER_SPECIES = 2
MIN_LAYER_CLIPS = 4

# Smallest reference spread per feature kind: the resolution below which two
# values count as the same, so a reference of near-identical clips does not
# make every other value an outlier.
FEATURE_SCALE_FLOOR: Dict[str, float] = {
    "asym": 0.1, "agree": 0.15, "lock": 0.05, "lr_phase": 0.08, "slip": 0.2, "tempo": 0.01,
}
# Features whose deviation counts on one side only: +1 high is bad, -1 low is bad.
ONE_SIDED: Dict[str, int] = {"asym": 1, "agree": -1, "lock": -1}
# Smallest deviation (robust z) a real-clip anchor may sit at.
CALIBRATION_FLOOR = 0.5
# Real clips sampled per head word for the calibration.
CALIBRATION_CLIPS_PER_HEAD = 80
# Real deviations a head word needs before its own anchors replace the pooled ones.
MIN_CALIBRATION_SAMPLES = 20

# Sub-scores are grouped into these terms; the dimension is their geometric mean.
TERMS = ("symmetry", "coordination", "tempo")


def _feature_kind(name: str) -> str:
    return name.split(":", 1)[0]


def _feature_term(name: str) -> str:
    kind = _feature_kind(name)
    if kind == "asym":
        return "symmetry"
    if kind == "tempo":
        return "tempo"
    return "coordination"


# ── Skeleton ─────────────────────────────────────────────────────────────────
def _is_mirror_pair(ja: int, jb: int, partner: Sequence, parents: np.ndarray) -> bool:
    """Whether two limb tips are mirror chains: a joint at or above one is the
    symmetry partner of a joint at or above the other."""

    def ancestors(j):
        out = []
        while j >= 0 and len(out) < 8:
            out.append(j)
            j = int(parents[j])
        return out

    above_b = set(ancestors(jb))
    for j in ancestors(ja):
        pj = partner[j] if j < len(partner) else None
        if pj is not None and int(pj) >= 0 and int(pj) != j and int(pj) in above_b:
            return True
    return False


class PriorProfile(PhysicsProfile):
    """``PhysicsProfile`` plus the sided parts and limbs the features follow."""

    def __init__(self, object_cond: Mapping[str, object]):
        super().__init__(object_cond)
        rest = rest_positions(object_cond)
        parents = np.asarray(object_cond["parents"], dtype=np.int64)
        sides = [str(s) for s in (object_cond.get("joint_side_labels") or [])]
        partner = list(object_cond.get("symmetry_partner_indices") or [])

        self.sided: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for part, joints in self.parts.items():
            left = [int(j) for j in joints if j < len(sides) and sides[j] == "left"]
            right = [int(j) for j in joints if j < len(sides) and sides[j] == "right"]
            if left and right:
                self.sided[part] = (np.asarray(left), np.asarray(right))

        legs = [int(min(chain, key=lambda j: rest[j, 1])) for chain in prefill_foot_chains(object_cond)]
        wings = []
        wing_joints = self.parts.get("wing", np.zeros(0, dtype=np.int64))
        root = rest[self.translation_root_index]
        for side in ("left", "right"):
            joints = [int(j) for j in wing_joints if j < len(sides) and sides[j] == side]
            if joints:
                wings.append(max(joints, key=lambda j: np.linalg.norm(rest[j] - root)))
        self.limbs: Dict[str, List[int]] = {}
        if len(legs) >= 2:
            self.limbs["legs"] = legs
        if len(wings) == 2:
            self.limbs["wings"] = wings
        # Left/right pairs among the limbs, by index into self.limbs[name].
        self.limb_pairs: Dict[str, List[Tuple[int, int]]] = {}
        for name, tips in self.limbs.items():
            if name == "wings":
                self.limb_pairs[name] = [(0, 1)]
                continue
            self.limb_pairs[name] = [
                (a, b) for a in range(len(tips)) for b in range(a + 1, len(tips))
                if _is_mirror_pair(tips[a], tips[b], partner, parents)
            ]


# ── Per-clip features ────────────────────────────────────────────────────────
def _spectral_centroid(vel: np.ndarray) -> float:
    t = vel.shape[0]
    if t < 4:
        return 0.0
    window = np.hanning(t).reshape((t,) + (1,) * (vel.ndim - 1))
    power = np.abs(np.fft.rfft((vel - vel.mean(axis=0)) * window, axis=0)) ** 2
    power = power.reshape(power.shape[0], -1).sum(axis=1)
    power[0] = 0.0
    total = power.sum()
    return float((np.fft.rfftfreq(t) * power).sum() / total) if total > 0 else 0.0


def _limb_features(name: str, signals: np.ndarray, pairs: Sequence[Tuple[int, int]]) -> Dict[str, float]:
    """Coordination features of ``signals`` (n_limbs, T)."""
    n, t = signals.shape
    if t < 2 * PHASE_EDGE_FRAMES + 8:
        return {}
    centred = signals - signals.mean(axis=1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centred * np.hanning(t), axis=1))
    spectrum[:, 0] = 0.0
    _, counts = np.unique(spectrum.argmax(axis=1), return_counts=True)
    feats = {f"agree:{name}": float(counts.max() / n)}
    unit = centred / (centred.std(axis=1, keepdims=True) + 1e-9)
    phase = np.angle(hilbert(unit, axis=1))[:, PHASE_EDGE_FRAMES:t - PHASE_EDGE_FRAMES]
    feats[f"lock:{name}"] = float(np.mean([
        abs(np.mean(np.exp(1j * (phase[a] - phase[b])))) for a in range(n) for b in range(a + 1, n)
    ]))
    if pairs:
        feats[f"lr_phase:{name}"] = float(np.mean([
            abs(np.angle(np.mean(np.exp(1j * (phase[a] - phase[b]))))) / np.pi for a, b in pairs
        ]))
    return feats


def clip_features(motion: np.ndarray, profile: PriorProfile) -> Dict[str, float]:
    """Quality features of one clip (lengths in rest spans, time in frames)."""
    if motion.ndim != 3 or motion.shape[-1] != FEATS_LEN:
        raise ValueError(f"expected (T, J, {FEATS_LEN}), got {motion.shape}")
    if motion.shape[1] != profile.n_joints:
        raise ValueError(f"motion has {motion.shape[1]} joints, skeleton has {profile.n_joints}")
    scale = 1.0 / profile.span
    pos = np.asarray(motion[:, :, CH_POS], dtype=np.float64) * scale
    tri = profile.translation_root_index
    local = pos - pos[:, tri:tri + 1]
    feats: Dict[str, float] = {}

    smooth = gaussian_filter1d(local, TEMPO_LOWPASS_SIGMA, axis=0, mode="nearest")
    for part, joints in profile.parts.items():
        feats[f"tempo:{part}"] = _spectral_centroid(np.diff(smooth[:, joints], axis=0))

    for part, (left, right) in profile.sided.items():
        amp_l = np.sqrt(np.mean((local[:, left] - local[:, left].mean(axis=0)) ** 2))
        amp_r = np.sqrt(np.mean((local[:, right] - local[:, right].mean(axis=0)) ** 2))
        feats[f"asym:{part}"] = float(abs(np.log((amp_l + SPEED_FLOOR) / (amp_r + SPEED_FLOOR))))

    for name, tips in profile.limbs.items():
        axis = 2 if name == "legs" else 1     # legs: fore-aft swing; wings: flap height
        signals = local[:, tips, axis].T
        feats.update(_limb_features(name, signals, profile.limb_pairs.get(name, [])))
        if name == "legs" and signals.shape[1] >= 8:
            t = signals.shape[1]
            spectrum = np.abs(np.fft.rfft((signals - signals.mean(axis=1, keepdims=True)) * np.hanning(t), axis=1))
            spectrum[:, 0] = 0.0
            step_freq = np.fft.rfftfreq(t)[spectrum.sum(axis=0).argmax()]
            stride = np.ptp(signals, axis=1).mean()
            root_speed = np.sqrt(np.mean((np.asarray(motion[:, tri, [9, 11]], dtype=np.float64) * scale) ** 2))
            feats["slip:legs"] = float(np.log((root_speed + SPEED_FLOOR) / (stride * step_freq + SPEED_FLOOR)))
    return feats


# ── Corpus features ──────────────────────────────────────────────────────────
class CorpusFeatures:
    """``clip_features`` and label words of every labelled corpus clip, cached per source."""

    def __init__(self, corpus: ReferenceCorpus):
        self.corpus = corpus
        self.species_of: Dict[str, str] = {}
        for species, by_head in corpus._clips_by_species.items():
            for paths in by_head.values():
                for path in paths:
                    self.species_of[path] = species
        self.features: Dict[str, Dict[str, float]] = {}
        self.label_words: Dict[str, Tuple[str, ...]] = {}
        self._profiles: Dict[str, PriorProfile] = {}
        for source in corpus.sources:
            self._load_source(source)

    def profile(self, species: str) -> PriorProfile:
        profile = self._profiles.get(species)
        if profile is None:
            profile = PriorProfile(self.corpus.cond_lookup[species])
            self._profiles[species] = profile
        return profile

    def _load_source(self, source) -> None:
        metadata = load_motion_metadata(source.root)
        motion_dir = Path(source.motion_dir)
        paths = sorted(p for p in self.species_of if Path(p).parent == motion_dir)
        for path in paths:
            entry = metadata.get(Path(path).name) or {}
            self.label_words[path] = tuple(vocab_words_in(str(entry.get("action_label") or "")))
        digest = hashlib.sha1(f"v{PRIOR_VERSION}".encode())
        for path in paths:
            st = os.stat(path)
            digest.update(f"{Path(path).name}:{st.st_size}:{st.st_mtime_ns}:{self.species_of[path]}".encode())
        fingerprint = digest.hexdigest()
        cache_path = Path(source.root) / "cache" / f"prior_features_v{PRIOR_VERSION}.npz"

        if cache_path.is_file():
            try:
                with np.load(cache_path, allow_pickle=False) as cached:
                    if str(cached["fingerprint"]) == fingerprint:
                        keys = [str(k) for k in cached["keys"]]
                        for name, row in zip(cached["names"], cached["values"]):
                            self.features[str(motion_dir / str(name))] = {
                                k: float(v) for k, v in zip(keys, row) if np.isfinite(v)
                            }
                        return
            except (OSError, KeyError, ValueError):
                pass

        print(f"[prior] featurising {len(paths)} clips of {source.namespace} ...")
        rows: Dict[str, Dict[str, float]] = {}
        for path in paths:
            motion = np.load(path)
            if motion.ndim != 3 or motion.shape[-1] != FEATS_LEN or motion.shape[0] < MIN_CLIP_FRAMES:
                continue
            try:
                rows[path] = clip_features(motion, self.profile(self.species_of[path]))
            except ValueError as exc:
                print(f"[prior]   skip {Path(path).name}: {exc}")
        self.features.update(rows)
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


def label_match_weight(query_words: Sequence[str], clip_words: Sequence[str]) -> float:
    query_heads = set(head_words_in(query_words))
    clip_heads = set(head_words_in(clip_words))
    if not query_heads & clip_heads:
        return 0.0
    if query_heads == clip_heads:
        return LABEL_MATCH_WEIGHTS[0] if set(query_words) == set(clip_words) else LABEL_MATCH_WEIGHTS[1]
    return LABEL_MATCH_WEIGHTS[2]


# ── Reference statistics ─────────────────────────────────────────────────────
def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cdf = np.cumsum(weights[order])
    return float(np.interp(0.5 * cdf[-1], cdf, values[order]))


class ReferenceMatrix:
    """Corpus features as one matrix, for fast per-query reference statistics."""

    def __init__(self, features: CorpusFeatures, cond_lookup: Mapping[str, Mapping[str, object]]):
        self.paths = sorted(features.features)
        self.names = sorted({k for p in self.paths for k in features.features[p]})
        index = {n: i for i, n in enumerate(self.names)}
        self.x = np.full((len(self.paths), len(self.names)), np.nan)
        for row, path in enumerate(self.paths):
            for name, value in features.features[path].items():
                self.x[row, index[name]] = value
        species = [features.species_of[p] for p in self.paths]
        self.species_names = sorted(set(species))
        species_index = {s: i for i, s in enumerate(self.species_names)}
        self.species = np.asarray([species_index[s] for s in species])
        self.subset = np.asarray([species_subset(cond_lookup[s]) for s in species])
        self.label_words = [features.label_words.get(p, ()) for p in self.paths]
        self.floor = np.asarray([FEATURE_SCALE_FLOOR[_feature_kind(n)] for n in self.names])
        self._label_cache: Dict[Tuple[str, ...], np.ndarray] = {}

    def label_weights(self, query_words: Tuple[str, ...]) -> np.ndarray:
        key = tuple(sorted(query_words))
        cached = self._label_cache.get(key)
        if cached is None:
            cached = np.asarray([label_match_weight(query_words, words) for words in self.label_words])
            self._label_cache[key] = cached
        return cached

    def layer_stats(self, clip_weight: np.ndarray, mask: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """``{feature: (median, scale)}`` over the clips in ``mask``, species weighted equally."""
        w = np.where(mask, clip_weight, 0.0)
        stats: Dict[str, Tuple[float, float]] = {}
        for f, name in enumerate(self.names):
            valid = (w > 0) & np.isfinite(self.x[:, f])
            if valid.sum() < MIN_LAYER_CLIPS:
                continue
            species = self.species[valid]
            if len(np.unique(species)) < MIN_LAYER_SPECIES:
                continue
            cw = w[valid]
            total = np.bincount(species, weights=cw, minlength=len(self.species_names))
            best = np.zeros(len(self.species_names))
            np.maximum.at(best, species, cw)
            weights = best[species] * cw / total[species]
            values = self.x[valid, f]
            median = _weighted_median(values, weights)
            mad = _weighted_median(np.abs(values - median), weights)
            stats[name] = (median, max(1.4826 * mad, float(self.floor[f])))
        return stats


def deviation(name: str, value: float, layers: Sequence[Mapping[str, Tuple[float, float]]]) -> Optional[float]:
    """Robust z of ``value`` against the layer it deviates from least (bad side only
    for one-sided features); None when no layer has the feature."""
    side = ONE_SIDED.get(_feature_kind(name), 0)
    devs = []
    for layer in layers:
        if name not in layer:
            continue
        median, scale = layer[name]
        z = (value - median) / scale
        devs.append(abs(z) if side == 0 else max(0.0, side * z))
    return min(devs) if devs else None


# ── Report ───────────────────────────────────────────────────────────────────
@dataclass
class PriorReport:
    object_type: str
    action_label: str
    subset: str
    clips: List[Dict[str, object]]
    score: float
    terms: Dict[str, float]
    worst: List[Tuple[str, float]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "prior": round(self.score, 4),
            "terms": {k: round(v, 4) for k, v in self.terms.items()},
            "worst": [[k, round(v, 4)] for k, v in self.worst],
            "clips": self.clips,
            "meta": {
                "object_type": self.object_type,
                "action_label": self.action_label,
                "subset": self.subset,
                "n_clips": len(self.clips),
            },
        }


class PriorScorer:
    """Score clips' quality features against the layered reference of their action,
    calibrated by how far real clips deviate from theirs."""

    def __init__(self, dataset_root=None, seed: int = 0, corpus: Optional[ReferenceCorpus] = None):
        self._corpus = corpus if corpus is not None else ReferenceCorpus(tuple(resolve_sources(dataset_root)))
        self._features = CorpusFeatures(self._corpus)
        self._matrix = ReferenceMatrix(self._features, self._corpus.cond_lookup)
        self._query_cond_lookup: Dict[str, Mapping[str, object]] = dict(self._corpus.cond_lookup)
        self._query_profiles: Dict[str, PriorProfile] = {}
        self._layer_cache: Dict[tuple, List[Dict[str, Tuple[float, float]]]] = {}
        self.calibration, self.head_calibration = self._calibrate(seed)

    # ── Query skeletons ──────────────────────────────────────────────────────
    def register_cond(self, cond_dict: dict) -> None:
        """Register query skeletons from a custom cond.npy (they are never references)."""
        for key, entry in cond_dict.items():
            self._query_cond_lookup[str(key)] = entry
            self._query_profiles.pop(str(key), None)

    def _resolve_object_type_key(self, object_type: str) -> str:
        resolved = resolve_species_key(self._query_cond_lookup, object_type)
        if resolved is None:
            raise KeyError(f"Unknown object_type {object_type!r} in cond.npy")
        return resolved

    def _query_profile(self, key: str) -> PriorProfile:
        profile = self._query_profiles.get(key)
        if profile is None:
            profile = PriorProfile(self._query_cond_lookup[key])
            self._query_profiles[key] = profile
        return profile

    # ── Reference ────────────────────────────────────────────────────────────
    def _layers(self, query_words: Tuple[str, ...], subset: str, exclude: Optional[str]) -> List[Dict[str, Tuple[float, float]]]:
        cache_key = (tuple(sorted(query_words)), subset, exclude)
        cached = self._layer_cache.get(cache_key)
        if cached is not None:
            return cached
        m = self._matrix
        weight = m.label_weights(query_words)
        keep = weight > 0
        if exclude in m.species_names:
            keep &= m.species != m.species_names.index(exclude)
        layers = [m.layer_stats(weight, keep)]
        in_subset = keep & (m.subset == subset)
        if len(np.unique(m.species[in_subset])) >= MIN_LAYER_SPECIES:
            layers.append(m.layer_stats(weight, in_subset))
        self._layer_cache[cache_key] = layers
        return layers

    def _calibrate(self, seed: int) -> Tuple[Dict[str, Tolerance], Dict[str, Dict[str, Tolerance]]]:
        """Tolerances from the deviations of sampled real clips, each against the
        reference of its own label without its own species: ``({feature:
        pooled}, {head word: {feature: tolerance}})``."""
        m = self._matrix
        rng = np.random.default_rng(seed)
        by_head: Dict[str, List[int]] = {}
        for row, words in enumerate(m.label_words):
            heads = head_words_in(words)
            if heads:
                by_head.setdefault(heads[0], []).append(row)
        pooled: Dict[str, List[float]] = {}
        per_head: Dict[str, Dict[str, List[float]]] = {}
        for head in sorted(by_head):
            rows = by_head[head]
            for row in rng.choice(rows, min(CALIBRATION_CLIPS_PER_HEAD, len(rows)), replace=False):
                species = m.species_names[m.species[row]]
                layers = self._layers(tuple(m.label_words[row]), str(m.subset[row]), species)
                for f, name in enumerate(m.names):
                    if np.isfinite(m.x[row, f]):
                        d = deviation(name, float(m.x[row, f]), layers)
                        if d is not None:
                            pooled.setdefault(name, []).append(d)
                            per_head.setdefault(head, {}).setdefault(name, []).append(d)
        qs = [q for q, _ in SCORE_AT_QUANTILE]

        def tolerance(values, layer):
            return Tolerance.from_quantiles(np.quantile(values, qs), CALIBRATION_FLOOR, layer)

        return (
            {name: tolerance(v, "real") for name, v in pooled.items() if len(v) >= MIN_CALIBRATION_SAMPLES},
            {
                head: {name: tolerance(v, f"real:{head}") for name, v in devs.items() if len(v) >= MIN_CALIBRATION_SAMPLES}
                for head, devs in per_head.items()
            },
        )

    def _tolerances(self, query_words: Sequence[str]) -> Dict[str, Tolerance]:
        """Per-feature tolerance for a label: the widest anchors of its head words,
        the pooled ones for a feature none of its heads calibrated."""
        tables = [self.head_calibration[h] for h in head_words_in(query_words) if h in self.head_calibration]
        out: Dict[str, Tolerance] = {}
        for name, pooled in self.calibration.items():
            own = [t[name] for t in tables if name in t]
            out[name] = max(own, key=lambda t: t.anchors[-1]) if own else pooled
        return out

    # ── Scoring ──────────────────────────────────────────────────────────────
    def score_features(self, feats: Mapping[str, float], layers, tolerances: Mapping[str, Tolerance]) -> Dict[str, object]:
        devs: Dict[str, float] = {}
        sub: Dict[str, float] = {}
        for name, value in feats.items():
            d = deviation(name, value, layers)
            if d is None or name not in tolerances:
                continue
            devs[name] = d
            sub[name] = tolerances[name].score(d)
        terms: Dict[str, float] = {}
        for term in TERMS:
            values = [max(v, 1e-6) for n, v in sub.items() if _feature_term(n) == term]
            if values:
                terms[term] = float(np.exp(np.mean(np.log(values))))
        score = float(np.exp(np.mean(np.log(list(terms.values()))))) if terms else 1.0
        return {
            "score": round(score, 4),
            "terms": {k: round(v, 4) for k, v in terms.items()},
            "sub": {k: round(v, 4) for k, v in sub.items()},
            "deviation": {k: round(v, 3) for k, v in devs.items()},
            "features": {k: round(v, 5) for k, v in feats.items()},
        }

    def evaluate(
        self,
        motions: List[np.ndarray],
        object_type: str,
        action_label: str,
        names: Optional[Sequence[str]] = None,
    ) -> PriorReport:
        """Score ``motions`` against the reference of ``action_label``, without the
        query's own species."""
        key = self._resolve_object_type_key(object_type)
        profile = self._query_profile(key)
        subset = species_subset(self._query_cond_lookup[key])
        query_words = tuple(vocab_words_in(str(action_label)))
        layers = self._layers(query_words, subset, key)
        tolerances = self._tolerances(query_words)
        if not layers[0]:
            raise ValueError(f"no reference clips for action {action_label!r}")
        clips = []
        for i, motion in enumerate(motions):
            if motion.ndim != 3 or motion.shape[-1] != FEATS_LEN or motion.shape[0] < MIN_CLIP_FRAMES:
                continue
            row = self.score_features(clip_features(np.asarray(motion, dtype=np.float32), profile), layers, tolerances)
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
        return PriorReport(
            object_type=key,
            action_label=str(action_label).strip(),
            subset=subset,
            clips=clips,
            score=float(np.median([c["score"] for c in clips])),
            terms=terms,
            worst=worst,
        )
