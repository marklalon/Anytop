"""Morphology (form-class) expert adapters for AnyTop.

The "species expert" is really a *morphology-class* expert: instead of one set
of parameters per species, a small bottleneck adapter is learned per coarse
morphology group. Every ``object_type`` is routed to one of a fixed, ordered
registry of groups via ``species_tags.jsonl`` (``motion_tags[0]``).

Design constraints (see ``shared_species_expert_plan.md``):

* The adapter residual is **zero at step 0** (last Linear zero-init, scale fixed
  at 1.0), so loading a pre-expert checkpoint is byte-identical to the baseline.
* Routing is **dense** (compute-all-groups-then-select) so the module keeps
  static shapes and stays torch.compile / cudagraph friendly -- no
  data-dependent Python masking over the batch.
* The group registry order is **permanently fixed**; new morphology classes are
  only ever appended, never reordered, so old adapter weights never get
  misrouted.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

# Permanently fixed registry. Order MUST NOT change: a checkpoint's
# morphology_adapter[i] is bound to position i. New morphology classes are appended
# at the end only (see plan: "registry 扩展规则").
MORPHOLOGY_GROUPS: Tuple[str, ...] = (
    "Quadruped",
    "Biped",
    "Multiped",
    "Winged",
    "Serpentine",
    "Aquatic",
)

_DEFAULT_TAGS_PATH = os.path.join(
    "dataset", "truebones", "zoo", "truebones_processed", "species_tags.jsonl"
)


class MorphologyAdapter(nn.Module):
    """Bottleneck residual adapter for a single morphology group.

    ``forward`` returns the *residual* only; the caller adds it to the hidden
    state. The final Linear is zero-initialized and the residual scale is fixed
    at 1.0 -- this is the only permitted initialization. Do not add a learnable
    scale initialized to 0: ``scale=0`` together with ``last_linear=0`` would
    make the block's gradient identically 0, and it would never train.
    """

    def __init__(self, dim: int, bottleneck: int = 64, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, dim),
        )
        # Zero-init the last Linear: residual is exactly 0 at step 0.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MorphologyExpertBank(nn.Module):
    """A bank of ``num_groups`` :class:`MorphologyAdapter`, dense-routed per sample.

    ``forward(x, group_ids)`` returns the per-sample-selected residual (NOT
    ``x + residual``); the decoder layer adds it. ``x`` is the decoder hidden
    state laid out as ``(frames, batch, joints, feature)`` -- the morphology
    group dimension lives on the batch axis (dim 1).

    Every group's adapter is evaluated on the whole batch and the right one is
    gathered back per sample. This keeps shapes static (all adapter params used
    every step), matching the ``global_energy_active`` ``torch.where`` pattern
    already used in this codebase, at the cost of computing G small adapters per
    step (cheap relative to attention).
    """

    def __init__(self, dim: int, num_groups: int, bottleneck: int = 64, dropout: float = 0.05):
        super().__init__()
        self.num_groups = int(num_groups)
        self.adapters = nn.ModuleList(
            [MorphologyAdapter(dim, bottleneck=bottleneck, dropout=dropout) for _ in range(self.num_groups)]
        )

    def forward(self, x: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        # x: (T, B, J, C); group_ids: (B,) long in [0, num_groups)
        if group_ids is None:
            raise ValueError("MorphologyExpertBank.forward requires group_ids")
        T, B, J, C = x.shape
        group_ids = group_ids.to(device=x.device, dtype=torch.long).reshape(-1)
        if group_ids.numel() != B:
            raise ValueError(
                f"group_ids batch dim must match motion batch size, got {group_ids.numel()} for batch {B}"
            )
        # (G, T, B, J, C): every adapter on the whole batch.
        outs = torch.stack([adapter(x) for adapter in self.adapters], dim=0)
        idx = group_ids.view(1, 1, B, 1, 1).expand(1, T, B, J, C)
        sel = torch.gather(outs, 0, idx).squeeze(0)  # (T, B, J, C)
        return sel


def _normalize_layers_spec(layers_spec: str, num_layers: int) -> int:
    """Return the number of *trailing* layers that carry a morphology expert.

    Accepts ``"lastN"`` (e.g. ``"last4"``) or ``"allN"`` / ``"all"``.
    """
    spec = str(layers_spec).strip().lower()
    if spec in ("all", f"all{num_layers}"):
        return num_layers
    if spec.startswith("all"):
        # all8 with num_layers=8, or a mismatched allK -> clamp to num_layers
        return num_layers
    if spec.startswith("last"):
        try:
            n = int(spec[len("last"):])
        except ValueError as exc:
            raise ValueError(f"Invalid morphology_expert_layers spec: {layers_spec!r}") from exc
        if n <= 0:
            raise ValueError(f"morphology_expert_layers count must be >= 1, got {layers_spec!r}")
        return min(n, num_layers)
    raise ValueError(
        f"Unrecognized morphology_expert_layers spec {layers_spec!r}; expected 'lastN' or 'allN'."
    )


def resolve_morphology_ids(
    tags_path: str | None = None,
    groups: Sequence[str] = MORPHOLOGY_GROUPS,
) -> Tuple[Tuple[str, ...], Dict[str, int]]:
    """Build the ``object_type -> group_id`` table from ``species_tags.jsonl``.

    Each jsonl record's ``motion_tags[0]`` is the morphology group for that
    ``species`` (== ``object_type``). Group ids follow the fixed ``groups``
    registry order. An object_type whose group is not in the registry, or a
    group that never appears, is a hard error (see plan risk 4).
    """
    groups = tuple(groups)
    group_to_id = {name: i for i, name in enumerate(groups)}

    if tags_path is None:
        tags_path = _DEFAULT_TAGS_PATH
    if not os.path.isfile(tags_path):
        raise FileNotFoundError(
            f"morphology tags file not found: {tags_path}. Pass --morphology_tags_path."
        )

    object_type_to_group_id: Dict[str, int] = {}
    with open(tags_path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            object_type = str(record["species"])
            motion_tags = record.get("motion_tags") or []
            if not motion_tags:
                raise ValueError(
                    f"{tags_path}:{line_no} '{object_type}' has no motion_tags; cannot route."
                )
            group_name = str(motion_tags[0])
            if group_name not in group_to_id:
                raise ValueError(
                    f"{tags_path}:{line_no} '{object_type}' maps to unknown morphology group "
                    f"'{group_name}'. Known groups: {list(groups)}. Append new classes to "
                    f"MORPHOLOGY_GROUPS (end only) before training on them."
                )
            object_type_to_group_id[object_type] = group_to_id[group_name]

    if not object_type_to_group_id:
        raise ValueError(f"No object_type -> group entries parsed from {tags_path}.")

    return groups, object_type_to_group_id


def validate_morphology_registry(saved_groups: Sequence[str]) -> Tuple[str, ...]:
    """Validate a checkpoint's saved group registry against the live constant.

    The registry is append-only: a saved registry must be a *prefix* of the
    current :data:`MORPHOLOGY_GROUPS` (new classes may have been appended since,
    but nothing may be reordered or relabeled, or position ``i`` would bind to
    the wrong adapter). Returns the saved registry as a tuple on success.
    """
    saved = tuple(str(g) for g in saved_groups)
    if saved != MORPHOLOGY_GROUPS[: len(saved)]:
        raise ValueError(
            "Saved morphology registry is incompatible with the current "
            "MORPHOLOGY_GROUPS constant (it must be a prefix; append-only).\n"
            f"  saved:   {list(saved)}\n"
            f"  current: {list(MORPHOLOGY_GROUPS)}\n"
            "Reordering/relabeling groups would misroute existing adapters."
        )
    return saved


def validate_object_type_to_group_id(
    object_type_to_group_id: Dict[str, int],
    num_groups: int,
) -> Dict[str, int]:
    """Coerce/validate a loaded routing table (json keys are strings)."""
    table: Dict[str, int] = {}
    for object_type, gid in object_type_to_group_id.items():
        gid = int(gid)
        if not 0 <= gid < num_groups:
            raise ValueError(
                f"morphology routing table maps '{object_type}' to group id {gid}, "
                f"out of range [0, {num_groups})."
            )
        table[str(object_type)] = gid
    if not table:
        raise ValueError("morphology routing table is empty.")
    return table


def object_types_to_group_id_tensor(
    object_types: Sequence[str],
    object_type_to_group_id: Dict[str, int],
    device,
) -> torch.Tensor:
    """Map a batch of object_type strings to a ``(B,)`` long group-id tensor."""
    ids: List[int] = []
    for ot in object_types:
        ot = str(ot)
        gid = object_type_to_group_id.get(ot)
        if gid is None:
            raise KeyError(
                f"object_type '{ot}' is not in the morphology routing table. "
                f"Add it to species_tags.jsonl (with a registry morphology group as motion_tags[0])."
            )
        ids.append(gid)
    return torch.tensor(ids, device=device, dtype=torch.long)
