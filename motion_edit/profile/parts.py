"""Per-joint body parts and ground contacts of a skeleton, as motion_edit uses them.

The default is the prefill (``joint_parts.prefill_joint_parts``: names, the tree,
geometry). A person corrects it per species in the dataset's
``joint_parts_overrides.json`` and per package in its manifest; each layer holds
only the joints it changes, ``{joint name: {"part": ..., "contact": 0 / 1}}``
(either key may be left out). Later layers win.

The parts drive the editing groups (``decompose.chain_groups``), the secondary
motion channels (``tail`` joints, ``soft`` joints are passive), the IK limbs
and the strike chain prior; the contact flags are the contact set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from data_loaders.truebones.truebones_utils.joint_parts import (
    ALL_PART_LABELS,
    HELPER_PART,
    prefill_joint_parts,
)

JOINT_PARTS_OVERRIDES_FILE = "joint_parts_overrides.json"
# Provenance of a resolved joint beyond the prefill pass that proposed it.
LAYER_SOURCES = ("species", "package")
# Parts that make up one kind of limb: an IK limb or a spread limb stops where its kind does.
LIMB_KIND = {"arm": "arm", "hand": "arm", "leg": "leg", "foot": "leg", "wing": "wing"}


def validate_entries(entries: Optional[dict], where: str) -> dict:
    """``{name: {"part"?, "contact"?}}`` checked and normalized; a helper with a contact
    is refused (:func:`check_helper_contacts` covers a helper part set by another layer)."""
    out = {}
    for name, entry in (entries or {}).items():
        entry = dict(entry or {})
        clean = {}
        if "part" in entry:
            if entry["part"] not in ALL_PART_LABELS:
                raise ValueError(f"{where}: {name}: unknown part {entry['part']!r}")
            clean["part"] = str(entry["part"])
        if "contact" in entry:
            if entry["contact"] not in (0, 1, True, False):
                raise ValueError(f"{where}: {name}: contact must be 0 or 1")
            clean["contact"] = int(bool(entry["contact"]))
        if clean.get("part") == HELPER_PART and clean.get("contact"):
            raise ValueError(f"{where}: {name}: a helper joint cannot be a contact")
        if clean:
            out[str(name)] = clean
    return out


@dataclass
class ResolvedParts:
    """One skeleton's parts and contacts after every layer, in joint order."""

    names: list[str]
    parts: list[str]
    contact: list[bool]
    src: list[str]            # the prefill pass, or the layer that last changed the joint
    why: list[str]
    prefill: dict             # {name: {part, contact, src, why}}
    unknown_names: list[str] = field(default_factory=list)

    @property
    def contacts(self) -> list[int]:
        return [j for j, c in enumerate(self.contact) if c]

    def joints_of(self, *parts: str) -> list[int]:
        return [j for j, p in enumerate(self.parts) if p in parts]

    def rows(self) -> list[dict]:
        return [{"part": p, "contact": int(c), "src": s, "why": w}
                for p, c, s, w in zip(self.parts, self.contact, self.src, self.why)]


def resolve_parts(cond_entry: dict, *layers: Optional[dict], prefill: Optional[dict] = None) -> ResolvedParts:
    """The prefill of ``cond_entry`` (or the given one) with each override layer applied
    in turn (``layers`` in ``LAYER_SOURCES`` order)."""
    names = [str(n) for n in cond_entry["joints_names"]]
    prefill = prefill if prefill is not None else prefill_joint_parts(cond_entry)
    parts = [prefill[n]["part"] for n in names]
    contact = [bool(prefill[n]["contact"]) for n in names]
    src = [prefill[n]["src"] for n in names]
    why = [prefill[n]["why"] for n in names]
    index_of = {n: j for j, n in enumerate(names)}
    unknown = []
    for source, layer in zip(LAYER_SOURCES, layers):
        for name, entry in validate_entries(layer, source).items():
            j = index_of.get(name)
            if j is None:
                unknown.append(name)
                continue
            parts[j] = entry.get("part", parts[j])
            contact[j] = bool(entry.get("contact", contact[j]))
            src[j] = source
    contact = [c and p != HELPER_PART for p, c in zip(parts, contact)]
    return ResolvedParts(names, parts, contact, src, why, prefill, unknown)


def check_helper_contacts(wanted: dict, where: str) -> None:
    """Refuse a ``{name: {part, contact}}`` state that makes a helper joint a contact."""
    bad = sorted(name for name, entry in wanted.items()
                 if entry.get("part") == HELPER_PART and entry.get("contact"))
    if bad:
        raise ValueError(f"{where}: a helper joint cannot be a contact: {', '.join(bad)}")


def layer_for(base: ResolvedParts, wanted: dict) -> dict:
    """The override entries that turn ``base`` into ``wanted`` (``{name: {part, contact}}``,
    joints left out keep their ``base`` values)."""
    out = {}
    for j, name in enumerate(base.names):
        want = wanted.get(name)
        if want is None:
            continue
        entry = {}
        if "part" in want and want["part"] != base.parts[j]:
            entry["part"] = want["part"]
        if "contact" in want and bool(want["contact"]) != base.contact[j]:
            entry["contact"] = int(bool(want["contact"]))
        if entry:
            out[name] = entry
    return validate_entries(out, "edit")


def merge_layers(lower: Optional[dict], upper: Optional[dict]) -> dict:
    """``upper`` over ``lower``, key by key."""
    out = {name: dict(entry) for name, entry in (lower or {}).items()}
    for name, entry in (upper or {}).items():
        out.setdefault(name, {}).update(entry)
    return out
