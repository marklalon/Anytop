"""Declarative clip selection shared by training and the preview command.

Rules within include/exclude are ORed; fields inside a rule are ANDed.
Exclusion always wins. No sampling weights or augmentation live here.
"""
from __future__ import annotations

from collections import Counter
from fnmatch import fnmatchcase
import hashlib
import json
from pathlib import Path
import random
import warnings

from .motion_labels import ACTION_GROUPS, HEAD_VOCAB, head_words_in, parse_action_label


FIELDS = frozenset({
    "namespaces", "namespace_prefixes", "object_types", "action_groups",
    "species_tags_any", "species_tags_all", "species_tags_none", "species_tags_exact",
    "heads_any", "heads_all", "heads_none", "clips", "clip_globs",
})


def load_sample_filter(value):
    """Read a strict JSON object, or validate an in-memory config."""
    if isinstance(value, (str, Path)):
        with Path(value).open(encoding="utf-8-sig") as handle:
            value = json.load(handle)
    if not isinstance(value, dict) or set(value) - {"include", "exclude"}:
        raise ValueError("sample_filter must be an object containing only include/exclude")
    if not isinstance(value.get("include"), list) or not value["include"]:
        raise ValueError("sample_filter.include must be a nonempty rule list; [{}] selects all")
    result = {}
    for section in ("include", "exclude"):
        rules = value.get(section, [])
        if not isinstance(rules, list):
            raise ValueError(f"sample_filter.{section} must be a rule list")
        result[section] = []
        for index, rule in enumerate(rules):
            where = f"sample_filter.{section}[{index}]"
            if not isinstance(rule, dict) or set(rule) - FIELDS:
                raise ValueError(f"{where}: unknown fields or invalid rule: {rule!r}")
            normalized = {}
            for field, values in rule.items():
                if (not isinstance(values, list) or not values or
                        any(not isinstance(v, str) or not v.strip() for v in values)):
                    raise ValueError(f"{where}.{field}: expected a nonempty list of strings")
                values = [v.strip() for v in values]
                if field.startswith("species_tags_"):
                    values = [v.casefold() for v in values]
                if field.startswith("heads_"):
                    unknown = set(values) - set(HEAD_VOCAB)
                    if unknown:
                        raise ValueError(f"{where}.{field}: unknown head words {sorted(unknown)}")
                if field == "action_groups" and set(values) - set(ACTION_GROUPS):
                    raise ValueError(f"{where}: unknown action_groups {values}")
                normalized[field] = sorted(set(values))
            result[section].append(normalized)
    return result


def _set_match(actual, wanted, mode):
    actual, wanted = set(actual), set(wanted)
    if mode == "any":
        return bool(actual & wanted)
    if mode == "all":
        return wanted <= actual
    if mode == "none":
        return not actual & wanted
    return actual == wanted


def matches_rule(record, rule):
    for field, wanted in rule.items():
        if field == "namespaces" and record["namespace"] not in wanted:
            return False
        if field == "namespace_prefixes" and not any(
            record["namespace"] == p or record["namespace"].startswith(p + "/") for p in wanted
        ):
            return False
        if field == "object_types" and record["object_type"] not in wanted:
            return False
        if field == "action_groups" and record["action_group"] not in wanted:
            return False
        if field == "clips" and record["clip_id"] not in wanted:
            return False
        if field == "clip_globs" and not any(fnmatchcase(record["clip_id"], p) for p in wanted):
            return False
        if field.startswith("species_tags_") and not _set_match(
            (tag.casefold() for tag in record["species_tags"]), wanted, field.removeprefix("species_tags_")
        ):
            return False
        if field.startswith("heads_") and not _set_match(record["heads"], wanted, field.removeprefix("heads_")):
            return False
    return True


def select_records(config, records, *, known_clips=None, known_objects=None, known_namespaces=None,
                   known_tags=None):
    """Select within an already split-bounded candidate pool, without duplicates."""
    config = load_sample_filter(config)
    universes = {
        "clips": set(known_clips if known_clips is not None else (r["clip_id"] for r in records)),
        "object_types": set(known_objects if known_objects is not None else (r["object_type"] for r in records)),
        "namespaces": set(known_namespaces if known_namespaces is not None else (r["namespace"] for r in records)),
        "species_tags": {t.casefold() for t in (known_tags if known_tags is not None else
                                                (t for r in records for t in r["species_tags"]))},
    }
    for section, rules in config.items():
        for index, rule in enumerate(rules):
            for field, values in rule.items():
                universe = universes.get("species_tags" if field.startswith("species_tags_") else field)
                if universe is not None and set(values) - universe:
                    raise ValueError(f"{section}[{index}].{field}: unknown values {sorted(set(values) - universe)}")
                if field == "namespace_prefixes":
                    for prefix in values:
                        if not any(ns == prefix or ns.startswith(prefix + "/") for ns in universes["namespaces"]):
                            raise ValueError(f"{section}[{index}]: unknown namespace prefix {prefix!r}")
    counts = {section: [0] * len(rules) for section, rules in config.items()}
    selected = {}
    for record in records:
        hits = {}
        for section, rules in config.items():
            hits[section] = [i for i, rule in enumerate(rules) if matches_rule(record, rule)]
            for i in hits[section]:
                counts[section][i] += 1
        if hits["include"] and not hits["exclude"]:
            selected[record["clip_id"]] = record
    for section, values in counts.items():
        for index, count in enumerate(values):
            if not count:
                warnings.warn(f"sample_filter.{section}[{index}] matches no clips in this split", stacklevel=2)
    return list(selected.values()), counts


def resolve_sample_filter(config, cond, sources, metadata_by_namespace, tags_by_object, *,
                          split="train", min_length=20, sample_limit=0, num_frames=60, seed=10):
    """Build the clip index, evaluate rules, then apply training length/limit rules.

    Reads selected motion headers with mmap, never writes length caches. Missing
    metadata and unreadable selected files are explicit failures.
    """
    import numpy as np

    config = load_sample_filter(config)
    if split not in ("train", "val", "all"):
        raise ValueError(f"Unknown split {split!r}")
    files_by_namespace = {
        source.namespace: sorted(p.name for p in Path(source.motion_dir).glob("*.npy"))
        for source in sources
    }
    known_clips = {f"{ns}/{name}" for ns, names in files_by_namespace.items() for name in names}
    candidates = []
    paths = {}
    for source in sources:
        if split != "all" and source.split != split:
            continue
        objects = [(key, entry["species_name"]) for key, entry in cond.items()
                   if entry["dataset_namespace"] == source.namespace]
        for name in files_by_namespace[source.namespace]:
            # Prefer the most specific species prefix (species names may contain underscores).
            owners = [(key, species) for key, species in objects if name.startswith(species + "_")]
            if not owners:
                continue
            object_type, _ = max(owners, key=lambda pair: len(pair[1]))
            clip_id = f"{source.namespace}/{name}"
            metadata = metadata_by_namespace[source.namespace].get(name)
            if metadata is None:
                raise ValueError(f"{clip_id}: missing motion metadata")
            candidates.append({
                "clip_id": clip_id, "namespace": source.namespace, "object_type": object_type,
                "species_tags": list(tags_by_object.get(object_type, ())),
                "action_group": str(metadata.get("action_group") or ""),
                "action_label": str(metadata.get("action_label") or ""),
                "heads": head_words_in(parse_action_label(str(metadata.get("action_label") or ""))),
            })
            paths[clip_id] = Path(source.motion_dir) / name
    selected, counts = select_records(
        config, candidates, known_clips=known_clips, known_objects=cond,
        known_namespaces=[source.namespace for source in sources],
        known_tags=[tag for tags in tags_by_object.values() for tag in tags],
    )
    for record in selected:
        try:
            motion = np.load(paths[record["clip_id"]], mmap_mode="r")
            record["length"] = int(motion.shape[0])
            del motion
        except Exception as exc:
            raise ValueError(f"{record['clip_id']}: cannot read motion length: {exc}") from exc
    eligible = [r for r in selected if r["length"] >= min_length]
    too_short = [r["clip_id"] for r in selected if r["length"] < min_length]
    eligible.sort(key=lambda r: (r["length"], r["clip_id"]))
    if sample_limit > 0:
        preferred = [r for r in eligible if r["length"] >= num_frames]
        pool = preferred if len(preferred) >= sample_limit else eligible
        eligible = random.Random(seed).sample(pool, sample_limit) if len(pool) > sample_limit else pool
    eligible.sort(key=lambda r: r["clip_id"])
    if not eligible:
        raise ValueError(f"sample_filter selects no trainable clips in split {split!r} "
                         f"(candidates={len(candidates)}, matched={len(selected)}, too_short={len(too_short)})")
    groups = sorted({r["action_group"] for r in eligible})
    report = {
        "config": config, "split": split, "candidate_count": len(candidates),
        "matched_count": len(selected), "rule_matches": counts, "too_short": too_short,
        "sample_limit_removed": len(selected) - len(too_short) - len(eligible),
        "clips": eligible, "effective_action_groups": groups,
        "action_group": groups[0] if len(groups) == 1 else "all",
    }
    report["distributions"] = {
        field: dict(sorted(Counter(r[field] for r in eligible).items()))
        for field in ("namespace", "object_type", "action_group")
    }
    report["distributions"]["heads"] = dict(sorted(Counter(h for r in eligible for h in r["heads"]).items()))
    return report


def selection_snapshot(report):
    """Stable identity of the rules, eligibility settings and effective clip metadata."""
    payload = {k: report[k] for k in ("config", "split", "clips")}
    payload["fingerprint"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return payload


def record_sample_selection(report, directory, *, resume_directory=None):
    """Check before writing: resumes may not silently change their training pool."""
    snapshot = selection_snapshot(report)
    if resume_directory is not None:
        previous = Path(resume_directory) / "sample_selection.json"
        if not previous.is_file():
            raise ValueError(f"Cannot verify resumed sample selection: missing {previous}; use a fresh save_dir")
        old = json.loads(previous.read_text(encoding="utf-8"))
        if old != snapshot:
            raise ValueError("Resuming would change sample_filter or effective clips; use a fresh save_dir")
    directory = Path(directory)
    (directory / "sample_selection.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (directory / "sample_filter.json").write_text(
        json.dumps(report["config"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
