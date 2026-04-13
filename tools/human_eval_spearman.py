from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loaders.truebones.offline_reference_dataset import get_motion_dir, infer_object_type, load_cond_dict, resolve_dataset_root
from eval.motion_quality_scorer import MotionQualityScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Spearman correlation between pairwise human preferences and motion scorer margins.")
    parser.add_argument("--annotations", required=True, help="JSON file containing human preference pairs.")
    parser.add_argument("--checkpoint-dir", required=True, help="Motion scorer checkpoint directory or explicit model checkpoint path.")
    parser.add_argument("--dataset-dir", default="", help="Processed dataset root. Empty uses the scorer default dataset path.")
    parser.add_argument("--device", default="cuda", help="Scoring device.")
    parser.add_argument("--tie-margin", default=0.0, type=float, help="Treat model margins with absolute value <= this threshold as ties.")
    parser.add_argument("--output-json", default="", help="Optional path to write the full evaluation report as JSON.")
    return parser.parse_args()


def _load_pairs(path: Path) -> list[dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        pairs = payload.get("pairs", [])
    elif isinstance(payload, list):
        pairs = payload
    else:
        raise ValueError("Annotations JSON must be either a list of pairs or an object with a 'pairs' field.")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("No annotation pairs were found.")
    return pairs


def _normalize_winner(value: object) -> int:
    token = str(value or "").strip().lower()
    if token in {"a", "clip_a", "left", "1", "winner_a"}:
        return 1
    if token in {"b", "clip_b", "right", "-1", "winner_b"}:
        return -1
    if token in {"tie", "equal", "0", "draw"}:
        return 0
    raise ValueError(f"Unsupported winner token: {value}")


def _resolve_motion_path(raw_value: object, motion_dir: Path) -> Path:
    candidate = Path(str(raw_value))
    if candidate.is_file():
        return candidate.resolve()
    candidate_under_motion_dir = (motion_dir / str(raw_value)).resolve()
    if candidate_under_motion_dir.is_file():
        return candidate_under_motion_dir
    raise FileNotFoundError(f"Motion file was not found: {raw_value}")


def _resolve_object_type(entry: dict[str, object], key: str, motion_path: Path, object_types: list[str]) -> str:
    explicit = entry.get(key)
    if explicit:
        return str(explicit)
    return infer_object_type(motion_path.name, object_types)


def _derive_human_preference(entry: dict[str, object]) -> int:
    if "agreed_winner" in entry:
        return _normalize_winner(entry["agreed_winner"])
    if "winner" in entry:
        return _normalize_winner(entry["winner"])
    if "annotator_1" in entry and "annotator_2" in entry:
        first = _normalize_winner(entry["annotator_1"])
        second = _normalize_winner(entry["annotator_2"])
        if first == second:
            return first
        raise ValueError(f"Pair {entry.get('pair_id', '<unknown>')} has no agreed_winner and annotators disagree.")
    raise ValueError(f"Pair {entry.get('pair_id', '<unknown>')} does not define a usable winner field.")


def main() -> int:
    args = parse_args()
    annotations_path = Path(args.annotations).resolve()
    dataset_root = resolve_dataset_root(args.dataset_dir or None)
    motion_dir = get_motion_dir(dataset_root)
    cond_dict = load_cond_dict(dataset_root)
    object_types = sorted(cond_dict.keys(), key=len, reverse=True)
    scorer = MotionQualityScorer(args.checkpoint_dir, device=args.device, dataset_dir=dataset_root)

    pairs = _load_pairs(annotations_path)
    pair_reports: list[dict[str, object]] = []
    human_preferences: list[float] = []
    model_margins: list[float] = []
    model_preferences: list[int] = []

    for index, entry in enumerate(pairs, start=1):
        pair_id = str(entry.get("pair_id") or f"pair_{index:03d}")
        clip_a_path = _resolve_motion_path(entry["clip_a"], motion_dir)
        clip_b_path = _resolve_motion_path(entry["clip_b"], motion_dir)
        object_type_a = _resolve_object_type(entry, "object_type_a", clip_a_path, object_types)
        object_type_b = _resolve_object_type(entry, "object_type_b", clip_b_path, object_types)
        if entry.get("object_type"):
            shared_object_type = str(entry["object_type"])
            object_type_a = shared_object_type
            object_type_b = shared_object_type

        score_a = float(scorer.score_npy(clip_a_path, object_type_a)["quality_score"].item())
        score_b = float(scorer.score_npy(clip_b_path, object_type_b)["quality_score"].item())
        human_preference = _derive_human_preference(entry)
        margin = score_a - score_b
        if abs(margin) <= float(args.tie_margin):
            model_preference = 0
        else:
            model_preference = 1 if margin > 0.0 else -1

        human_preferences.append(float(human_preference))
        model_margins.append(float(margin))
        model_preferences.append(int(model_preference))
        pair_reports.append(
            {
                "pair_id": pair_id,
                "clip_a": str(clip_a_path),
                "clip_b": str(clip_b_path),
                "object_type_a": object_type_a,
                "object_type_b": object_type_b,
                "quality_score_a": score_a,
                "quality_score_b": score_b,
                "model_margin": margin,
                "human_preference": human_preference,
                "model_preference": model_preference,
                "match": bool(model_preference == human_preference),
            }
        )

    non_tie_pairs = [report for report in pair_reports if report["human_preference"] != 0]
    non_tie_matches = [report for report in non_tie_pairs if report["match"]]
    spearman = spearmanr(human_preferences, model_margins)
    summary = {
        "annotations": str(annotations_path),
        "checkpoint_dir": str(args.checkpoint_dir),
        "dataset_root": str(dataset_root),
        "num_pairs": len(pair_reports),
        "num_non_tie_pairs": len(non_tie_pairs),
        "pair_accuracy_non_tie": float(len(non_tie_matches) / max(len(non_tie_pairs), 1)),
        "spearman_rho": None if spearman.statistic is None else float(spearman.statistic),
        "spearman_pvalue": None if spearman.pvalue is None else float(spearman.pvalue),
        "tie_margin": float(args.tie_margin),
    }
    report = {
        "summary": summary,
        "pairs": pair_reports,
    }

    print(json.dumps(summary, indent=2))
    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())