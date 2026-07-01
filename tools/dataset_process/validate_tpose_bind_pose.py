"""
validate_tpose_bind_pose.py

Export one bind/rest-pose GLB per Truebones species and validate that every
file in that species folder uses the same bind pose.

The exported pose is the armature bind/rest pose (``bone.matrix_local``), not
the first animation frame.  Source-file priority for the exported bind carrier:

    1. filename stem ending with TPOSE / TPOS
    2. filename stem equal to the species name
    3. first action file

Outputs are written to a sibling ``Truebone_Z-OO_TPOSE`` directory by default,
one flat ``<Species>_TPOSE.glb`` file per species.  Export is incremental by
default: existing TPOSE GLBs are not re-exported unless ``--overwrite`` is set.
Bind-pose, selected bind-pose symmetry, source-FPS, and first-frame consistency
checks always run for every selected species, regardless of whether its TPOSE
GLB already exists.

Use ``--quick`` to skip export, bind-pose consistency, and FPS checks.  Quick
mode only checks selected bind-pose symmetry plus first-frame-vs-bind-pose for
TPOSE/species-name files.

Requires bpy (Blender as a Python module) -- run with the project's .venv::

    .venv/Scripts/python.exe Anytop/tools/dataset_process/validate_tpose_bind_pose.py
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Put the Anytop package root on sys.path so data_loaders / motion_lib / utils
# imports resolve to the Anytop copies (NOT the top-level pcvg utils).
_ANYTOP_DIR = Path(__file__).resolve().parent.parent.parent
if str(_ANYTOP_DIR) not in sys.path:
    sys.path.insert(0, str(_ANYTOP_DIR))


TOLERANCE = 1e-2
TARGET_FPS = 30.0
FPS_TOLERANCE = 0.1
SYMMETRY_IGNORE_TERMINAL_LAYERS = 2
SYMMETRY_RELATIVE_TOLERANCE = 0.3
SUPPORTED_EXTENSIONS = {".fbx", ".glb", ".gltf"}

MatrixRows = tuple[tuple[float, float, float, float], ...]
RestSignature = tuple[tuple[str, str | None, MatrixRows], ...]
Vector3 = tuple[float, float, float]
RestSkeleton = tuple[tuple[str, str | None, Vector3], ...]
SymmetryFailure = tuple[str, str, float, float]


def _compact_name(value: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(value or "").lower())


def _is_tpose_file(path: str) -> bool:
    compact = _compact_name(Path(path).stem)
    return compact.endswith("tpose") or compact.endswith("tpos")


def _is_species_name_file(path: str, object_type: str) -> bool:
    return _compact_name(Path(path).stem) == _compact_name(object_type)


def _is_pose_reference_file(path: str, object_type: str) -> bool:
    return _is_tpose_file(path) or _is_species_name_file(path, object_type)


def _source_sort_key(path: str) -> tuple[str, int, str]:
    priority = {".fbx": 0, ".glb": 1, ".gltf": 2}
    p = Path(path)
    return p.stem.lower(), priority.get(p.suffix.lower(), 99), p.name.lower()


def _list_source_files(directory: str) -> list[str]:
    """Return sorted FBX/GLB/GLTF files in *directory*."""
    files: list[str] = []
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            continue
        files.append(path)

    return sorted(files, key=_source_sort_key)


def _get_object_dirs(dataset_dir: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for name in sorted(os.listdir(dataset_dir)):
        dir_path = os.path.join(dataset_dir, name)
        if os.path.isdir(dir_path) and not name.startswith("."):
            entries.append((name, dir_path))
    return entries


def _select_bind_pose_source(object_type: str, files: list[str]) -> tuple[str, str]:
    """Return ``(path, reason)`` using TPOSE -> species-name -> first-file priority."""
    sorted_files = sorted(files, key=_source_sort_key)
    for path in sorted_files:
        if _is_tpose_file(path):
            return path, "tpose"
    for path in sorted_files:
        if _is_species_name_file(path, object_type):
            return path, "species-name"
    return sorted_files[0], "first-file"


def _confirm_yes_no(prompt: str) -> bool:
    while True:
        response = input(prompt).strip().lower()
        if response in ("yes", "y"):
            return True
        if response in ("no", "n"):
            return False
        print("Invalid response. Please enter 'yes', 'y', 'no', or 'n'.")


def _warning_sort_key(message: str) -> tuple[str, str]:
    species = str(message).split(":", 1)[0].split("/", 1)[0].strip().lower()
    return species, str(message).lower()


# ---------------------------------------------------------------------------
# Blender helpers, used inside workers
# ---------------------------------------------------------------------------


def _matrix_to_rows(matrix) -> MatrixRows:
    return tuple(
        tuple(float(matrix[row][col]) for col in range(4))
        for row in range(4)
    )


def _matrices_equal(left: MatrixRows, right: MatrixRows, tol: float) -> bool:
    for row in range(4):
        for col in range(4):
            if abs(left[row][col] - right[row][col]) >= tol:
                return False
    return True


def _find_armature(bpy, path: str):
    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError(f"No armature found in {path}")
    return max(armatures, key=lambda obj: len(obj.data.bones))


def _import_scene_file(filepath: str, *, for_export: bool = False) -> None:
    from motion_lib.FBX import (
        _silence_os_std,
        clear_scene,
        import_fbx,
        import_gltf,
        remove_lights_and_cameras,
    )

    clear_scene()
    suffix = Path(filepath).suffix.lower()
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()), \
         _silence_os_std():
        if suffix == ".fbx":
            import_fbx(filepath, use_image_search=for_export)
        elif suffix in {".glb", ".gltf"}:
            import_gltf(filepath)
        else:
            raise ValueError(f"Unsupported source format: {suffix}")

    remove_lights_and_cameras()


def _resolve_textures_for_export(bpy, armature, source_path: str) -> None:
    if Path(source_path).suffix.lower() != ".fbx":
        return
    try:
        from utils.texture_resolve import resolve_main_character_textures

        with contextlib.redirect_stdout(io.StringIO()):
            resolve_main_character_textures(bpy, armature, source_path)
    except Exception:
        # Texture repair is best-effort; bind-pose validation/export should not
        # fail just because an optional diffuse lookup cannot be repaired.
        pass


def _extract_rest_signature(armature) -> RestSignature:
    rows: list[tuple[str, str | None, MatrixRows]] = []
    for bone in armature.data.bones:
        parent_name = bone.parent.name if bone.parent is not None else None
        rows.append((bone.name, parent_name, _matrix_to_rows(bone.matrix_local)))
    return tuple(sorted(rows, key=lambda item: item[0].lower()))


def _extract_rest_skeleton(armature) -> RestSkeleton:
    rows: list[tuple[str, str | None, Vector3]] = []
    for bone in armature.data.bones:
        parent_name = bone.parent.name if bone.parent is not None else None
        if hasattr(bone, "head_local"):
            position = tuple(float(bone.head_local[index]) for index in range(3))
        else:
            matrix = bone.matrix_local
            position = tuple(float(matrix[index][3]) for index in range(3))
        rows.append((bone.name, parent_name, position))
    return tuple(rows)


def _rest_skeleton_arrays(
    rest_skeleton: RestSkeleton,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    names = [name for name, _parent, _position in rest_skeleton]
    name_to_index = {name: index for index, name in enumerate(names)}
    parents = np.full(len(rest_skeleton), -1, dtype=np.int64)
    positions = np.zeros((len(rest_skeleton), 3), dtype=np.float64)

    for index, (_name, parent_name, position) in enumerate(rest_skeleton):
        if parent_name is not None and parent_name in name_to_index:
            parents[index] = name_to_index[parent_name]
        positions[index] = np.asarray(position, dtype=np.float64)

    return names, parents, positions


def _terminal_distances_to_leaf(parents: np.ndarray) -> list[int]:
    children: list[list[int]] = [[] for _ in range(len(parents))]
    for index, parent_index in enumerate(parents):
        if parent_index >= 0:
            children[int(parent_index)].append(index)

    memo: dict[int, int] = {}

    def visit(index: int) -> int:
        if index in memo:
            return memo[index]
        if not children[index]:
            memo[index] = 0
        else:
            memo[index] = 1 + max(visit(child_index) for child_index in children[index])
        return memo[index]

    return [visit(index) for index in range(len(parents))]


def _symmetry_pair_error(
    left_index: int,
    right_index: int,
    parents: np.ndarray,
    positions: np.ndarray,
    partner_indices: list[int],
    mirror_axis: int,
    mirror_value: float,
) -> tuple[float, float, float]:
    left_parent = int(parents[left_index])
    right_parent = int(parents[right_index])
    use_local = (
        left_parent >= 0
        and right_parent >= 0
        and (
            left_parent == right_parent
            or (
                left_parent < len(partner_indices)
                and int(partner_indices[left_parent]) == right_parent
            )
        )
    )

    if use_local:
        left_delta = positions[left_index] - positions[left_parent]
        right_delta = positions[right_index] - positions[right_parent]
        plane_axes = [axis for axis in range(3) if axis != mirror_axis]
        mirror_error = abs(float(left_delta[mirror_axis] + right_delta[mirror_axis]))
        yz_error = float(np.linalg.norm(left_delta[plane_axes] - right_delta[plane_axes]))
        scale = max(float(np.linalg.norm(left_delta)), float(np.linalg.norm(right_delta)))
    else:
        mirrored_left = positions[left_index].copy()
        mirrored_left[mirror_axis] = (
            2.0 * mirror_value - mirrored_left[mirror_axis]
        )
        diff = mirrored_left - positions[right_index]
        plane_axes = [axis for axis in range(3) if axis != mirror_axis]
        mirror_error = abs(float(diff[mirror_axis]))
        yz_error = float(np.linalg.norm(diff[plane_axes]))
        scale = max(
            float(np.linalg.norm(positions[left_index] - positions[right_index])),
            1e-6,
        )

    return mirror_error, yz_error, max(scale, 1e-6)


def _infer_best_mirror_axis(
    unique_pairs: list[tuple[int, int]],
    parents: np.ndarray,
    positions: np.ndarray,
    partner_indices: list[int],
) -> tuple[int, float]:
    best_axis = 0
    best_value = 0.0
    best_score: tuple[float, float, int] | None = None

    for axis in range(3):
        pair_centers = [
            (float(positions[left_index, axis]) + float(positions[right_index, axis])) * 0.5
            for left_index, right_index in unique_pairs
        ]
        mirror_value = float(np.median(np.asarray(pair_centers, dtype=np.float64)))
        normalized_errors = []
        absolute_errors = []
        for left_index, right_index in unique_pairs:
            mirror_error, yz_error, scale = _symmetry_pair_error(
                left_index,
                right_index,
                parents,
                positions,
                partner_indices,
                axis,
                mirror_value,
            )
            error = max(mirror_error, yz_error)
            normalized_errors.append(error / max(scale, 1e-6))
            absolute_errors.append(error)

        score = (
            float(np.max(normalized_errors)) if normalized_errors else 0.0,
            float(np.mean(absolute_errors)) if absolute_errors else 0.0,
            axis,
        )
        if best_score is None or score < best_score:
            best_score = score
            best_axis = axis
            best_value = mirror_value

    return best_axis, best_value


def _analyze_rest_skeleton_symmetry(
    rest_skeleton: RestSkeleton,
    ignore_terminal_layers: int = SYMMETRY_IGNORE_TERMINAL_LAYERS,
    relative_tolerance: float = SYMMETRY_RELATIVE_TOLERANCE,
) -> tuple[int, list[SymmetryFailure]]:
    if not rest_skeleton:
        return 0, []

    from data_loaders.truebones.truebones_utils.physics_joint_annotation import (
        _infer_symmetry_metadata,
    )

    names, parents, positions = _rest_skeleton_arrays(rest_skeleton)
    _labels, partner_indices, pairs = _infer_symmetry_metadata(
        names,
        parents,
        positions,
    )
    if not pairs:
        return 0, []

    terminal_layers = max(int(ignore_terminal_layers), 0)
    terminal_distances = _terminal_distances_to_leaf(parents)
    included_indices = {
        index
        for index, distance in enumerate(terminal_distances)
        if distance >= terminal_layers
    }

    unique_pairs = sorted(
        {
            tuple(sorted((int(left_index), int(right_index))))
            for left_index, right_index in pairs
            if int(left_index) in included_indices and int(right_index) in included_indices
        }
    )
    if not unique_pairs:
        return 0, []

    mirror_axis, mirror_value = _infer_best_mirror_axis(
        unique_pairs,
        parents,
        positions,
        partner_indices,
    )

    failures: list[SymmetryFailure] = []
    rel_tol = max(float(relative_tolerance), 0.0)
    for left_index, right_index in unique_pairs:
        mirror_error, yz_error, scale = _symmetry_pair_error(
            left_index,
            right_index,
            parents,
            positions,
            partner_indices,
            mirror_axis,
            mirror_value,
        )
        threshold = max(rel_tol * scale, TOLERANCE)
        error = max(mirror_error, yz_error)
        if error > threshold:
            failures.append((names[left_index], names[right_index], error, threshold))

    failures.sort(key=lambda item: item[2] - item[3], reverse=True)
    return len(unique_pairs), failures


def _format_symmetry_warning(
    object_type: str,
    source_path: str,
    checked_pairs: int,
    failures: list[SymmetryFailure],
    ignore_terminal_layers: int,
    relative_tolerance: float,
) -> str | None:
    if checked_pairs < 2 or not failures:
        return None

    worst_left, worst_right, worst_error, worst_threshold = failures[0]
    return (
        f"{object_type}: bind pose is not symmetric in "
        f"{os.path.basename(source_path)} "
        f"(worst={worst_left}<->{worst_right}, "
        f"error={worst_error:.6g}, threshold={worst_threshold:.6g})"
    )


def _compare_rest_signatures(
    reference: RestSignature,
    candidate: RestSignature,
    tol: float,
) -> tuple[bool, str]:
    ref_by_name = {name: (parent, matrix) for name, parent, matrix in reference}
    cand_by_name = {name: (parent, matrix) for name, parent, matrix in candidate}

    missing = sorted(set(ref_by_name) - set(cand_by_name))
    extra = sorted(set(cand_by_name) - set(ref_by_name))
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing {len(missing)} bone(s)")
        if extra:
            parts.append(f"extra {len(extra)} bone(s)")
        return False, ", ".join(parts)

    parent_mismatches = 0
    matrix_mismatches = 0
    for name in sorted(ref_by_name):
        ref_parent, ref_matrix = ref_by_name[name]
        cand_parent, cand_matrix = cand_by_name[name]
        if ref_parent != cand_parent:
            parent_mismatches += 1
            continue
        if not _matrices_equal(ref_matrix, cand_matrix, tol):
            matrix_mismatches += 1

    if parent_mismatches or matrix_mismatches:
        return False, (
            f"{parent_mismatches} parent mismatch(es), "
            f"{matrix_mismatches} matrix mismatch(es)"
        )

    return True, ""


def _matrix_max_abs_error(left, right) -> float:
    return max(
        abs(left[row][col] - right[row][col])
        for row in range(4)
        for col in range(4)
    )


def _first_frame_bind_mismatch_stats(bpy, armature, tol: float) -> tuple[int, float]:
    scene = bpy.context.scene
    action = armature.animation_data.action if armature.animation_data else None
    if action and hasattr(action, "frame_start"):
        first_frame = int(action.frame_start)
    else:
        first_frame = int(scene.frame_start)

    scene.frame_set(first_frame)
    bpy.context.view_layer.update()

    mismatch_count = 0
    max_abs_error = 0.0
    for pose_bone in armature.pose.bones:
        data_bone = armature.data.bones.get(pose_bone.name)
        if data_bone is None:
            continue
        error = _matrix_max_abs_error(pose_bone.matrix, data_bone.matrix_local)
        max_abs_error = max(max_abs_error, error)
        if error >= tol:
            mismatch_count += 1
    return mismatch_count, max_abs_error


def _infer_source_frame_time(bpy, armature) -> float:
    from motion_lib.FBX import get_action_sample_times, infer_sample_fps

    fps = float(infer_sample_fps(bpy.context.scene, get_action_sample_times(armature)))
    return 1.0 / fps if fps > 0 else (1.0 / TARGET_FPS)


def _clear_pose_and_animation_for_rest_export(bpy, armature) -> None:
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)

    for obj in bpy.data.objects:
        obj.animation_data_clear()
        if hasattr(obj.data, "animation_data_clear"):
            obj.data.animation_data_clear()

    bpy.ops.object.mode_set(mode="POSE")
    for pose_bone in armature.pose.bones:
        pose_bone.location = (0.0, 0.0, 0.0)
        pose_bone.rotation_mode = "QUATERNION"
        pose_bone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        pose_bone.scale = (1.0, 1.0, 1.0)
    bpy.ops.object.mode_set(mode="OBJECT")

    bpy.context.scene.frame_set(0)
    bpy.context.view_layer.update()


def _export_rest_glb(bpy, output_path: str) -> None:
    from motion_lib.FBX import _silence_os_std

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()), \
         _silence_os_std():
        bpy.ops.export_scene.gltf(
            filepath=output_path,
            export_format="GLB",
            export_animations=False,
            export_apply=False,
        )


# ---------------------------------------------------------------------------
# Per-worker entry point  (must be module-level for pickle / ProcessPool)
# ---------------------------------------------------------------------------


def _process_one_species(
    object_type: str,
    dir_path: str,
    output_path: str,
    overwrite: bool,
    quick: bool,
    tolerance: float,
    symmetry_ignore_terminal_layers: int,
    symmetry_tolerance: float,
) -> dict[str, object]:
    import bpy
    from motion_lib.FBX import clear_scene

    result: dict[str, object] = {
        "object_type": object_type,
        "files": 0,
        "selected": "",
        "selected_reason": "",
        "output_path": output_path,
        "exported": False,
        "skipped_export": False,
        "quick": quick,
        "bind_warnings": [],
        "symmetry_warnings": [],
        "fps_warnings": [],
        "first_frame_warnings": [],
        "errors": [],
    }

    files = _list_source_files(dir_path)
    result["files"] = len(files)
    if not files:
        result["errors"] = [f"{object_type}: no FBX/GLB/GLTF files found"]
        return result

    selected_path, selected_reason = _select_bind_pose_source(object_type, files)
    result["selected"] = selected_path
    result["selected_reason"] = selected_reason

    selected_seen = False
    reference_signature: RestSignature | None = None
    signatures: dict[str, RestSignature] = {}
    fps_checked_files = 0
    fps_mismatch_values: list[tuple[float, float]] = []
    first_frame_check_files = {
        path for path in files
        if _is_pose_reference_file(path, object_type)
    }
    process_files = (
        sorted({selected_path, *first_frame_check_files}, key=_source_sort_key)
        if quick
        else files
    )

    try:
        for path in process_files:
            for_export = (
                not quick
                and path == selected_path
                and (overwrite or not os.path.exists(output_path))
            )
            _import_scene_file(path, for_export=for_export)
            armature = _find_armature(bpy, path)

            if not quick:
                signature = _extract_rest_signature(armature)
                signatures[path] = signature
                if path == selected_path:
                    reference_signature = signature

            if path == selected_path:
                selected_seen = True
                rest_skeleton = _extract_rest_skeleton(armature)
                checked_pairs, symmetry_failures = _analyze_rest_skeleton_symmetry(
                    rest_skeleton,
                    ignore_terminal_layers=symmetry_ignore_terminal_layers,
                    relative_tolerance=symmetry_tolerance,
                )
                symmetry_warning = _format_symmetry_warning(
                    object_type,
                    path,
                    checked_pairs,
                    symmetry_failures,
                    symmetry_ignore_terminal_layers,
                    symmetry_tolerance,
                )
                if symmetry_warning:
                    result["symmetry_warnings"].append(symmetry_warning)

            if not quick and not _is_pose_reference_file(path, object_type):
                fps_checked_files += 1
                frame_time = _infer_source_frame_time(bpy, armature)
                source_fps = 1.0 / frame_time if frame_time > 0 else 0.0
                if source_fps and abs(source_fps - TARGET_FPS) > FPS_TOLERANCE:
                    fps_mismatch_values.append((source_fps, frame_time))

            if path in first_frame_check_files:
                mismatch_count, max_abs_error = _first_frame_bind_mismatch_stats(
                    bpy, armature, tolerance
                )
                if mismatch_count:
                    rel = f"{object_type}/{os.path.basename(path)}"
                    result["first_frame_warnings"].append(
                        f"{rel}: first animation frame differs from bind pose "
                        f"({mismatch_count} bone(s), max_abs_error={max_abs_error:.6g})"
                    )

            if not quick and path == selected_path:
                if overwrite or not os.path.exists(output_path):
                    _resolve_textures_for_export(bpy, armature, path)
                    _clear_pose_and_animation_for_rest_export(bpy, armature)
                    _export_rest_glb(bpy, output_path)
                    result["exported"] = True
                else:
                    result["skipped_export"] = True

            clear_scene()

        if not selected_seen:
            result["errors"] = [f"{object_type}: failed to read selected bind pose"]
            return result

        if not quick:
            if reference_signature is None:
                result["errors"] = [f"{object_type}: failed to read selected bind pose"]
                return result

            bind_mismatch_details: list[str] = []
            for path, signature in signatures.items():
                if path == selected_path:
                    continue
                ok, detail = _compare_rest_signatures(reference_signature, signature, tolerance)
                if not ok:
                    bind_mismatch_details.append(f"{os.path.basename(path)} ({detail})")

            if bind_mismatch_details:
                result["bind_warnings"].append(
                    f"{object_type}: bind pose differs across "
                    f"{len(bind_mismatch_details)} / {max(len(signatures) - 1, 0)} "
                    f"checked file(s) compared with {os.path.basename(selected_path)}"
                )

        if not quick and fps_mismatch_values:
            fps_counts: dict[str, int] = {}
            fps_frame_times: dict[str, float] = {}
            for fps, frame_time in fps_mismatch_values:
                key = f"{fps:.2f}"
                fps_counts[key] = fps_counts.get(key, 0) + 1
                fps_frame_times[key] = frame_time
            values = ", ".join(
                f"{fps} FPS (frame_time={fps_frame_times[fps]:.6f}s, {count} file(s))"
                for fps, count in sorted(fps_counts.items(), key=lambda item: float(item[0]))
            )
            result["fps_warnings"].append(
                f"{object_type}: {len(fps_mismatch_values)} / {fps_checked_files} "
                f"motion file(s) have FPS != {TARGET_FPS:.6g}; values: {values}"
            )

    except Exception as exc:
        result["errors"] = [f"{object_type}: {exc}"]
        try:
            clear_scene()
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_tpose_bind_pose(
    dataset_dir: str | None = None,
    output_dir: str | None = None,
    only_objects: set[str] | None = None,
    overwrite: bool = False,
    quick: bool = False,
    yes: bool = False,
    workers: int = 16,
    tolerance: float = TOLERANCE,
    symmetry_ignore_terminal_layers: int = SYMMETRY_IGNORE_TERMINAL_LAYERS,
    symmetry_tolerance: float = SYMMETRY_RELATIVE_TOLERANCE,
) -> int:
    """Incrementally export bind-pose GLBs and always validate bind-pose consistency."""
    from data_loaders.truebones.truebones_utils.param_utils import get_raw_data_dir

    dataset_dir_resolved = Path(get_raw_data_dir(dataset_dir)).resolve()
    output_dir_resolved = (
        Path(output_dir).resolve()
        if output_dir
        else dataset_dir_resolved.parent / f"{dataset_dir_resolved.name}_TPOSE"
    )

    object_dirs = _get_object_dirs(str(dataset_dir_resolved))
    if not object_dirs:
        print(f"[WARN] no character subdirectories found in {dataset_dir_resolved}")
        return 0

    if only_objects is not None:
        filtered = [(n, p) for n, p in object_dirs if n in only_objects]
        missing = sorted(only_objects - {n for n, _ in filtered})
        if missing:
            print(f"[WARN] --filter names not found in dataset: {', '.join(missing)}")
        object_dirs = filtered
        if not object_dirs:
            print("[ERROR] no matching object types found; nothing to validate.")
            return 1

    jobs = [
        (
            object_type,
            dir_path,
            str(output_dir_resolved / f"{object_type}_TPOSE.glb"),
            overwrite,
            quick,
            tolerance,
            symmetry_ignore_terminal_layers,
            symmetry_tolerance,
        )
        for object_type, dir_path in object_dirs
    ]

    existing_outputs = sum(
        1
        for _o, _d, out, _ow, _quick, _t, _stl, _st in jobs
        if os.path.exists(out)
    )
    pending_exports = 0 if quick else (len(jobs) if overwrite else len(jobs) - existing_outputs)

    print(f"Dataset : {dataset_dir_resolved}")
    print(f"Output  : {output_dir_resolved}")
    print(f"Species : {len(jobs)}  (--overwrite={overwrite}, --quick={quick}, --workers={workers})")
    if quick:
        print("Export  : disabled by --quick")
        print("Validate: quick mode checks selected bind-pose symmetry and TPOSE/species first frame only")
    else:
        print(f"Export  : {pending_exports} pending, {existing_outputs if not overwrite else 0} existing")
        print("Validate: always checks all selected species/files")
    print(
        "Symmetry: selected bind pose, "
        f"ignore_terminal_layers={symmetry_ignore_terminal_layers}, "
        f"rel_tol={symmetry_tolerance:.3g}"
    )

    if overwrite and existing_outputs and not quick and not yes:
        print(f"\nThis will overwrite {existing_outputs} existing TPOSE GLB file(s).")
        if not _confirm_yes_no("Proceed? [y/N] "):
            print("[ABORT]")
            return 0

    n_workers = min(max(int(workers), 1), len(jobs))
    print(f"\nStarting validation/export with {n_workers} worker(s) ...\n")

    exported = 0
    skipped = 0
    failed = 0
    bind_warnings: list[str] = []
    symmetry_warnings: list[str] = []
    fps_warnings: list[str] = []
    first_frame_warnings: list[str] = []

    total = len(jobs)
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        fut_to_job = {
            executor.submit(_process_one_species, *job): job
            for job in jobs
        }
        for fut in as_completed(fut_to_job):
            (
                object_type,
                _dir_path,
                output_path,
                _overwrite,
                _quick,
                _tol,
                _sym_terminal_layers,
                _sym_tol,
            ) = fut_to_job[fut]
            done += 1
            try:
                result = fut.result()
            except Exception as exc:
                failed += 1
                print(f"[{done}/{total}] [FAIL]   {object_type}: {exc}")
                continue

            errors = list(result.get("errors") or [])
            if errors:
                failed += 1
                print(f"[{done}/{total}] [FAIL]   {object_type}: {errors[0]}")
                continue

            bind_warnings.extend(str(msg) for msg in result.get("bind_warnings") or [])
            symmetry_warnings.extend(
                str(msg) for msg in result.get("symmetry_warnings") or []
            )
            fps_warnings.extend(str(msg) for msg in result.get("fps_warnings") or [])
            first_frame_warnings.extend(
                str(msg) for msg in result.get("first_frame_warnings") or []
            )

            if result.get("exported"):
                exported += 1
                status = "exported"
            elif result.get("skipped_export"):
                skipped += 1
                status = "export-skipped"
            elif result.get("quick"):
                status = "quick-checked"
            else:
                status = "checked"

            selected = os.path.basename(str(result.get("selected") or ""))
            reason = str(result.get("selected_reason") or "")
            print(
                f"[{done}/{total}] [OK]     {object_type}: {status} "
                f"{os.path.basename(output_path)}  "
                f"(source={selected}, reason={reason}, files={result.get('files')})"
            )

    print(f"\n{'=' * 70}")
    print(f"  Exported: {exported}, Export-skipped: {skipped}, Failed: {failed}, Total: {total}")

    if quick:
        print("\n  Bind-pose consistency warnings: skipped (--quick)")
    elif bind_warnings:
        print(f"\n  Bind-pose consistency warnings ({len(bind_warnings)}):")
        for msg in sorted(bind_warnings, key=_warning_sort_key):
            print(f"    [WARN] {msg}")
    else:
        print("\n  Bind-pose consistency warnings: none")

    if symmetry_warnings:
        print(f"\n  Bind-pose symmetry warnings ({len(symmetry_warnings)}):")
        for msg in sorted(symmetry_warnings, key=_warning_sort_key):
            print(f"    [WARN] {msg}")
    else:
        print("\n  Bind-pose symmetry warnings: none")

    if quick:
        print("\n  FPS warnings: skipped (--quick)")
    elif fps_warnings:
        print(f"\n  FPS warnings ({len(fps_warnings)} species):")
        for msg in sorted(fps_warnings, key=_warning_sort_key):
            print(f"    [WARN] {msg}")
    else:
        print("\n  FPS warnings: none")

    if first_frame_warnings:
        print(f"\n  First-frame vs bind-pose warnings ({len(first_frame_warnings)}):")
        for msg in sorted(first_frame_warnings, key=_warning_sort_key):
            print(f"    [WARN] {msg}")
    else:
        print("\n  First-frame vs bind-pose warnings: none")

    return 1 if failed else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export one bind/rest-pose GLB per Truebones species and validate "
            "bind-pose consistency across that species' source files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset-dir",
        default="",
        type=str,
        help=(
            "Root directory containing per-character subfolders with FBX/GLB files. "
            "Defaults to get_raw_data_dir() (Truebone_Z-OO)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="",
        type=str,
        help=(
            "Directory for exported <Species>_TPOSE.glb files. Defaults to a "
            "sibling Truebone_Z-OO_TPOSE directory."
        ),
    )
    parser.add_argument(
        "--filter",
        default="",
        type=str,
        help=(
            "Comma/semicolon-separated object-type names to process "
            "(e.g. 'Buffalo,Horse'). Default: all characters."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Re-export TPOSE GLBs even if the output file already exists. "
            "Validation runs either way."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Skip export, per-file bind-pose consistency checks, and FPS checks. "
            "Only check selected bind-pose symmetry plus first-frame vs bind-pose "
            "for TPOSE/species-name files."
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        default=False,
        help="Skip all interactive confirmation prompts (headless / automated mode).",
    )
    parser.add_argument(
        "--workers", "-j",
        default=16,
        type=int,
        help="Number of parallel Blender worker processes. Default 16.",
    )
    parser.add_argument(
        "--tolerance",
        default=TOLERANCE,
        type=float,
        help=f"Matrix comparison tolerance. Default {TOLERANCE}.",
    )
    parser.add_argument(
        "--symmetry-ignore-terminal-layers",
        default=SYMMETRY_IGNORE_TERMINAL_LAYERS,
        type=int,
        help=(
            "Ignore this many terminal leaf-side skeleton layers for bind-pose "
            "symmetry checks. Default 2."
        ),
    )
    parser.add_argument(
        "--symmetry-tolerance",
        default=SYMMETRY_RELATIVE_TOLERANCE,
        type=float,
        help=(
            "Relative tolerance for relaxed bind-pose symmetry checks. "
            f"Default {SYMMETRY_RELATIVE_TOLERANCE}."
        ),
    )
    args = parser.parse_args()

    only_objects: set[str] | None = {
        token.strip()
        for token in args.filter.replace(";", ",").split(",")
        if token.strip()
    } or None

    try:
        return validate_tpose_bind_pose(
            dataset_dir=args.dataset_dir or None,
            output_dir=args.output_dir or None,
            only_objects=only_objects,
            overwrite=args.overwrite,
            quick=args.quick,
            yes=args.yes,
            workers=args.workers,
            tolerance=args.tolerance,
            symmetry_ignore_terminal_layers=args.symmetry_ignore_terminal_layers,
            symmetry_tolerance=args.symmetry_tolerance,
        )
    except Exception as exc:
        print(f"ERROR: TPOSE bind-pose validation failed: {exc}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
