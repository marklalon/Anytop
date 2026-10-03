#!/usr/bin/env python3
"""Local tuning UI for Edit Packages (section 5.4).

Usage (from ``Anytop/``)::

    python -m motion_edit.ui.serve --packages outputs/edit_packages [--port 8770]

Serves ``index.html`` (three.js viewer, parameter panel, timeline,
diagnostics) and runs every edit through ``EditRuntime`` here, so the page,
``motion_edit.apply_edit`` and the exported files all show the same result.
Packages are the ``<clip>.edit/`` directories under ``--packages``
(recursive); their id is the path relative to it.

API::

    GET  /api/packages              package list (manifest summaries)
    GET  /api/package/<id>          manifest, skeleton, original frames, contacts, mesh
    GET  /api/mesh?id=              the package's skinned preview GLB (``motion_edit.mesh``)
    POST /api/load    {id, stretch_factor,         re-decompose in place
                       fullbody_ik}
    POST /api/apply   {id, params, events,         edited frames + diagnostics (full path);
                       globals}                    ``events`` moves the strike events / chain,
                                                   ``globals`` adds world rotations (mesh skinning)
    POST /api/contacts {id, joints, species}       another contact joint set; ``species``
                                                   also writes it to contact_overrides.json
    POST /api/contacts {id, mask}                  hand-edited contact intervals, (K, F) 0/1
    POST /api/passive {id, joints, species}        another passive joint set; ``species``
                                                   also writes it to passive_overrides.json
    POST /api/export  {id, params, events,         ``motion_edit.apply_edit --sidecar --glb`` on a
                       root_motion, mesh, name}    sidecar in the package's exports/; download link
    GET  /api/download?id=&file=                   an exported file

Contact and passive edits rewrite the package in place.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, unquote, urlparse

ANYTOP_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ANYTOP_ROOT not in sys.path:
    sys.path.insert(0, ANYTOP_ROOT)

import numpy as np  # noqa: E402

from motion_edit.mesh import mesh_calibration, mesh_source, preview_path  # noqa: E402
from motion_edit.package import EditPackage, find_packages, is_package_dir  # noqa: E402
from motion_edit.profile.skeleton import passive_layer  # noqa: E402
from motion_edit.runtime import (  # noqa: E402
    PARAM_SPECS,
    EditRuntime,
    UnsupportedParameterError,
    forward_kinematics,
    param_manifest,
)

UI_DIR = os.path.dirname(os.path.abspath(__file__))
VENDOR_DIR = os.path.join(UI_DIR, "vendor")
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8"}
EXPORTS_DIR = "exports"          # inside the package directory
GLB_TYPE = "model/gltf-binary"
# Decimals sent to the page: far below a pixel, and keeps the JSON small.
POSITION_DECIMALS = 5
ROTATION_DECIMALS = 6


def _flat(array: np.ndarray, decimals: int) -> list:
    return np.round(np.asarray(array, dtype=np.float64), decimals).ravel().tolist()


class PackageStore:
    """Packages under one root, loaded on demand and reloaded when their files change."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self._lock = threading.Lock()
        # held across a package's read -> rebuild -> save, so two edits cannot interleave
        self.edit_lock = threading.Lock()
        self._cache: dict[str, tuple[float, EditRuntime]] = {}

    def path(self, package_id: str) -> str:
        path = os.path.abspath(os.path.join(self.root, package_id))
        if os.path.commonpath([path, self.root]) != self.root or not is_package_dir(path):
            raise KeyError(package_id)
        return path

    def runtime(self, package_id: str) -> EditRuntime:
        path = self.path(package_id)
        stamp = max(os.path.getmtime(os.path.join(path, f)) for f in ("manifest.json", "data.npz"))
        with self._lock:
            cached = self._cache.get(package_id)
            if cached is None or cached[0] != stamp:
                cached = (stamp, EditRuntime(EditPackage.load(path)))
                self._cache[package_id] = cached
            return cached[1]

    def payload(self, package_id: str) -> dict:
        return package_payload(self.runtime(package_id), self.path(package_id), package_id)

    def replace(self, package_id: str, package: EditPackage) -> None:
        path = self.path(package_id)
        with self._lock:
            package.save(path)
            self._cache.pop(package_id, None)

    def listing(self) -> list[dict]:
        rows = []
        for package_id in find_packages(self.root):
            with open(os.path.join(self.root, package_id, "manifest.json"), "r", encoding="utf-8") as handle:
                m = json.load(handle)
            rows.append({
                "id": package_id,
                "clip": m.get("clip"),
                "object_type": m.get("object_type"),
                "is_loop": m.get("is_loop"),
                "stretch_factor": m.get("stretch_factor"),
                "fullbody_ik": m.get("fullbody_ik", True),
                "frames": m.get("frame_count"),
                "plants": (m.get("diagnostics") or {}).get("plants"),
                "runtime_version": m.get("runtime_version"),
            })
        return rows


def mesh_payload(package_dir: str, package_id: str) -> dict | None:
    """The skinned preview: where to fetch it, and per bone its driving joint and matrix."""
    calibration = mesh_calibration(package_dir)
    if calibration is None or preview_path(package_dir) is None:
        return None
    source = mesh_source(package_dir)
    return {"url": f"/api/mesh?id={quote(package_id)}", "bones": calibration["bones"],
            "exportable": bool(source and os.path.isfile(source))}


def package_payload(runtime: EditRuntime, package_dir: str, package_id: str) -> dict:
    pkg = runtime.package
    original = runtime.apply()
    manifest = json.loads(json.dumps(pkg.manifest))
    # this server's parameter set, not the decomposer's
    manifest["available_params"] = [n for n in PARAM_SPECS if n in runtime.available]
    manifest["params"] = param_manifest(manifest["available_params"])
    source = manifest["contacts"]["source"]
    origin = {}
    for j in pkg["contact_joints"].tolist():
        origin[j] = ("package" if j in source.get("package_add", []) else
                     "species" if j in source.get("species_add", []) else "cond")
    return {
        "manifest": manifest,
        "species_writable": bool(pkg.manifest.get("dataset_root")),
        "skeleton": {
            "parents": pkg["parents"].tolist(),
            "names": [str(n) for n in pkg["names"]],
            "bvh_names": [str(n) for n in pkg["bvh_names"]],
            "groups": [str(g) for g in pkg["chain_group"]],
            "roles": [str(r) for r in pkg["profile_role"]],
        },
        "original": {"positions": _flat(original.global_positions, POSITION_DECIMALS)},
        "contacts": {
            "joints": pkg["contact_joints"].tolist(),
            "origin": [origin[j] for j in pkg["contact_joints"].tolist()],
            "mask": pkg["contact_mask"].astype(np.uint8).T.tolist(),
            "plants": pkg["plant_intervals"].tolist(),
            "anchors": _flat(pkg["plant_anchor"], POSITION_DECIMALS),
            "drift": _flat(pkg["plant_drift"], POSITION_DECIMALS),
        },
        "passive": passive_payload(pkg),
        "ground_velocity": _flat(pkg["ground_velocity"], POSITION_DECIMALS),
        "mesh": mesh_payload(package_dir, package_id),
    }


def passive_payload(pkg: EditPackage) -> dict:
    """Passive joints with where each came from, and the candidates a click may add."""
    m = pkg.manifest.get("passive") or {}
    source = m.get("source") or {}
    joints = [int(j) for j in np.flatnonzero(pkg["profile_passive"])] if "profile_passive" in pkg.arrays else []
    origin = ["package" if j in source.get("package_add", []) else
              "species" if j in source.get("species_add", []) else "name" for j in joints]
    return {"joints": joints, "origin": origin, "candidates": m.get("candidates", []),
            "promotable": bool(source.get("package_add") or source.get("package_remove"))}


def apply_payload(runtime: EditRuntime, params: dict, events: dict | None = None,
                  global_rotations: bool = False) -> dict:
    # always the full decompose -> gains -> recompose path, never the all-default replay
    result = runtime.apply(params, compose=True, events=events)
    original = runtime.apply()
    extra = {}
    if global_rotations:
        rotations, _ = forward_kinematics(runtime.package["parents"],
                                          np.asarray(result.animation.rotations.qs, dtype=np.float64),
                                          np.asarray(result.animation.positions, dtype=np.float64))
        extra["global_rotations"] = _flat(rotations, ROTATION_DECIMALS)
    return {
        **extra,
        "params": result.params,
        "frames": int(result.global_positions.shape[0]),
        "positions": _flat(result.global_positions, POSITION_DECIMALS),
        "rotations": _flat(result.animation.rotations.qs, ROTATION_DECIMALS),
        "ground_velocity": _flat(result.ground_velocity, POSITION_DECIMALS),
        "source_time": _flat(result.source_time, 4),
        "planted": (result.plant_id >= 0).astype(np.uint8).ravel().tolist(),
        "targets": _flat(np.nan_to_num(result.plant_target), POSITION_DECIMALS),
        "unreached": result.unreached.astype(np.uint8).ravel().tolist(),
        "strike": ({k: result.strike[k] for k in ("chain", "windup", "impact", "recover", "joints")}
                   if result.strike else None),
        "max_position_delta": (float(np.abs(result.global_positions - original.global_positions).max())
                               if result.global_positions.shape == original.global_positions.shape else None),
        "diagnostics": result.diagnostics,
    }


def export_name(name, fallback: str) -> str:
    """A bare file stem: no directories, no extension, nothing a file system refuses."""
    stem = os.path.splitext(os.path.basename(str(name or "").strip()))[0]
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem).strip(" .")
    return stem or fallback


def _write_json(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    os.replace(tmp, path)


def sidecar_entry(runtime: EditRuntime, params: dict, events: dict | None) -> dict:
    """``{params, events}`` as the sidecar stores them: non-default values only, refused
    like an apply."""
    resolved, _ = runtime.resolve_params(params)
    runtime.strike(events)
    return {"params": {n: v for n, v in resolved.items() if v != PARAM_SPECS[n].default},
            "events": {k: v for k, v in (events or {}).items() if v is not None} or None}


class Handler(BaseHTTPRequestHandler):
    server_version = "MotionEditUI/1"

    def log_message(self, fmt, *args):
        if not str(args[0] if args else "").startswith(("GET /vendor", "GET /api/package/")):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    @property
    def store(self) -> PackageStore:
        return self.server.store

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _error(self, code: int, message: str) -> None:
        self._json(code, {"error": message})

    def _static(self, path: str) -> None:
        ext = os.path.splitext(path)[1].lower()
        with open(path, "rb") as handle:
            self._send(200, handle.read(), CONTENT_TYPES.get(ext, "application/octet-stream"))

    def _download(self, query: dict) -> None:
        directory = os.path.join(self.store.path((query.get("id") or [""])[0]), EXPORTS_DIR)
        path = os.path.abspath(os.path.join(directory, (query.get("file") or [""])[0]))
        if os.path.dirname(path) != os.path.abspath(directory) or not os.path.isfile(path):
            return self._error(404, "not found")
        with open(path, "rb") as handle:
            body = handle.read()
        self.send_response(200)
        self.send_header("Content-Type", GLB_TYPE if path.lower().endswith(".glb") else "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Disposition", "attachment; filename*=UTF-8''" + quote(os.path.basename(path)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        path = unquote(url.path)
        try:
            if path in ("/", "/index.html"):
                return self._static(os.path.join(UI_DIR, "index.html"))
            if path.startswith("/vendor/"):
                target = os.path.abspath(os.path.join(VENDOR_DIR, path[len("/vendor/"):]))
                if os.path.commonpath([target, VENDOR_DIR]) != VENDOR_DIR or not os.path.isfile(target):
                    return self._error(404, "not found")
                return self._static(target)
            if path == "/api/packages":
                return self._json(200, {"root": self.store.root, "packages": self.store.listing()})
            if path.startswith("/api/package/"):
                return self._json(200, self.store.payload(path[len("/api/package/"):]))
            if path == "/api/download":
                return self._download(parse_qs(url.query))
            if path == "/api/mesh":
                preview = preview_path(self.store.path((parse_qs(url.query).get("id") or [""])[0]))
                if preview is None:
                    return self._error(404, "this package has no mesh")
                with open(preview, "rb") as handle:
                    return self._send(200, handle.read(), GLB_TYPE)
            self._error(404, "not found")
        except KeyError as exc:
            self._error(404, f"unknown package {exc}")
        except Exception as exc:   # report, keep serving
            traceback.print_exc()
            self._error(500, f"{type(exc).__name__}: {exc}")

    def do_POST(self):
        path = unquote(urlparse(self.path).path)
        # a cross-site page can only send a JSON content type after a CORS preflight,
        # which this server never grants: no other page can trigger an edit
        if self.headers.get_content_type() != "application/json":
            return self._error(415, "POST bodies must be application/json")
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            package_id = str(body.get("id", ""))
            if path == "/api/apply":
                runtime = self.store.runtime(package_id)
                return self._json(200, apply_payload(runtime, body.get("params") or {}, body.get("events"),
                                                    bool(body.get("globals"))))
            if path == "/api/load":
                from motion_edit.decompose import redecompose

                with self.store.edit_lock:
                    runtime = self.store.runtime(package_id)
                    m = runtime.package.manifest
                    stretch = float(body.get("stretch_factor", m["stretch_factor"]))
                    if not 0.0 <= stretch <= 1.0:
                        return self._error(400, "stretch_factor must be in [0, 1]")
                    ik = bool(body.get("fullbody_ik", m.get("fullbody_ik", True)))
                    # always decomposed afresh, settings changed or not
                    self.store.replace(package_id, redecompose(runtime.package, stretch, fullbody_ik=ik))
                    return self._json(200, self.store.payload(package_id))
            if path == "/api/contacts":
                with self.store.edit_lock:
                    return self._json(200, self._contacts(package_id, body))
            if path == "/api/passive":
                with self.store.edit_lock:
                    return self._json(200, self._passive(package_id, body))
            if path == "/api/export":
                return self._json(200, self._export(package_id, body))
            self._error(404, "not found")
        except KeyError as exc:
            self._error(404, f"unknown package or field {exc}")
        except (UnsupportedParameterError, ValueError) as exc:
            self._error(400, str(exc))
        except Exception as exc:
            traceback.print_exc()
            self._error(500, f"{type(exc).__name__}: {exc}")


    def _export(self, package_id: str, body: dict) -> dict:
        """Write the parameter sidecar and run ``motion_edit.apply_edit --sidecar`` on it: the
        export is the command line's, and bpy gets the main thread it needs."""
        runtime = self.store.runtime(package_id)
        package_dir = self.store.path(package_id)
        stem = export_name(body.get("name"), str(runtime.package.manifest.get("clip") or "edit"))
        root_motion = bool(body.get("root_motion"))
        mesh = bool(body.get("mesh"))
        if mesh and not (mesh_payload(package_dir, package_id) or {}).get("exportable"):
            raise ValueError("this package has no T-pose mesh to skin on (or it moved)")
        entry = sidecar_entry(runtime, body.get("params") or {}, body.get("events"))
        directory = os.path.join(package_dir, EXPORTS_DIR)
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{stem}.glb")
        sidecar = path + ".json"
        _write_json(sidecar, {**entry, "root_motion": root_motion, "mesh": mesh})
        command = [sys.executable, "-m", "motion_edit.apply_edit", package_dir, "--sidecar", sidecar,
                   *(["--root_motion"] if root_motion else []), *(["--mesh"] if mesh else []),
                   "--glb", path]
        # the subprocess reads the package from disk: hold off contact / passive rewrites
        with self.store.edit_lock:
            done = subprocess.run(command, cwd=ANYTOP_ROOT, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace")
        if done.returncode != 0 or not os.path.isfile(path):
            log = (done.stderr or done.stdout or "").strip().splitlines()
            raise RuntimeError(f"export failed (exit {done.returncode}): " + " | ".join(log[-3:]))
        name = os.path.basename(path)
        return {"file": name, "path": path, "sidecar": sidecar, "command": subprocess.list2cmdline(command),
                "url": f"/api/download?id={quote(package_id)}&file={quote(name)}"}

    def _contacts(self, package_id: str, body: dict) -> dict:
        from motion_edit.decompose import with_contact_joints, with_contact_mask, with_ground_height

        package = self.store.runtime(package_id).package
        if body.get("ground_height") is not None:
            height = float(body["ground_height"])
            if not np.isfinite(height):
                raise ValueError("ground_height must be a finite number")
            edited = with_ground_height(package, height)
        elif body.get("mask") is not None:
            edited = with_contact_mask(package, np.asarray(body["mask"], dtype=bool).T)
        elif body.get("joints") is not None:
            joints = sorted({int(j) for j in body["joints"]})
            if any(j < 0 or j >= package.joint_count for j in joints):
                raise ValueError("contact joint index out of range")
            if body.get("species"):
                edited = self._write_species(package, joints)
            else:
                edited = with_contact_joints(package, joints)
        else:
            raise ValueError("send 'joints', 'mask' or 'ground_height'")
        self.store.replace(package_id, edited)
        return self.store.payload(package_id)

    def _passive(self, package_id: str, body: dict) -> dict:
        from motion_edit.decompose import with_passive_joints

        package = self.store.runtime(package_id).package
        if body.get("joints") is None:
            raise ValueError("send 'joints'")
        joints = {int(j) for j in body["joints"]}
        if any(j < 0 or j >= package.joint_count for j in joints):
            raise ValueError("passive joint index out of range")
        # taken as sent: the page closes an addition over its subtree itself
        joints = sorted(joints)
        candidates = set((package.manifest.get("passive") or {}).get("candidates", []))
        outside = [str(package["names"][j]) for j in joints if j not in candidates]
        if outside:
            raise ValueError(f"a support joint hangs below: {', '.join(outside)}")
        if body.get("species"):
            add, remove = passive_layer(package["parents"], package.manifest["passive"]["named"], joints)
            self._write_override(package, "passive", add, remove)
            edited = with_passive_joints(package, joints, species_add=add, species_remove=remove)
        else:
            edited = with_passive_joints(package, joints)
        self.store.replace(package_id, edited)
        return self.store.payload(package_id)

    @staticmethod
    def _write_override(package, kind: str, add: list[int], remove: list[int]) -> None:
        """Write the species' ``contact`` / ``passive`` override row (joint names)."""
        from motion_edit.profile.data import CONTACT_OVERRIDES_FILE, PASSIVE_OVERRIDES_FILE, write_species_override

        root = package.manifest.get("dataset_root")
        if not root or not os.path.isdir(root):
            raise ValueError("package does not know its dataset; re-decompose it with "
                             "decompose_clip from a dataset to write a species override")
        names = [str(n) for n in package["names"]]
        write_species_override(root, CONTACT_OVERRIDES_FILE if kind == "contact" else PASSIVE_OVERRIDES_FILE,
                               package.manifest["object_type"], [names[j] for j in add],
                               [names[j] for j in remove], package.manifest["skeleton_hash"])

    def _write_species(self, package, joints):
        """Record ``joints`` against cond as the species' contact override, then use it."""
        from motion_edit.decompose import with_contact_joints

        cond = set(package.manifest["contacts"]["source"]["cond"])
        add = sorted(set(joints) - cond)
        remove = sorted(cond - set(joints))
        self._write_override(package, "contact", add, remove)
        return with_contact_joints(package, joints, species_add=add, species_remove=remove)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--packages", required=True, help="directory holding <clip>.edit packages")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    if not os.path.isdir(args.packages):
        parser.error(f"--packages {args.packages} is not a directory")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.store = PackageStore(args.packages)
    url = f"http://{args.host}:{args.port}/"
    print(f"Serving {len(server.store.listing())} package(s) from {server.store.root} at {url}", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
