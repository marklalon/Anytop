from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from data_loaders.truebones.truebones_utils.canonical_features import (  # noqa: E402
    CANONICAL_STAT_LEN,
    canonical_to_physical_hml,
)
from data_loaders.truebones.truebones_utils.param_utils import MAX_SOURCE_FRAMES_MULT
from diffusion.gaussian_diffusion import GaussianDiffusion, LossType, ModelMeanType, ModelVarType
from sample.conditioning import create_condition
from sample.output_lengths import (  # noqa: E402
    _finalize_output_lengths,
    _resample_window_to_output,
)
from sample.reference_motion import (  # noqa: E402
    _prepare_img2img_reference_bundle,
    _validate_reference_motion_path,
)
from sample.sampling import _sample_batch


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(1, 1)


class _CaptureDiffusion:
    def __init__(self, return_values: list[object] | None = None) -> None:
        self.last_call = None
        self.last_kwargs = None
        self.calls = []
        self.return_values = list(return_values or [])

    def _record_call(self, name: str, kwargs: dict) -> object:
        self.last_call = name
        self.last_kwargs = kwargs
        self.calls.append((name, kwargs))
        if self.return_values:
            return self.return_values.pop(0)
        return name

    def p_sample_loop(self, **kwargs):
        return self._record_call("ddpm", kwargs)

    def ddim_sample_loop(self, **kwargs):
        return self._record_call("ddim", kwargs)


def _make_full_cond_entry(n_joints: int, feature_len: int = 12) -> dict:
    parents = np.array([-1] + list(range(n_joints - 1)), dtype=np.int64)
    return {
        "joints_names": [f"Joint{i}" for i in range(n_joints)],
        "canonical_joint_names": [f"Joint{i}" for i in range(n_joints)],
        "canonical_bvh_joint_names": [f"Joint{i}" for i in range(n_joints)],
        "joint_side_labels": ["center"] * n_joints,
        "parents": parents,
        "canonical_feature_mean": np.zeros((CANONICAL_STAT_LEN,), dtype=np.float32),
        "canonical_feature_std": np.ones((CANONICAL_STAT_LEN,), dtype=np.float32),
        "rest_pose": np.zeros((n_joints, feature_len), dtype=np.float32),
        "joint_relations": np.zeros((n_joints, n_joints), dtype=np.int64),
        "joints_graph_dist": np.zeros((n_joints, n_joints), dtype=np.int64),
        "offsets": np.zeros((n_joints, 3), dtype=np.float32),
        "joints_names_embs": np.zeros((n_joints, 4), dtype=np.float32),
    }


def _make_diffusion(num_steps: int = 2) -> GaussianDiffusion:
    betas = np.linspace(0.001, 0.002, num_steps, dtype=np.float64)
    return GaussianDiffusion(
        betas=betas,
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )


def test_finalize_output_lengths_returns_frames_and_resample_speed() -> None:
    requested, target, resample_speed = _finalize_output_lengths(
        requested_frames=90, min_length=20, internal_num_frames=60
    )
    assert requested == 90
    assert target == 90
    assert resample_speed == pytest.approx(90.0 / 60.0)


def test_finalize_output_lengths_rejects_out_of_window() -> None:
    internal_num_frames = 60
    min_length = 20
    max_frames = MAX_SOURCE_FRAMES_MULT * internal_num_frames

    with pytest.raises(SystemExit):
        _finalize_output_lengths(
            requested_frames=min_length - 1,
            min_length=min_length,
            internal_num_frames=internal_num_frames,
        )
    # The upper bound is inclusive: the full source-frame budget is allowed,
    # one frame past it is not.
    requested, target, resample_speed = _finalize_output_lengths(
        requested_frames=max_frames,
        min_length=min_length,
        internal_num_frames=internal_num_frames,
    )
    assert (requested, target) == (max_frames, max_frames)
    assert resample_speed == pytest.approx(float(MAX_SOURCE_FRAMES_MULT))
    with pytest.raises(SystemExit):
        _finalize_output_lengths(
            requested_frames=max_frames + 1,
            min_length=min_length,
            internal_num_frames=internal_num_frames,
        )


def test_validate_reference_motion_path_accepts_supported_suffixes() -> None:
    assert _validate_reference_motion_path("clip.npy") == ".npy"
    assert _validate_reference_motion_path("clip.fbx") == ".fbx"
    assert _validate_reference_motion_path("clip.glb") == ".glb"
    assert _validate_reference_motion_path("clip.gltf") == ".gltf"


def test_validate_reference_motion_path_rejects_unsupported_suffix() -> None:
    with pytest.raises(ValueError, match="Unsupported reference motion format"):
        _validate_reference_motion_path("clip.txt")


def test_prepare_reference_bundle_uses_preloaded_cropped_features() -> None:
    # Crop path: feed exactly M=40 frames (as main() does for R > M). The bundle
    # must consume the preloaded array verbatim (no disk load) and not re-trim it.
    n_joints, feat = 3, 12
    cond = _make_full_cond_entry(n_joints, feature_len=feat)
    preloaded = np.random.default_rng(0).normal(
        size=(40, n_joints, feat)
    ).astype(np.float32)

    bundle = _prepare_img2img_reference_bundle(
        reference_motion_path="/nonexistent/should_not_be_loaded.npy",
        target_type="Horse",
        target_cond=cond,
        max_joints=n_joints,
        target_feature_len=feat,
        batch_size=2,
        requested_output_frame_count=60,
        requested_visible_frame_count=40,
        preloaded_features=preloaded,
    )

    assert bundle["loaded_reference_frame_count"] == 40
    assert bundle["loaded_reference_joint_count"] == n_joints
    # The model is a fixed-window model trained only at num_frames. The bundle
    # always runs at that native window (requested_output_frame_count=60) and
    # resamples the shorter reference up to it; the requested output length is
    # honored later by resampling the sampled motion. reference_source_frame_count
    # records the pre-resample reference length (40) for resample_speed.
    assert bundle["output_frame_count"] == 60
    assert bundle["reference_source_frame_count"] == 40
    assert tuple(bundle["reference_motion"].shape) == (2, n_joints, feat, 60)


@pytest.mark.parametrize("selection", ["explicit", "filename", "single_cond"])
def test_generate_npy_edit_uses_selected_target(tmp_path, monkeypatch, selection):
    from types import SimpleNamespace
    import sample.generate as generate

    human = "integrate_20260924/MXO_Human"
    wolf = "integrate_20260924/ORA_SnakeWolf"
    cond_dict = {human: _make_full_cond_entry(53), wolf: _make_full_cond_entry(37)}
    target = wolf if selection == "filename" else human
    if selection == "single_cond":
        cond_dict = {human: cond_dict[human]}
    reference = tmp_path / "ORA_SnakeWolf_Walk_reconstruction.npy"
    np.save(reference, _synthetic_cycle(50, joints=len(cond_dict[target]['parents'])))
    args = SimpleNamespace(
        seed=0, reference_motion=str(reference), skip_timesteps=5,
        object_type="MXO_Human" if selection == "explicit" else None,
        cond_path="custom_cond.npy" if selection == "single_cond" else None,
        model_path="unused.pt", output_dir=str(tmp_path / "output"),
        num_frames=None, loop="off", batch_size=1,
    )
    runtime = SimpleNamespace(
        opt=SimpleNamespace(fps=30, max_joints=53), cond_dict=cond_dict,
        actual_cond_file="unused_cond.npy", model=SimpleNamespace(feature_len=12),
        diffusion=None, sampling_method="ddpm", sampling_steps=10, amp_dtype="fp32",
    )
    monkeypatch.setattr(generate, 'prepare_generation_runtime', lambda *a, **kw: runtime)
    monkeypatch.setattr(generate, '_checkpoint_native_num_frames', lambda *a: 60)
    monkeypatch.setattr(generate, '_resolve_action_condition', lambda *a: None)

    class ReferencePrepared(Exception):
        pass

    def capture_reference(path, target_type, target_cond, **kwargs):
        assert target_type == target
        assert target_cond is cond_dict[target]
        bundle = _prepare_img2img_reference_bundle(path, target_type, target_cond, **kwargs)
        assert bundle['reference_motion'].shape == (1, len(target_cond['parents']), 12, 60)
        raise ReferencePrepared

    monkeypatch.setattr(generate, '_prepare_img2img_reference_bundle', capture_reference)
    with pytest.raises(ReferencePrepared):
        generate.main(args)


@pytest.mark.parametrize("reference_joints", [2, 4])
def test_reference_bundle_rejects_wrong_joint_count(reference_joints):
    with pytest.raises(ValueError, match=rf"has {reference_joints} joints.*requires 3"):
        _prepare_img2img_reference_bundle(
            reference_motion_path="Horse_Walk.npy", target_type="Horse",
            target_cond=_make_full_cond_entry(3), max_joints=3,
            target_feature_len=12, batch_size=1, requested_output_frame_count=60,
            preloaded_features=np.zeros((40, reference_joints, 12), dtype=np.float32),
        )


def _synthetic_cycle(period: int, joints: int = 3, *, closing_key: bool = False) -> np.ndarray:
    """(T, J, 12) HML clip running one sine cycle over ``period`` frames, with
    velocity channels that are the true frame deltas and a wrap-delta terminal
    row -- what preprocessing stores for a loop. ``closing_key`` appends frame 0
    again as the last frame."""
    frame_count = period + int(closing_key)
    phase = 2.0 * np.pi * (np.arange(frame_count) % period) / period
    clip = np.zeros((frame_count, joints, 12), dtype=np.float32)
    for joint in range(joints):
        clip[:, joint, 0] = 0.30 * np.sin(phase + joint)
        clip[:, joint, 1] = 1.0 + 0.10 * np.cos(phase + joint)
        clip[:, joint, 3] = np.cos(phase - joint)
        clip[:, joint, 4] = np.sin(phase - joint)
        clip[:, joint, 8] = 1.0
    positions = clip[:, :, 0:3]
    clip[:-1, :, 9:12] = positions[1:] - positions[:-1]
    clip[-1, :, 9:12] = positions[0] - positions[-1]
    return clip


def _seam_ratio(window: np.ndarray) -> float:
    """|x[0] - x[-1]| over the mean of the two steps beside it: 1.0 means the
    seam is an ordinary step, i.e. the window is a uniform cycle."""
    pose = window[:, :, :9].astype(np.float64)
    step = lambda a, b: float(np.linalg.norm(pose[a] - pose[b]))
    return step(0, -1) / (0.5 * (step(-1, -2) + step(1, 0)))


def _reference_window(cond, clip, output_frame_count=60):
    """The reference as it is placed into the model window, in physical space."""
    bundle = _prepare_img2img_reference_bundle(
        reference_motion_path="/nonexistent/should_not_be_loaded.npy",
        target_type="Horse",
        target_cond=cond,
        max_joints=clip.shape[1],
        target_feature_len=clip.shape[2],
        batch_size=1,
        requested_output_frame_count=output_frame_count,
        preloaded_features=clip,
    )
    window = bundle["reference_motion"][0].permute(2, 0, 1).numpy()  # [J, F, T] -> [T, J, F]
    return canonical_to_physical_hml(window, cond)


def test_reference_bundle_fills_the_window_end_to_end() -> None:
    # The reference is a one-shot clip whatever the loop verdict says, so it is
    # stretched end to end: window frame 0 and frame T-1 are the clip's own
    # first and last poses, and the wrap is left as the seam it is. --loop is a
    # statement about the WINDOW -- the model's is_loop condition -- and closing
    # the cycle is the model's job, not the fill's.
    n_joints, feat = 3, 12
    cond = _make_full_cond_entry(n_joints, feature_len=feat)
    clip = _synthetic_cycle(45)

    window = _reference_window(cond, clip)

    # Pose channels only: the resample rescales the velocity rows with the step.
    np.testing.assert_allclose(window[0, :, :9], clip[0, :, :9], atol=1e-6)
    np.testing.assert_allclose(window[-1, :, :9], clip[-1, :, :9], atol=1e-6)
    assert abs(_seam_ratio(window) - 1.0) > 0.25, _seam_ratio(window)


def test_reference_bundle_keeps_a_closing_key() -> None:
    # Nothing on the reference path reads the clip as a period any more, so a
    # stored loop whose last frame repeats frame 0 hands the window one frame
    # more of motion than the bare cycle does. It is the user's data.
    n_joints, feat = 3, 12
    cond = _make_full_cond_entry(n_joints, feature_len=feat)
    clean = _synthetic_cycle(45)
    with_key = _synthetic_cycle(45, closing_key=True)

    assert not np.allclose(
        _reference_window(cond, with_key),
        _reference_window(cond, clean),
        atol=1e-6,
    )


def test_window_to_output_keeps_a_loop_a_cycle() -> None:
    # The sampled window is a cycle under --loop, so rescaling it to the
    # requested output length must be periodic. Endpoint resampling left the
    # exported clip's wrap step at 1 source frame against (T-1)/(M-1) inside,
    # which is what "--loop + --reference_motion produced a non-loop" was.
    window = _synthetic_cycle(60)

    for target in (30, 90, 97):
        exported = _resample_window_to_output(window, target, 60, True)
        assert exported.shape[0] == target
        assert abs(_seam_ratio(exported) - 1.0) < 0.05, (target, _seam_ratio(exported))

        open_export = _resample_window_to_output(window, target, 60, False)
        assert abs(_seam_ratio(open_export) - 1.0) > 0.25, (target, _seam_ratio(open_export))

    # A window already at the requested length is a no-op either way.
    assert _resample_window_to_output(window, 60, 60, True) is window
    assert _resample_window_to_output(window, 60, 60, False) is window


def test_loop_export_map_is_the_windows_own() -> None:
    # The reference goes in end to end and the sampled window comes out
    # periodically under --loop, because it is the WINDOW that is declared a
    # cycle. The two are therefore not inverse: a clamped reference pose is
    # exported a frame or two from where it was placed (the drift
    # _resample_window_to_output documents, which the mask's union mapping
    # covers), and in exchange the exported seam stays an ordinary step
    # instead of a stall.
    n_joints, feat = 3, 12
    cond = _make_full_cond_entry(n_joints, feature_len=feat)
    clip = _synthetic_cycle(30)

    window = _reference_window(cond, clip)
    exported = _resample_window_to_output(window, 30, 60, True)

    assert exported.shape == clip.shape
    # Same motion, at most ~2 frames of drift: every exported frame matches the
    # reference within that neighbourhood.
    pose = lambda a: a[:, :, :9].astype(np.float64)
    scale = float(np.abs(np.diff(pose(clip), axis=0)).mean())
    for s_idx in range(30):
        drift = min(
            float(np.abs(pose(exported)[s_idx] - pose(clip)[r]).mean())
            for r in range(max(0, s_idx - 2), min(30, s_idx + 3))
        )
        assert drift < 0.5 * scale, (s_idx, drift, scale)


def test_create_condition_can_sample_at_target_joint_count() -> None:
    cond_dict = {"Horse": _make_full_cond_entry(3)}

    motion_batch, model_kwargs = create_condition(
        ["Horse", "Horse"],
        cond_dict,
        n_frames=4,
        max_joints=3,
        feature_len=12,
    )

    y = model_kwargs["y"]
    assert tuple(motion_batch.shape) == (2, 3, 12, 4)
    assert tuple(y["joints_padding_mask"].shape) == (2, 1, 1, 4, 4)
    assert tuple(y["graph_dist"].shape) == (2, 3, 3)
    assert torch.equal(y["n_joints"], torch.tensor([3, 3]))
    # The output coordinate frame is an unconditional model input: MotionDiffusion.forward
    # reads it every step, so the generation path has to stack it per sample too.
    assert tuple(y["canonical_feature_mean"].shape) == (2, CANONICAL_STAT_LEN)
    assert tuple(y["canonical_feature_std"].shape) == (2, CANONICAL_STAT_LEN)


def test_create_condition_rejects_cond_entry_without_canonical_stats() -> None:
    # A stats-free cond can only produce a y the model cannot read, so generation
    # must fail loudly at the boundary rather than hand forward an incomplete y.
    entry = _make_full_cond_entry(3)
    del entry["canonical_feature_mean"]
    cond_dict = {"Horse": entry}

    with pytest.raises(KeyError, match="canonical_feature_mean"):
        create_condition(
            ["Horse"],
            cond_dict,
            n_frames=4,
            max_joints=3,
            feature_len=12,
        )


def test_p_sample_loop_visits_final_timestep(monkeypatch: pytest.MonkeyPatch) -> None:
    diffusion = _make_diffusion(num_steps=4)
    reverse_ts: list[int] = []
    sample = torch.zeros((1, 1, 1, 1), dtype=torch.float32)

    def fake_p_sample(model, x, t, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, const_noise=False):
        reverse_ts.append(int(t.item()))
        return {"sample": sample.clone(), "pred_xstart": torch.zeros_like(sample)}

    monkeypatch.setattr(diffusion, "p_sample", fake_p_sample)

    diffusion.p_sample_loop(
        _DummyModel(),
        shape=tuple(sample.shape),
        noise=torch.zeros_like(sample),
        device=torch.device("cpu"),
        progress=False,
    )

    assert reverse_ts == [3, 2, 1, 0]


def test_ddim_sample_loop_visits_final_timestep(monkeypatch: pytest.MonkeyPatch) -> None:
    diffusion = _make_diffusion(num_steps=4)
    reverse_ts: list[int] = []
    sample = torch.zeros((1, 1, 1, 1), dtype=torch.float32)

    def fake_ddim_sample(model, x, t, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, eta=0.0):
        reverse_ts.append(int(t.item()))
        return {"sample": sample.clone(), "pred_xstart": torch.zeros_like(sample)}

    monkeypatch.setattr(diffusion, "ddim_sample", fake_ddim_sample)

    diffusion.ddim_sample_loop(
        _DummyModel(),
        shape=tuple(sample.shape),
        noise=torch.zeros_like(sample),
        device=torch.device("cpu"),
        progress=False,
        eta=0.0,
    )

    assert reverse_ts == [3, 2, 1, 0]


def test_ddim_sample_loop_rejects_const_noise() -> None:
    diffusion = _make_diffusion(num_steps=2)

    with pytest.raises(NotImplementedError):
        diffusion.ddim_sample_loop(
            _DummyModel(),
            shape=(1, 1, 1, 1),
            const_noise=True,
        )



@pytest.mark.parametrize("method", ["ddpm", "ddim"])
@pytest.mark.parametrize("skip", [0, 2])
def test_sample_batch_plain_and_whole_reference_img2img(method, skip):
    diffusion = _CaptureDiffusion()
    reference = torch.randn(2, 3, 12, 4)
    condition = {"y": {"n_joints": torch.tensor([3, 3])}}
    result = _sample_batch(
        diffusion, _DummyModel(), condition, method, tuple(reference.shape),
        ddim_eta=0.0, seed=42, device=torch.device("cpu"),
        reference_motion=reference if skip else None, skip_timesteps=skip,
        progress=False,
    )
    assert result == method
    call = diffusion.last_kwargs
    assert call["skip_timesteps"] == skip
    assert call["model_kwargs"] is condition
    assert torch.equal(call["init_image"], reference) if skip else call["init_image"] is None
    assert call["noise"].shape == reference.shape
