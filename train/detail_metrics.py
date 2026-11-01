"""Physical preservation metrics shared by training validation and final evaluation."""
import torch

from model.motion_detail_refiner import fk, sample_context, spectral_project, ric_body_positions
from utils.rotation_conversions import rotation_6d_to_matrix_safe


def boundary_acceleration(signal, loop):
    """Discrete acceleration at the seam relative to the clip's interior.

    A static offset has zero cost; a periodic trajectory need not have equal
    first/last poses. This is a diagnostic, not a calibrated acceptance gate.
    The caller chooses normalized positions or rotation matrices and units.
    """
    if len(signal) < 4:
        return signal.new_zeros(()), signal.new_zeros(())
    if loop:
        accel = signal.roll(1, 0) - 2 * signal + signal.roll(-1, 0)
    else:
        accel = signal[:-2] - 2 * signal[1:-1] + signal[2:]
    magnitude = accel.flatten(1).square().mean(1).sqrt()
    boundary = magnitude[[0, -1]].max()
    interior = magnitude[1:-1].max() if len(magnitude) > 2 else magnitude.max()
    return boundary, (boundary - interior).clamp_min(0)


def preservation_metrics(out, base, y, config):
    rows = []
    for b in range(len(out)):
        n, t, loop, dt, parents = sample_context(y, b, config)
        x, ref = (v[b, :n, :, :t].permute(2, 0, 1) for v in (out, base))
        rest = y['rest_pos_ric_hml'][b, :n].to(x)
        scale = (rest - rest[:1]).square().sum(-1).mean().sqrt().clamp_min(1e-3)
        local, local_ref = (rotation_6d_to_matrix_safe(v[..., 3:9]) for v in (x, ref))
        global_, _ = fk(local, rest, parents)
        global_ref, _ = fk(local_ref, rest, parents)
        lp = lambda v: spectral_project(v, dt, loop, high=config.cutoff_hz)
        position_diff = (x[..., :3] - ref[..., :3]) / scale
        px, pr = (ric_body_positions(v[..., :3]) for v in (x, ref))
        body_position = (global_[:, :1].transpose(-1, -2) @ (px - px[:, :1])[..., None]).squeeze(-1) / scale
        body_position_ref = (global_ref[:, :1].transpose(-1, -2) @ (pr - pr[:, :1])[..., None]).squeeze(-1) / scale
        body = lp(body_position)
        body_ref = lp(body_position_ref)
        amplitude = (body - body.mean(0)).square().mean().sqrt()
        amplitude_ref = (body_ref - body_ref.mean(0)).square().mean().sqrt()
        bone_errors = []
        for j, p in enumerate(parents):
            if p >= 0:
                a = torch.linalg.vector_norm(px[:, j] - px[:, p], dim=-1)
                r = torch.linalg.vector_norm(pr[:, j] - pr[:, p], dim=-1)
                bone_errors.append((a - r) / scale)
        boundary = position_diff[0] - position_diff[-1] if loop else position_diff[[0, -1]]
        acceleration = {}
        for label, values in (('position', body_position), ('base_position', body_position_ref),
                              ('rotation', global_), ('base_rotation', global_ref)):
            magnitude, excess = boundary_acceleration(values, loop)
            acceleration[f'boundary_{label}_acceleration'] = float(magnitude / dt**2)
            acceleration[f'boundary_{label}_acceleration_excess'] = float(excess / dt**2)
        rows.append({
            **acceleration,
            'root_max_error': float((x[:, 0] - ref[:, 0]).abs().max()),
            'position_change_rms_relative': float(position_diff.square().mean().sqrt()),
            'local_rotation_change_rms': float((local - local_ref).square().mean().sqrt()),
            'global_rotation_change_rms': float((global_ - global_ref).square().mean().sqrt()),
            'lowfreq_position_rms_relative': float(lp(position_diff).square().mean().sqrt()),
            'lowfreq_rotation_matrix_rms': float(torch.maximum(lp(global_ - global_ref).square().mean().sqrt(), lp(local - local_ref).square().mean().sqrt())),
            'mean_rotation_matrix_rms': float((global_ - global_ref).mean(0).square().mean().sqrt()),
            'body_amplitude_relative_change': float((amplitude - amplitude_ref).abs() / amplitude_ref.clamp_min(.001)),
            'bone_length_rms_relative': float(torch.stack(bone_errors).square().mean().sqrt()),
            'boundary_position_change_relative': float(boundary.square().mean().sqrt()),
        })
    return {k: max(row[k] for row in rows) for k in rows[0]}


