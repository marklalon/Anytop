"""
Differentiable quaternion utilities (PyTorch).

Convention: quaternions stored as [w, x, y, z] with shape (..., 4).
All operations are autograd-compatible.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Basic operations
# ---------------------------------------------------------------------------

def quat_normalize(q: Tensor) -> Tensor:
    """Normalise quaternion(s) to unit length."""
    return F.normalize(q, p=2, dim=-1)


def quat_conjugate(q: Tensor) -> Tensor:
    """Return conjugate q* = [w, -x, -y, -z]."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_multiply(q1: Tensor, q2: Tensor) -> Tensor:
    """Hamilton product q1 ⊗ q2.  Both tensors broadcast together."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=-1)


def quat_diff(q1: Tensor, q2: Tensor) -> Tensor:
    """Relative rotation from q2 to q1: q_diff = q1 ⊗ q2*."""
    return quat_multiply(q1, quat_conjugate(q2))


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------

def quat_to_matrix(q: Tensor) -> Tensor:
    """Convert unit quaternion(s) [..., 4] → rotation matrix [..., 3, 3]."""
    q = quat_normalize(q)
    w, x, y, z = q.unbind(-1)
    tx, ty, tz = 2*x, 2*y, 2*z
    twx, twy, twz = tx*w, ty*w, tz*w
    txx, txy, txz = tx*x, ty*x, tz*x
    tyy, tyz, tzz = ty*y, tz*y, tz*z
    mat = torch.stack([
        1 - (tyy + tzz),  txy - twz,        txz + twy,
        txy + twz,        1 - (txx + tzz),  tyz - twx,
        txz - twy,        tyz + twx,        1 - (txx + tyy),
    ], dim=-1)
    return mat.view(*q.shape[:-1], 3, 3)


def matrix_to_quat(R: Tensor) -> Tensor:
    """Convert rotation matrix [..., 3, 3] → unit quaternion [..., 4] [w,x,y,z]."""
    # Shepperd's method
    batch = R.shape[:-2]
    m = R.view(-1, 3, 3)
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = torch.zeros(m.shape[0], 4, dtype=R.dtype, device=R.device)

    s = torch.sqrt(F.relu(trace + 1.0)) * 2  # 4*w
    w = 0.25 * s
    x = (m[:, 2, 1] - m[:, 1, 2]) / (s + 1e-8)
    y = (m[:, 0, 2] - m[:, 2, 0]) / (s + 1e-8)
    z = (m[:, 1, 0] - m[:, 0, 1]) / (s + 1e-8)
    mask0 = trace > 0
    q[mask0] = torch.stack([w, x, y, z], dim=-1)[mask0]

    # Handle degenerate cases (trace <= 0)
    c0 = (~mask0) & (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2])
    s2 = torch.sqrt(F.relu(1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2])) * 2
    q[c0, 0] = (m[c0, 2, 1] - m[c0, 1, 2]) / (s2[c0] + 1e-8)
    q[c0, 1] = 0.25 * s2[c0]
    q[c0, 2] = (m[c0, 0, 1] + m[c0, 1, 0]) / (s2[c0] + 1e-8)
    q[c0, 3] = (m[c0, 0, 2] + m[c0, 2, 0]) / (s2[c0] + 1e-8)

    c1 = (~mask0) & (~c0) & (m[:, 1, 1] > m[:, 2, 2])
    s3 = torch.sqrt(F.relu(1.0 + m[:, 1, 1] - m[:, 0, 0] - m[:, 2, 2])) * 2
    q[c1, 0] = (m[c1, 0, 2] - m[c1, 2, 0]) / (s3[c1] + 1e-8)
    q[c1, 1] = (m[c1, 0, 1] + m[c1, 1, 0]) / (s3[c1] + 1e-8)
    q[c1, 2] = 0.25 * s3[c1]
    q[c1, 3] = (m[c1, 1, 2] + m[c1, 2, 1]) / (s3[c1] + 1e-8)

    c2 = (~mask0) & (~c0) & (~c1)
    s4 = torch.sqrt(F.relu(1.0 + m[:, 2, 2] - m[:, 0, 0] - m[:, 1, 1])) * 2
    q[c2, 0] = (m[c2, 1, 0] - m[c2, 0, 1]) / (s4[c2] + 1e-8)
    q[c2, 1] = (m[c2, 0, 2] + m[c2, 2, 0]) / (s4[c2] + 1e-8)
    q[c2, 2] = (m[c2, 1, 2] + m[c2, 2, 1]) / (s4[c2] + 1e-8)
    q[c2, 3] = 0.25 * s4[c2]

    return F.normalize(q.view(*batch, 4), dim=-1)


def quat_to_axis_angle(q: Tensor) -> Tensor:
    """Convert unit quaternion(s) to axis-angle [..., 3].  Magnitude = angle."""
    q = quat_normalize(q)
    angle = 2.0 * torch.acos(q[..., :1].clamp(-1.0, 1.0))
    axis  = F.normalize(q[..., 1:], dim=-1)
    return axis * angle


def axis_angle_to_quat(aa: Tensor) -> Tensor:
    """Convert axis-angle [..., 3] → quaternion [..., 4]."""
    angle = aa.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis  = aa / angle
    half  = angle * 0.5
    w = torch.cos(half)
    xyz = axis * torch.sin(half)
    return torch.cat([w, xyz], dim=-1)


def quat_to_euler(q: Tensor, order: str = "xyz") -> Tensor:
    """Quaternion → Euler angles [..., 3] (radians).  order ∈ {"xyz","zyx",...}."""
    R = quat_to_matrix(q)
    if order == "xyz":
        sy = R[..., 0, 2]
        cy = torch.sqrt(R[..., 0, 0]**2 + R[..., 0, 1]**2 + 1e-12)
        x = torch.atan2(-R[..., 1, 2], R[..., 2, 2])
        y = torch.atan2(sy, cy)
        z = torch.atan2(-R[..., 0, 1], R[..., 0, 0])
        return torch.stack([x, y, z], dim=-1)
    raise NotImplementedError(f"Euler order '{order}' not implemented")


def euler_to_quat(euler: Tensor, order: str = "xyz") -> Tensor:
    """Euler angles [..., 3] (radians) → quaternion [..., 4]."""
    if order == "xyz":
        ax = axis_angle_to_quat(euler[..., :1] * torch.tensor([1., 0., 0.], device=euler.device))
        ay = axis_angle_to_quat(euler[..., 1:2] * torch.tensor([0., 1., 0.], device=euler.device))
        az = axis_angle_to_quat(euler[..., 2:3] * torch.tensor([0., 0., 1.], device=euler.device))
        return quat_multiply(az, quat_multiply(ay, ax))
    raise NotImplementedError(f"Euler order '{order}' not implemented")


# ---------------------------------------------------------------------------
# Kinematics helpers
# ---------------------------------------------------------------------------

def quat_angular_velocity(q: Tensor, dt: float) -> Tensor:
    """Estimate angular velocity [F-1, 3] from quaternion sequence [F, 4].

    Uses finite differences on the quaternion manifold:
        omega ≈ 2 * log(q_{t+1} ⊗ q_t*) / dt
    """
    dq   = quat_multiply(q[1:], quat_conjugate(q[:-1]))
    dq   = quat_normalize(dq)
    aa   = quat_to_axis_angle(dq)   # [F-1, 3], magnitude = rotation angle
    return aa / dt


def quat_slerp(q0: Tensor, q1: Tensor, t: Tensor) -> Tensor:
    """Spherical linear interpolation between q0 and q1.

    Args:
        q0, q1: [..., 4] unit quaternions
        t:      [...] scalar in [0, 1]
    """
    dot = (q0 * q1).sum(-1, keepdim=True).clamp(-1.0, 1.0)
    # Ensure shortest path
    q1_ = torch.where(dot < 0, -q1, q1)
    dot_ = dot.abs()
    theta = torch.acos(dot_.clamp(max=1.0 - 1e-7))
    sin_theta = torch.sin(theta).clamp(min=1e-8)
    t = t.unsqueeze(-1)
    w0 = torch.sin((1 - t) * theta) / sin_theta
    w1 = torch.sin(t * theta) / sin_theta
    # Fall back to lerp near theta=0
    lerp = (1 - t) * q0 + t * q1_
    slerp_val = w0 * q0 + w1 * q1_
    return quat_normalize(torch.where(theta < 1e-4, lerp, slerp_val))
