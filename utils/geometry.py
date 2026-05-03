"""
Geometry utility functions (differentiable via PyTorch).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


def finite_diff(x: Tensor, dt: float, dim: int = 0) -> Tensor:
    """Forward finite differences along *dim*.

    Returns tensor of length ``x.shape[dim] - 1`` along *dim*  (one fewer
    element than the input).  Forward differences are used uniformly;
    callers that need central differences at interior frames should use
    :func:`finite_diff_central` instead.
    """
    slices_fwd = [slice(None)] * x.ndim
    slices_bwd = [slice(None)] * x.ndim
    slices_fwd[dim] = slice(1, None)
    slices_bwd[dim] = slice(None, -1)
    return (x[tuple(slices_fwd)] - x[tuple(slices_bwd)]) / dt


def finite_diff_central(x: Tensor, dt: float, dim: int = 0) -> Tensor:
    """Central finite differences along *dim*.

    Interior frames use central differences ``(x[i+1] - x[i-1]) / (2*dt)``.
    Boundary frames (first and last) fall back to forward/backward differences.
    Returns tensor of the **same length** as *x* along *dim*.
    """
    n = x.shape[dim]

    def _sl(start, stop):
        s = [slice(None)] * x.ndim
        s[dim] = slice(start, stop)
        return tuple(s)

    # Interior: central diff (length n-2)
    interior = (x[_sl(2, None)] - x[_sl(None, n - 2)]) / (2.0 * dt)
    # Forward diff for first frame
    first = (x[_sl(1, 2)] - x[_sl(0, 1)]) / dt
    # Backward diff for last frame
    last  = (x[_sl(n - 1, None)] - x[_sl(n - 2, n - 1)]) / dt

    return torch.cat([first, interior, last], dim=dim)


def rotate_vector(v: Tensor, R: Tensor) -> Tensor:
    """Rotate vectors v [..., 3] by rotation matrices R [..., 3, 3]."""
    return (R @ v.unsqueeze(-1)).squeeze(-1)


def project_to_plane(points: Tensor, plane_origin: Tensor,
                     plane_normal: Tensor) -> Tensor:
    """Orthogonally project points [..., 3] onto a plane defined by
    origin + normal (both shape [3]).

    Returns projected points with the same shape as *points*.
    """
    n = F.normalize(plane_normal, dim=-1)
    offset = points - plane_origin
    dist = (offset * n).sum(-1, keepdim=True)
    return points - dist * n


def convex_hull_2d(points: Tensor) -> Tensor:
    """Compute 2-D convex hull vertices (non-differentiable, for geometry only).

    Args:
        points: [N, 2] tensor on CPU.
    Returns:
        [M, 2] hull vertices in counter-clockwise order.
    """
    import numpy as np
    from scipy.spatial import ConvexHull
    pts = points.detach().cpu().numpy()
    if len(pts) < 3:
        return points
    try:
        hull = ConvexHull(pts)
        return torch.from_numpy(pts[hull.vertices]).to(points.device)
    except Exception:
        return points


def point_in_convex_hull(point: Tensor, hull_vertices: Tensor,
                          eps: float = 1e-6) -> Tensor:
    """Test whether a 2-D point lies inside a convex polygon (non-differentiable).

    Returns a boolean scalar tensor.
    """
    import numpy as np
    from scipy.spatial import ConvexHull, Delaunay
    pts_np = hull_vertices.detach().cpu().numpy()
    p_np   = point.detach().cpu().numpy()
    if len(pts_np) < 3:
        return torch.tensor(False)
    tri = Delaunay(pts_np)


def encode_vertex_positions_as_colors(vertices: torch.Tensor) -> torch.Tensor:
    """Encode 3D vertex positions as RGB colors for debugging.

    Normalizes vertices to [0, 1] based on their collective bounding box.

    Args:
        vertices: [V, 3] float tensor.
    Returns:
        [V, 3] float tensor in [0, 1].
    """
    vmin = vertices.min(dim=0).values
    vmax = vertices.max(dim=0).values
    denom = vmax - vmin
    denom[denom < 1e-6] = 1.0  # avoid div-by-zero
    return (vertices - vmin) / denom
    inside = tri.find_simplex(p_np) >= 0
    return torch.tensor(bool(inside))


def signed_dist_to_convex_hull(point: Tensor, hull_vertices: Tensor) -> Tensor:
    """Signed distance from a 2-D point to the boundary of a convex polygon.

    Negative = inside, positive = outside.  Differentiable w.r.t. *point*
    via a soft approximation (log-sum-exp over half-planes).
    """
    # For each edge of the hull compute signed distance to half-plane
    n = hull_vertices.shape[0]
    dists = []
    for i in range(n):
        a = hull_vertices[i]
        b = hull_vertices[(i + 1) % n]
        edge = b - a
        normal = torch.stack([edge[1], -edge[0]])   # inward normal (CCW hull)
        normal = F.normalize(normal, dim=-1)
        dists.append(((point - a) * normal).sum(-1))
    dists = torch.stack(dists)                       # [n]
    # point is inside iff all half-plane distances > 0
    # signed dist to hull ≈ -min(dists)
    return -dists.min()
