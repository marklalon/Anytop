"""
Linear Blend Skinning (LBS) — differentiable vertex deformation.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor


def lbs_deform(
    vertices_rest:  Tensor,     # [V, 3]   rest-pose vertex positions
    skin_weights:   Tensor,     # [V, J]   per-vertex bone weights (sum=1)
    bone_transforms: Tensor,    # [F, J, 4, 4]  world bone matrices
    bind_matrices:  Tensor,     # [J, 4, 4]    inverse bind-pose matrices
) -> Tensor:
    """Apply LBS to deform a mesh.

    Returns:
        deformed_vertices: [F, V, 3]
    """
    F_frames, J = bone_transforms.shape[:2]
    V = vertices_rest.shape[0]

    # Skinning matrix for each bone: M_skin = M_world @ M_bind_inv
    # [F, J, 4, 4]
    skinning_mats = bone_transforms @ bind_matrices.unsqueeze(0)

    # Blend skinning matrices per vertex using einsum (avoids large intermediate)
    # skin_weights [V, J], skinning_mats [F, J, 4, 4] -> T_blended [F, V, 4, 4]
    T_blended = torch.einsum('vj,fjab->fvab', skin_weights, skinning_mats)

    # Homogeneous rest vertices [V, 4]
    ones = torch.ones(V, 1, dtype=vertices_rest.dtype, device=vertices_rest.device)
    verts_h = torch.cat([vertices_rest, ones], dim=-1)  # [V, 4]

    # Deform: [F, V, 4, 4] @ [V, 4] → [F, V, 4] → [F, V, 3]
    deformed_h = torch.einsum('fvab,vb->fva', T_blended, verts_h)
    return deformed_h[..., :3]


def normalize_skin_weights(skin_weights: Tensor) -> Tensor:
    """Ensure per-vertex bone weights sum to 1."""
    return skin_weights / (skin_weights.sum(-1, keepdim=True) + 1e-8)
