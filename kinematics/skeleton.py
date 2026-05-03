"""
Skeleton data structure.

A Skeleton holds a list of Bone objects in topological (parent-before-child)
order together with the rest-pose bind matrices required for LBS.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import torch
from torch import Tensor


@dataclass
class Bone:
    """Single bone in the skeletal hierarchy."""
    id:            int
    name:          str
    parent_id:     Optional[int]             # None = root
    # Rest-pose local offset from parent joint (world-space when root)
    rest_offset:   Tensor                    # [3]
    # Rest-pose local rotation (quaternion [w,x,y,z])
    rest_rotation: Tensor                    # [4]
    # Children ids (populated by Skeleton after construction)
    children_ids:  List[int] = field(default_factory=list)

    @property
    def is_root(self) -> bool:
        return self.parent_id is None


class Skeleton:
    """Immutable skeleton hierarchy with convenience accessors.

    Attributes:
        bones:          list of Bone in topological order
        bind_matrices:  [J, 4, 4] inverse bind-pose matrices for LBS
        joint_limits:   optional [J, 3, 2] per-joint per-axis (min, max) radians
    """

    def __init__(self, bones: List[Bone],
                 bind_matrices: Optional[Tensor] = None,
                 joint_limits:  Optional[Tensor] = None):
        self.bones = bones
        self._name_to_id: Dict[str, int] = {b.name: b.id for b in bones}

        # Wire up children
        for b in bones:
            if b.parent_id is not None:
                bones[b.parent_id].children_ids.append(b.id)

        J = len(bones)
        device = bones[0].rest_offset.device if bones else torch.device("cpu")

        # Default bind matrices = identity
        self.bind_matrices: Tensor = (
            bind_matrices if bind_matrices is not None
            else torch.eye(4, device=device).unsqueeze(0).expand(J, -1, -1).clone()
        )

        # Default limits = unconstrained (-π, π)
        self.joint_limits: Optional[Tensor] = joint_limits

        # Pre-compute depth-batched topology for fast FK
        self._build_depth_levels(device)

    # ------------------------------------------------------------------
    # Depth-batched topology cache (for vectorised FK)
    # ------------------------------------------------------------------

    def _build_depth_levels(self, device: torch.device) -> None:
        """Pre-compute joint groups by tree depth and stacked offsets.

        After this call the following attributes are available:
          depth_levels: list of (bone_ids, parent_ids) tuples per depth.
                        depth 0 = root bones (parent_id is None).
          rest_offsets: [J, 3]  stacked rest-pose offsets
          root_bone_ids: list[int]  bones with parent_id == None
        """
        J = len(self.bones)
        depths = [0] * J
        for b in self.bones:
            if b.parent_id is not None:
                depths[b.id] = depths[b.parent_id] + 1
        max_depth = max(depths) if depths else 0

        self.depth_levels: List[tuple] = []  # [(bone_ids_tensor, parent_ids_tensor)]
        self.root_bone_ids: List[int] = []

        for d in range(max_depth + 1):
            bone_ids = [b.id for b in self.bones if depths[b.id] == d]
            if d == 0:
                self.root_bone_ids = bone_ids
                # Roots have no parent — store (-1) as placeholder
                parent_ids = [-1] * len(bone_ids)
            else:
                parent_ids = [self.bones[bid].parent_id for bid in bone_ids]
            self.depth_levels.append((
                torch.tensor(bone_ids, dtype=torch.long, device=device),
                torch.tensor(parent_ids, dtype=torch.long, device=device),
            ))

        # Stacked rest offsets [J, 3]
        self.rest_offsets = torch.stack(
            [b.rest_offset for b in self.bones], dim=0
        ).to(device)  # [J, 3]

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def num_joints(self) -> int:
        return len(self.bones)

    def bone_by_name(self, name: str) -> Bone:
        return self.bones[self._name_to_id[name]]

    def topological_order(self) -> List[Bone]:
        """Bones already stored in topological order; return as-is."""
        return self.bones

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict, device: torch.device = torch.device("cpu")) -> "Skeleton":
        """Construct Skeleton from a plain-dict representation.

        Expected keys per bone entry (list under 'bones'):
            id, name, parent_id, rest_offset [3], rest_rotation [4]
        Optional top-level:
            bind_matrices [J,4,4], joint_limits [J,3,2]
        """
        bones = []
        for bd in data["bones"]:
            bones.append(Bone(
                id=bd["id"],
                name=bd["name"],
                parent_id=bd.get("parent_id"),
                rest_offset=torch.tensor(bd["rest_offset"], dtype=torch.float32, device=device),
                rest_rotation=torch.tensor(bd["rest_rotation"], dtype=torch.float32, device=device),
            ))

        bind_matrices = None
        if "bind_matrices" in data:
            bind_matrices = torch.tensor(data["bind_matrices"], dtype=torch.float32, device=device)

        joint_limits = None
        if "joint_limits" in data:
            joint_limits = torch.tensor(data["joint_limits"], dtype=torch.float32, device=device)

        return cls(bones, bind_matrices, joint_limits)

    @classmethod
    def make_rest_skeleton(cls, num_joints: int = 1,
                           device: torch.device = torch.device("cpu")) -> "Skeleton":
        """Create a trivial single-bone skeleton for testing."""
        bones = [Bone(
            id=0, name="root", parent_id=None,
            rest_offset=torch.zeros(3, device=device),
            rest_rotation=torch.tensor([1., 0., 0., 0.], device=device),
        )]
        for i in range(1, num_joints):
            bones.append(Bone(
                id=i, name=f"bone_{i}", parent_id=i - 1,
                rest_offset=torch.tensor([0., 1., 0.], device=device),
                rest_rotation=torch.tensor([1., 0., 0., 0.], device=device),
            ))
        return cls(bones)
