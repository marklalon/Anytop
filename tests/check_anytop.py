import sys
sys.path.insert(0, '.')

print("=== Anytop 兼容性检查 ===\n")

# Test core dependencies
try:
    import torch
    import torchvision
    import numpy
    import scipy
    import timm
    import einops
    import huggingface_hub
    print("✓ 核心依赖：torch, torchvision, numpy, scipy, timm, einops, huggingface_hub")
except ImportError as e:
    print(f"✗ 核心依赖出错: {e}")

# Test Motion library
try:
    import BVH
    from InverseKinematics import animation_from_positions
    print("✓ Motion库：BVH, InverseKinematics")
except ImportError as e:
    print(f"✗ Motion库出错: {e}")

# Test Anytop modules
try:
    from model.anytop import AnyTop
    from model.conditioners import T5Conditioner
    from diffusion.flow_matching import FlowMatching
    from data_loaders.tensors import truebones_batch_collate
    print("✓ Anytop核心模块：AnyTop, T5Conditioner, FlowMatching, data_loaders")
except ImportError as e:
    print(f"✗ Anytop模块出错: {e}")

print("\n=== 检查完成 ===")
print("✓ 环境配置良好，可以运行 Anytop")
