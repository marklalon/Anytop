import pytest
import torch


@pytest.fixture
def full_precision_matmul():
    # Training may enable TF32 in an earlier test. Serial and batched GEMMs
    # have different shapes, so strict FP32 equivalence needs TF32 disabled.
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('highest')
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)
