"""Host -> device copies that do not stall the training step."""
import torch


def host_to_device(value, device, dtype=None):
    """``torch.as_tensor(value, dtype=dtype, device=device)`` without a stream sync.

    A blocking host->CUDA copy (``as_tensor(..., device=cuda)``, ``.to(cuda)``)
    synchronizes the current stream: the host waits for every queued kernel
    before it may launch the next one. In a kernel-launch-bound step that wait
    is the host's lost time. Staging through pinned memory lets the copy be
    queued like any other kernel; the caching host allocator keeps the staging
    block alive until the copy has run.
    """
    device = torch.device(device)
    if torch.is_tensor(value) and value.device.type != 'cpu':
        return value.to(device=device, dtype=dtype)
    tensor = torch.as_tensor(value, dtype=dtype)
    if device.type != 'cuda':
        return tensor.to(device)
    if not tensor.is_pinned():
        tensor = tensor.pin_memory()
    return tensor.to(device, non_blocking=True)
