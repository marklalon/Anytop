import numpy as np
import torch
import random


def fixseed(seed, cudnn_benchmark=True, allow_tf32=True):
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# SEED = 10
# EVALSEED = 0
# # Provoc warning: not fully functionnal yet
# # torch.set_deterministic(True)
# torch.backends.cudnn.benchmark = False
# fixseed(SEED)
