import numpy as np
import torch
import random


def fixseed(seed):
    # Seeds only. TF32 is a precision policy, not a seed: generate.py reseeds before
    # every batch, so forcing it off here silently undid fp32 inference's TF32 and
    # made each caller's matmul precision depend on call order. The bit-exact
    # verification setup lives in utils/numerical_verification.py.
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
