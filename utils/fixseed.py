import numpy as np
import torch
import random


def fixseed(seed):
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# SEED = 10
# EVALSEED = 0
# # Provoc warning: not fully functionnal yet
# # torch.set_deterministic(True)
# torch.backends.cudnn.benchmark = False
# fixseed(SEED)
