import logging
import numpy as np
import random
import torch

log = logging.getLogger(__name__)

def seed_everything(seed) -> None:
    if not seed:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

