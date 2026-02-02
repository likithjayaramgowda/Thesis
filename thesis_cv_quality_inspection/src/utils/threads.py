import os
import torch


def set_thread_count(n: int) -> None:
    if n and n > 0:
        os.environ["OMP_NUM_THREADS"] = str(n)
        os.environ["MKL_NUM_THREADS"] = str(n)
        torch.set_num_threads(n)
    else:
        # use default
        pass
