import numpy as np
import torch
import torch.distributed as dist


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

@torch.no_grad()
def momentum(m, dx, lamb, b1_m=2):
    a = np.pi / 2
    if b1_m == 1:
        m = np.sin(lamb * a) * m + dx
    elif b1_m == 2:
        m = np.sin(lamb * a) * m + np.cos(lamb * a) * dx
    elif b1_m == 3:
        m = np.power(np.sin(lamb * a), 2) * m + np.power(np.cos(lamb * a), 2) * dx
    else:
        raise f"Do not support that"
    return m
