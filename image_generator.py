import datetime
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torchvision.utils as tvu
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel

from configs import parse_option
from models import build_ddpm
from utils import compute_alpha, create_logger, get_named_beta_schedule, get_xt_next

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
except ImportError:
    amp = None


def main(local_rank):
    # Load hyperparameter
    _, config = parse_option()

    # Check whether user is installed the amp or not
    if config.amp_opt_level != "O0":
        assert amp is not None, "amp not installed!"
    Path(config.output).mkdir(parents=True, exist_ok=True)

    # Multi-node communication
    ip = "127.0.0.1"
    port = "29510"
    hosts = 1  # number of node
    rank = 0  # rank of current node
    gpus = torch.cuda.device_count()  # Number of GPUs per node

    # World_size is the number of global GPU, rank is the global index of current GPU
    dist.init_process_group(backend='nccl', init_method=f'tcp://{ip}:{port}', world_size=hosts * gpus,
                            rank=rank * gpus + local_rank)
    torch.cuda.set_device(local_rank)

    config.defrost()
    config.local_rank = local_rank
    config.freeze()

    # Fix the random seed
    seed = int(config.seed + local_rank * 50000)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Create the logger
    logger_name = f"dic_diff"
    logger = create_logger(output_dir=config.output, dist_rank=local_rank, name=logger_name)

    if dist.get_rank() == 0:
        path = Path(config.output).joinpath("config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(config)
        logger.info(f"Full config saved to {path}")

    # The length of how many sample is requested to be generated. Defined as 50000 for FID
    generate_sample_len = 50000 if config.max_length is None else config.max_length

    # Build the ddpm model
    ddpm_model = build_ddpm(config, logger)
    ddpm_model.cuda()
    if config.amp_opt_level != "O0":
        ddpm_model = amp.initialize(ddpm_model, opt_level=config.amp_opt_level)
        ddpm_model = ApexDDP(ddpm_model, delay_allreduce=True)
    else:
        ddpm_model = DistributedDataParallel(ddpm_model,
                                             device_ids=[local_rank],
                                             broadcast_buffers=False)
    ddpm_model.eval()

    timesteps = config.dm.num_diffusion_timesteps

    betas = torch.from_numpy(get_named_beta_schedule(schedule_name=config.dm.schedule_name,
                                                     num_diffusion_timesteps=timesteps,
                                                     beta_start=config.dm.beta_start,
                                                     beta_end=config.dm.beta_end)).float()

    # Hyperparameter for the DDPM/DDIM
    betas = betas.cuda(non_blocking=True)
    eta = config.dm.eta
    skip = int(config.dm.num_diffusion_timesteps / config.dm.sample_timesteps)

    if config.dm.skip_type == "uniform":
        seq = range(0, timesteps, skip)
    elif config.dm.skip_type == "quad":
        seq = (
                np.linspace(
                        0, np.sqrt(timesteps * 0.8), config.dm.sample_timesteps
                )
                ** 2
        )
        seq = [int(s) for s in list(seq)]
    else:
        raise NotImplementedError

    # Hyperparameter for the adaptive momentum sampler
    if config.beta1:
        # Beta1 coefficient to adjust momentum mechanism
        lam_min = config.beta1[0]
        lam_max = config.beta1[1]
    else:
        lam_min = 0.0
        lam_max = 0.0

    if config.beta2:
        # Beta2 coefficient to adjust adaptive updating pace
        beta2 = config.beta2
    else:
        beta2 = 1.0

    # Schedule the beta1
    a = np.pi / 2
    if config.ascending:
        lam = np.linspace(start=lam_min, stop=lam_max, num=int(timesteps / skip))
    else:
        lam = np.linspace(start=lam_max, stop=lam_min, num=int(timesteps / skip))

    B = config.model.batch_size
    num = 0  # Number of generated image

    # the output path
    out_dir_path = Path(config.output)
    out_path = out_dir_path.joinpath("fake_imgs")
    out_path.mkdir(parents=True, exist_ok=True)

    files = os.listdir(out_path)

    # Count the saved image number
    saved_num = 0
    for file in files:
        if "png" in file.__str__():
            saved_num += 1

    while num < generate_sample_len:
        s_time = time.time()
        x = torch.randn(B,
                        config.data.channels,
                        config.data.img_size,
                        config.data.img_size).cuda()
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xt = torch.clone(x).to(x.device)
        b = betas

        m = torch.zeros_like(x).cuda()  # Average of momentum
        v = torch.ones_like(x).cuda()  # Average of second-order moments

        with torch.no_grad():
            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                noise = torch.randn_like(x)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())

                # jump the saved image
                if num < saved_num:
                    continue
                if config.amp_opt_level != "O0":
                    et = ddpm_model(xt, t.half())
                else:
                    et = ddpm_model(xt, t.float())

                if eta != 2.0:
                    # fix-small in DDPM or DDIM
                    c1 = (eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt())
                    c2 = ((1 - at_next) - c1 ** 2).sqrt()
                    p1 = (c2 / at_next.sqrt() - (1 - at).sqrt() / at.sqrt())
                    p2 = c1 / at_next.sqrt()
                else:
                    # fix-large in DDPM
                    beta_t = 1 - at / at_next
                    mask = 1 - (t == 0).float()
                    mask = mask.view(-1, 1, 1, 1)
                    logvar = beta_t.log()
                    c1 = mask * torch.exp(0.5 * logvar)
                    p1 = -(1.0 / at - 1).sqrt() * beta_t / (1.0 - at)
                    p2 = c1 / at_next.sqrt()

                dx = p1 * et + p2 * noise

                # Adaptive pace
                if config.method == "rms" or config.method == "adam":
                    norm = torch.nn.functional.normalize(dx, p=2.0)
                    v = beta2 * v + (1 - beta2) * torch.pow(norm, 2)
                    v = torch.sqrt(v) + 1e-8

                # Momentum
                if config.method == "momentum" or config.method == "adam":
                    lamb = lam[int((timesteps - 1 - i) / skip)]
                    if config.b1_m == 1:
                        m = np.sin(lamb * a) * m + np.cos(lamb * a) * dx
                    elif config.b1_m == 2:
                        m = np.power(np.sin(lamb * a), 2) * m + np.power(np.cos(lamb * a), 2) * dx
                    else:
                        raise f"Do not support that"

                    dx = m

                xt_next = get_xt_next(at, at_next, xt, dx, v)
                xt = torch.clone(xt_next).to(x.device)
                torch.cuda.synchronize()

        # jump the saved image
        if num < saved_num:
            num += min(B * gpus, saved_num)
            continue

        # save images
        imgs = xt.detach().cpu()
        imgs = (imgs + 1.0) / 2.0
        for i in range(imgs.shape[0]):
            img_path = out_path.joinpath(f"{config.method}_"
                                         f"{num + i + B * int(config.local_rank)}.png")
            tvu.save_image(imgs[i,], img_path)
        num += B * gpus
        torch.cuda.synchronize()
        end_time = time.time() - s_time
        logger.info(f"the number of saved image: {num}\t."
                    f'It takes {datetime.timedelta(seconds=int(end_time))}.')

    time.sleep(2)

    logger.info("Done!")


if __name__ == '__main__':
    ngpus = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(), nprocs=ngpus)
