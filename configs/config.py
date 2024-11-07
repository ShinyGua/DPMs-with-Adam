import argparse
import os
from pathlib import Path

import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.base = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.data = CN()
# Path to dataset base
_C.data.data_root_path = "datasets"
# Dataset type
_C.data.dataset_name = "CIFAR10"
# number of image channels
_C.data.channels = 3
# Image patch size (default: 32)
_C.data.img_size = 32
# Mean pixel value of dataset
_C.data.mean = (0.5, 0.5, 0.5)
# Std deviation of dataset
_C.data.std = (0.5, 0.5, 0.5)
_C.data.length = 50000

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.model = CN()
# Denoising Diffusion Probabilistic Models from https://arxiv.org/abs/2006.11239
_C.model.ddpm = CN()
_C.model.ddpm.type = "simple"
_C.model.ddpm.in_channels = 3
_C.model.ddpm.out_ch = 3
_C.model.ddpm.ch = 128
_C.model.ddpm.ch_mult = [1, 2, 2, 2]
_C.model.ddpm.num_res_blocks = 2
_C.model.ddpm.attn_resolutions = [16, ]
_C.model.ddpm.dropout = 0.1
_C.model.ddpm.var_type = "fixedlarge"
_C.model.ddpm.ema_rate = 0.9999
_C.model.ddpm.ema = True
_C.model.ddpm.resamp_with_conv = True
# DDPM pre-trained model
_C.model.ddpm.initial_checkpoint = ""
# Input batch size for image generator (default: 128)
_C.model.batch_size = 128
# Validation batch size override (default: None)
_C.model.validation_batch_size = None

# -----------------------------------------------------------------------------
# diffusion model settings
# -----------------------------------------------------------------------------
_C.dm = CN()
# schedule name of diffusion model ("linear" or "cosine")
_C.dm.schedule_name = "linear"
# number of steps involved (default: 1000)
_C.dm.num_diffusion_timesteps = 1000
# eta used to control the variances of sigma
_C.dm.eta = 1.0
_C.dm.sample_timesteps = 1000
_C.dm.beta_start = 1000
_C.dm.skip_type = "uniform"
_C.dm.beta_start = 0.0001
_C.dm.beta_end = 0.02

# random seed (default: 42)
_C.seed = 42
# Frequency to logging info
_C.log_freq = 10
# Frequency to save checkpoint
_C.save_freq = 5
# how many training processes to use
_C.workers = 8
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.pin_mem = False
# path to output folder (default: /results)
_C.output = "results"
# local rank for DistributedDataParallel, given by command line argument
_C.local_rank = -1
# Tag of experiment, overwritten by command line argument
_C.tag = 'test'
# Auto resume from latest checkpoint
_C.auto_resume = True
_C.amp_opt_level = 'O1'
_C.device = "cuda:0"

# Hyper-parameters for momentum diffusion model sampling
_C.beta1 = None
_C.beta2 = None
_C.b1_m = 2
_C.method = "momentum"
_C.ascending = True

#
_C.max_length = None


def _update_config_from_file(config, cfg_file):
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                    config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)


def parse_option():
    parser = argparse.ArgumentParser('Momentum diffusion model sampling', add_help=False)

    parser.add_argument('--cfg', type=str, default="configs/CIFAR10.yaml",
                        metavar="FILE", help='path to config file', )
    parser.add_argument(
            "--opts",
            help="Modify config options by adding 'KEY VALUE' pairs. ",
            default=None,
            nargs='+',
    )
    parser.add_argument('--method', '-adam', default="momentum",
                        choices=['o', "momentum", "rms", "adam"],
                        type=str, help="method for adjusting diffusion sampling")
    parser.add_argument('--batch-size', '-b', type=int, help="batch size for single GPU")
    parser.add_argument('--checkpoint-path', type=str, help='path to pre-trained model')
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--num-workers', default=8, type=int, help='number of worker for dataloader')
    parser.add_argument('--tag', '-t', default="AM", help='tag of experiment')
    parser.add_argument('--seed', default=96, type=int, help='random seed')

    parser.add_argument('--eta', default=1.0, type=float, help='ddim of ddpm')
    parser.add_argument('--st', type=int, help='number of sample timesteps')

    parser.add_argument('--b1-min', default=0.0, type=float, help='beta min for momentum')
    parser.add_argument('--b1-max', default=0.1, type=float, help='beta max for momentum')
    parser.add_argument('--b2', default=0.999, type=float, help='beta for RMSProp')
    parser.add_argument('--b1-m', default=2, type=int, help='beta for momentum')

    parser.add_argument('--device', type=str,
                        help='Device to use. Like cuda, cuda:0 or cpu')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return args, config


def update_config(config, args):
    config.defrost()
    if args.cfg:
        _update_config_from_file(config, args.cfg)

    if getattr(args, 'opts', None) is not None:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if getattr(args, 'method', None) is not None:
        config.method = args.method
    if getattr(args, 'batch_size', None) is not None:
        config.model.batch_size = args.batch_size
    if getattr(args, 'checkpoint_path', None) is not None:
        config.model.ddpm.initial_checkpoint = args.checkpoint_path
    if getattr(args, 'num_workers', None) is not None:
        config.workers = args.num_workers
    if getattr(args, 'tag', None) is not None:
        config.tag = args.tag
    if getattr(args, 'amp_opt_level', None) is not None:
        config.amp_opt_level = args.amp_opt_level
    if getattr(args, 'seed', None) is not None:
        config.seed = args.seed
    if getattr(args, 'device', None) is not None:
        config.device = args.device

    if getattr(args, 'eta', None) is not None:
        config.dm.eta = args.eta
    if getattr(args, 'st', None) is not None:
        config.dm.sample_timesteps = args.st

    if getattr(args, 'b1_min', None) is not None and getattr(args, 'b1_max', None):
        config.beta1 = [args.b1_min, args.b1_max]
    if getattr(args, 'b1_m', None) is not None:
        config.b1_m = args.b1_m
    if getattr(args, 'b2', None) is not None:
        config.beta2 = args.b2
    if getattr(args, 'dims', None) is not None:
        config.fid.dims = args.dims

    eta_map = {0.0: "ddim", 1.0: "ddpm", 2.0: "ddpm_larger"}

    dir_name = f"{config.tag}_seed_{config.seed}_st_{config.dm.sample_timesteps}_{eta_map[config.dm.eta]}"
    if config.method:
        dir_name += f"_{config.method}"
    if config.method != 'o' and config.method != 'rms' and config.beta1:
        dir_name += f"_b1_{config.beta1[0]}to{config.beta1[1]}_sum_{config.b1_m}"
    if config.method != 'o' and config.method != 'momentum' and config.beta2:
        dir_name += f"_b2_{config.beta2}"

    if config.data.dataset_name == "CIFAR10":
        config.data.length = 50000
    elif config.data.dataset_name == "CHURCH":
        config.data.length = 126227

    config.output = Path(config.output).joinpath(config.data.dataset_name).joinpath(dir_name).__str__()
    config.data.data_root_path = Path(config.data.data_root_path).joinpath(config.data.dataset_name).__str__()

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)
    return config
