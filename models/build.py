import torch

from models.DDPM import DdpmModel


def build_ddpm(config, logger):
    model = DdpmModel(ch=config.model.ddpm.ch,
                      in_channels=config.model.ddpm.in_channels,
                      out_ch=config.model.ddpm.out_ch,
                      ch_mult=tuple(config.model.ddpm.ch_mult),
                      num_res_blocks=config.model.ddpm.num_res_blocks,
                      attn_resolutions=config.model.ddpm.attn_resolutions,
                      dropout=config.model.ddpm.dropout,
                      resamp_with_conv=config.model.ddpm.resamp_with_conv,
                      model_type=config.model.ddpm.var_type,
                      img_size=config.data.img_size,
                      num_timesteps=config.dm.num_diffusion_timesteps)

    if config.model.ddpm.initial_checkpoint != "":
        logger.info(f"======> Loading pre-trained model {config.model.ddpm.initial_checkpoint}")
        states = torch.load(config.model.ddpm.initial_checkpoint)
        msg = model.load_state_dict(states)
        logger.info(msg)
        logger.info(f"======> Success loading pre-trained model {config.model.ddpm.initial_checkpoint}")

    return model