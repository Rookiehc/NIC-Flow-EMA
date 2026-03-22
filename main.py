#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2023-10-26 20:20:36

import warnings
warnings.filterwarnings("ignore")

# Ensure local 'src' package is importable before site-packages (for bundled diffusers, etc.)
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.exists():
        # Prepend to sys.path so 'import diffusers' resolves to ./src/diffusers first
        sys.path.insert(0, str(_SRC))

import argparse
from omegaconf import OmegaConf

from utils.util_common import get_obj_from_str
from utils.util_opts import str2bool
import importlib

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
            "--save_dir",
            type=str,
            default="./save_dir",
            help="Folder to save the checkpoints and training log",
            )
    parser.add_argument(
            "--resume",
            type=str,
            const=True,
            default="",
            nargs="?",
            help="resume from the save_dir or checkpoint",
            )
    parser.add_argument(
            "--cfg_path",
            type=str,
            default="./configs/sd-turbo-sr-ldis.yaml",
            help="Configs of yaml file",
            )
    parser.add_argument(
            "--ldif",
            type=float,
            default=None,
            help="Loss coefficient for diffsuion in latent space",
            )
    parser.add_argument(
            "--llpips",
            type=float,
            default=None,
            help="Loss coefficient for latent lpips",
            )
    parser.add_argument(
            "--ldis",
            type=float,
            default=None,
            help="Loss coefficient for latent discriminator",
            )
    parser.add_argument(
            "--use_text",
            type=str2bool,
            default=None,
            help="Text Prompt",
            )
    parser.add_argument(
            "--objective",
            type=str,
            default='default',
            choices=['default','imm'],
            help="Training objective: default or imm",
            )
    args = parser.parse_args()

    return args

if __name__ == "__main__":
        args = get_parser()

        configs = OmegaConf.load(args.cfg_path)
        # Ensure a top-level `train` and `train.loss_coef` exist for backward compatibility
        if not hasattr(configs, 'train') or configs.train is None:
                configs.train = OmegaConf.create({})
        if not hasattr(configs.train, 'loss_coef') or configs.train.loss_coef is None:
                configs.train.loss_coef = OmegaConf.create({})

        # Apply CLI overrides safely
        if args.ldif is not None:
                configs.train.loss_coef.ldif = args.ldif
        if args.ldis is not None:
                configs.train.loss_coef.ldis = args.ldis
        if args.llpips is not None:
                configs.train.loss_coef.llpips = args.llpips
        if args.use_text is not None:
                configs.train.use_text = args.use_text

        # merge args to config
        for key in vars(args):
                if key in ['cfg_path', 'save_dir', 'resume', ]:
                        configs[key] = getattr(args, key)

        # If IMM objective selected, swap trainer target to TrainerIMM when available
        if args.objective == 'imm':
                # prefer configs.trainer.target if already set to TrainerIMM
                if hasattr(configs, 'trainer') and hasattr(configs.trainer, 'target'):
                        target = configs.trainer.target
                        if 'TrainerIMM' not in str(target):
                                configs.trainer.target = 'trainer_imm.TrainerIMM'
                else:
                        configs.trainer = OmegaConf.create({'target': 'trainer_imm.TrainerIMM'})

        # Increase NCCL/Distributed timeout for slow validation steps or initial compilation
        import datetime
        import torch.distributed as dist
        
        # We need to set timeout via init_process_group kwargs if it was called here.
        # But 'trainer_imm' usually handles init inside __init__ using 'env://'.
        # However, we can warn or try to set env vars before Trainer init if Pytorch supports it.
        # Setting NCCL_TIMEOUT/GLOO_TIMEOUT env vars might help before process group init.
        import os
        os.environ["NCCL_BLOCKING_WAIT"] = "1"
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
        # Increase default timeout significantly (e.g. 60min) to avoid crash during long validation
        # Only effective if init_process_group is called AFTER this.
        # trainer_imm.py calls init_process_group inside __init__.

        trainer_cls = get_obj_from_str(configs.trainer.target)
        trainer = trainer_cls(configs)
        trainer.train()
