from argparse import ArgumentParser
from datetime import datetime

from omegaconf import OmegaConf

from .train import train
from .ensemble import ensemble
from .utils import seed_everything

def main():
    args = parse_arguments()

    if args.mode == 'hello':
        print('hello')
    elif args.mode == 'train':
        cfg = OmegaConf.load(args.config)
        train(cfg, **cfg.misc)
    elif args.mode == 'ensemble':
        cfg = OmegaConf.load(args.config)
        ensemble(cfg, **cfg.misc)
    else:
        raise ValueError('Invalid mode passed by comandline arguments')

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('mode', type=str)
    parser.add_argument('--config', type=str, default='configs/xgboost.yaml')
    return parser.parse_args()
