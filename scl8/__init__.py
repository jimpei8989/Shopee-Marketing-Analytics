from argparse import ArgumentParser
from datetime import datetime

from omegaconf import OmegaConf

from .train import train

def main():
    args = parse_arguments()

    if args.mode == 'hello':
        print('hello')
    elif args.mode == 'train':
        cfg = OmegaConf.load(args.config)
        train(cfg, args.datadir, args.model_path, args.prediction_path)
    else:
        raise ValueError('Invalid mode passed by comandline arguments')


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('mode', type=str)
    parser.add_argument('--config', type=str, default='configs/xgboost.yaml')
    parser.add_argument('--datadir', type=str, default='data/')
    parser.add_argument('--model_path', type=str, default='models/baseline.pkl')
    parser.add_argument('--prediction_path', type=str, default=f'submissions/output-{datetime.now().strftime("%m%d-%H%M")}.csv')
    return parser.parse_args()
