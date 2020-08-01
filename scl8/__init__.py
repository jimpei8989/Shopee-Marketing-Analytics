from argparse import ArgumentParser

from .train import train

def main():
    args = parse_arguments()

    if args.mode == 'hello':
        print('hello')
    elif args.mode == 'train':
        train(args.datadir, args.output_file)
    else:
        raise ValueError('Invalid mode passed by comandline arguments')


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('mode', type=str)
    parser.add_argument('--datadir', type=str, default='data/')
    parser.add_argument('--output_file', type=str, default='submissions/output.csv')
    return parser.parse_args()
