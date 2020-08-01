from argparse import ArgumentParser

def main():
    args = parse_arguments()

    if args.mode == 'hello':
        print('hello')
    else:
        raise ValueError('Invalid mode passed by comandline arguments')


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('mode', type=str)
    return parser.parse_args()
