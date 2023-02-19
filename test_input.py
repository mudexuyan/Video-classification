import argparse

parser = argparse.ArgumentParser()
parser.add_argument('h', type=str, help='display an integer')
args = parser.parse_args()

print(args.h)