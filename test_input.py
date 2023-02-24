# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('h', type=str, help='display an integer')
# args = parser.parse_args()

# print(args.h)
import torch
a = torch.randn(2, 3, 4, 5)
b = a.transpose(-1, -2).softmax(dim=-1)
c = a.transpose(-2, -1)
print(b.shape)
print(c.shape)