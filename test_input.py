# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('h', type=str, help='display an integer')
# args = parser.parse_args()

# print(args.h)
import torch
# a = torch.randn(2, 3, 4, 5)
# b = a.transpose(-1, -2).softmax(dim=-1)
# c = a.transpose(-2, -1)
# print(b.shape)
# print(c.shape)
a = torch.randn(2, 3)
b = torch.randn(2, 3)
c = torch.randn(2, 3)
res = torch.cat([a,b,c], dim=0) # bt hw m*3
print(res.shape)
