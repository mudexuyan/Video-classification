import torch
import torch.nn as nn
from functools import partial
import math
import warnings
import torch.nn.functional as F
import numpy as np


from torch import einsum
from einops import rearrange, reduce, repeat



def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
# a = [[1,2,3],[4,5,6],[7,8,9]]
# a = torch.tensor(a)
# print(a.shape)
# a = rearrange(a, 'm n -> (m n)',m=3,n=3)
# print(a.shape)
# print(a)
window_size = 7
C = 3
x = torch.ones(2,14,14,C)
print(x.shape)


# [8, 7, 7, 3]
x_windows = window_partition(x, window_size)

print(x_windows.shape)

#[8, 49, 3]
x_windows = x_windows.view(-1, window_size * window_size, C)

print(x_windows.shape)

# attn_windows = self.attn(x_windows)
# [8, 7, 7, 3]
attn_windows = x_windows.view(-1,
                                window_size,
                                window_size,
                                C)
print(attn_windows.shape)
# [2, 14, 14, 3]
x = window_reverse(attn_windows, window_size, 14, 14)
print(x.shape)

