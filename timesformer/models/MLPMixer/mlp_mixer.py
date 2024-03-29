from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce
import torch
pair = lambda x: x if isinstance(x, tuple) else (x, x)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )

def MLPMixer(*, image_size, channels, num_patches, dim, depth=1, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
    image_h, image_w = pair(image_size)
    # assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    # num_patches = (image_h // patch_size) * (image_w // patch_size)
    num_patches = num_patches
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear
    # feed1 = FeedForward(num_patches, expansion_factor, dropout, chan_first)
    # feed2 = FeedForward(dim, expansion_factor_token, dropout, chan_last)
    # pre1 = PreNormResidual(dim,feed1) 
    # pre2 = PreNormResidual(dim,feed2)
    # list = torch.split(x,num_patches,dim=1)
    # res = []
    # for i in list:
    #     tmp = pre1(i);
    #     tmp = pre2(tmp)
    #     res.append(tmp)
    # return torch.cat(res,dim=1); 

    return nn.Sequential(
        # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        # nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
        ) for _ in range(depth)],
        # nn.LayerNorm(dim),
        # Reduce('b n c -> b c', 'mean'),
        # nn.Linear(dim, num_classes)
    )
