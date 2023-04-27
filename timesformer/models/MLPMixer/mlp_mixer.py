from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange
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

def MLPMixer(*, image_size, channels, num_patches, dim, depth=4, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
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

def MLPMixer_channel(*, image_size, channels, num_patches, dim, depth=4, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):

    num_patches = num_patches
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    # return nn.Sequential(
    #     PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
    #     PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
    # )
    return nn.Sequential(
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
        ) for _ in range(depth)],
    )

class MixerBase(nn.Module):
    def __init__(self,T,N,dim,dropout):
        super().__init__()
        self.T = T
        self.N = N
        self.dim = dim
        self.norm_s = nn.LayerNorm(N)
        self.norm_t = nn.LayerNorm(T)
        self.norm_c = nn.LayerNorm(dim)
        self.expansion_factor_s = 4
        self.fn_s = FeedForward(N,self.expansion_factor_s,dropout)
        self.expansion_factor_t = 2
        self.fn_t = FeedForward(T,self.expansion_factor_t,dropout)
        self.expansion_factor_c = 0.5
        self.fn_c = FeedForward(dim,self.expansion_factor_c,dropout)

    def forward(self, x):
        # x: (b t) h/p*w/p embed_dim
        # space 

        # x = x.transpose(1,2)
        # x_s = self.fn_s(self.norm_s(x)) + x
        # x_s = x_s.transpose(1,2)
        # # time
        # x_t = rearrange(x_s, '(b t) n d -> (b n) d t',n=self.N,t=self.T,d=self.dim)
        # x_t = self.fn_t(self.norm_t(x_t)) + x_t
        # x_t = rearrange(x_t, '(b n) d t -> (b t) n d',n=self.N,t=self.T,d=self.dim)
        # # channel
        # return self.fn_c(self.norm_c(x_t)) + x_t

        # 并行结构
        x_s = x.transpose(1,2)
        x_s = self.fn_s(self.norm_s(x_s)) + x_s
        x_s = rearrange(x_s, '(b t) d n -> b t n d ',n=self.N,t=self.T,d=self.dim)
        # time
        x_t = rearrange(x, '(b t) n d -> (b n) d t',n=self.N,t=self.T,d=self.dim)
        x_t = self.fn_t(self.norm_t(x_t)) + x_t
        x_t = rearrange(x_t, '(b n) d t -> b t n d',n=self.N,t=self.T,d=self.dim)
        # channel
        x_c = self.fn_c(self.norm_c(x)) + x
        x_c = rearrange(x_c, '(b t) n d -> b t n d ',n=self.N,t=self.T,d=self.dim)
        res = x_s + x_t + x_c
        res = rearrange(res, 'b t n d -> (b t) n d ',n=self.N,t=self.T,d=self.dim)
        return res


def Mixer(*, T, num_patches, dim, depth=4, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
 
    return nn.Sequential(
        *[nn.Sequential(
            MixerBase(T, num_patches, dim, dropout)
        ) for _ in range(depth)],
    )


class MixerEncoderBase(nn.Module):
    def __init__(self,T,N,dim,dropout,type):
        super().__init__()
        self.type = type
        self.T = T
        self.N = N
        self.dim = dim
        # print(N)
        if self.type == 'space':
            self.norm_s = nn.LayerNorm(N)
            self.expansion_factor_s = 4
            self.fn_s = FeedForward(N,self.expansion_factor_s,dropout)
        if self.type == 'time':
            self.norm_t = nn.LayerNorm(T)
            self.expansion_factor_t = 2
            self.fn_t = FeedForward(T,self.expansion_factor_t,dropout)
        if self.type == 'channel': 
            self.norm_c = nn.LayerNorm(dim)
            self.expansion_factor_c = 0.5
            self.fn_c = FeedForward(dim,self.expansion_factor_c,dropout)

    def forward(self, x):
        # x: (b t) h/p*w/p embed_dim
        if self.type == 'space':
            x_s = x.transpose(1,2)
            x_s = self.fn_s(self.norm_s(x_s)) + x_s
            res = rearrange(x_s, '(b t) d n -> (b t) n d ',n=self.N,t=self.T,d=self.dim)
        if self.type == 'time':
            x_t = rearrange(x, '(b t) n d -> (b n) d t',n=self.N,t=self.T,d=self.dim)
            x_t = self.fn_t(self.norm_t(x_t)) + x_t
            res = rearrange(x_t, '(b n) d t -> (b t) n d',n=self.N,t=self.T,d=self.dim)
        if self.type == 'channel': 
            res = self.fn_c(self.norm_c(x)) + x
            # res = rearrange(x_c, '(b t) n d -> b t n d ',n=self.N,t=self.T,d=self.dim)
        return res


def MixerEncoder(*, T, num_patches, dim, depth=4, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
 
    return nn.Sequential(
        *[nn.Sequential(
            MixerEncoderBase(T, num_patches, dim, dropout,'space'),
            MixerEncoderBase(T, num_patches, dim, dropout,'channel'),
            MixerEncoderBase(T, num_patches, dim, dropout,'time')
        ) for _ in range(depth)],
    )