"""
Hacked together from the b-cos and timm library.
https://github.com/moboehle/B-cos
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

#-----> Modules


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class NormedLinear(nn.Linear):
    """
    Standard linear layer, but with unit norm weights.
    """
    def forward(self, in_tensor):
        shape = self.weight.shape
        w = self.weight.view(shape[0], -1)
        w = w/(w.norm(p=2, dim=1, keepdim=True))
        return F.linear(in_tensor, w)


class BcosLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, max_out=2, b=2, 
                  scale=None, scale_fact=1, **kwargs):
        super().__init__()
        self.b = b
        self.max_out = max_out
        self.detach = False
        if scale is None:
            self.scale = np.sqrt(in_features) / scale_fact
        else:
            self.scale = scale
        self.linear = NormedLinear(in_features, out_features * max_out, bias)

    def explanation_mode(self, detach=True):
        """
        Enters 'explanation mode' by setting self.explain and self.detach.
        
        :param detach: whether to detach the weight tensor form the current graph
        :return: None
        """
        self.detach = detach

    def forward(self, in_tensor):
        """
        For B=2, we do not have calculate the cosine term.

        :param in_tensor: input tensor of shape (b, n, n)
        :return: BcosLinear transformation of input tensor
        """
        if self.b == 2:

            return self.fwd_2(in_tensor)
        else: 
            raise NotImplementedError

    def fwd_2(self, in_tensor):
        out = self.linear(in_tensor)
        norm = in_tensor.norm(p=2, dim=2, keepdim=True)

        if self.max_out > 1:
            bs, n, c = out.shape
            out = out.view(bs, n, self.max_out, -1)
            out = out.max(dim=2, keepdim=False)[0]

        if self.detach:
            out = (out * out.abs().detach())
            norm = norm.detach()
        else:
            out = (out * out.abs())

        return out / (norm * self.scale)
    

class BcosAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = BcosLinear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = BcosLinear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        self.save_attention_map(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map


class FeedForward(nn.Module):
    def __init__(self, dim=512, hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class BcosMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = BcosLinear(in_features, hidden_features)
        self.fc2 = BcosLinear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class BcosTransformerLayer(nn.Module):
    def __init__(self, dim=512, heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = BcosAttention(dim=dim, num_heads=heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ff = BcosMlp(in_features=dim, hidden_features=dim, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.ff(self.norm(x)))
        return x


class BcosTransformer(nn.Module):
    def __init__(self, num_classes, input_dim=2048):
        super().__init__()
        self.n_classes = num_classes

        self._fc1 = nn.Sequential(nn.Linear(input_dim, 512, bias=True), nn.ReLU())
        self.layer1 = BcosTransformerLayer(dim=512, heads=8)
        self.layer2 = BcosTransformerLayer(dim=512, heads=8)
        self._fc2 = nn.Linear(512, self.n_classes, bias=True)

    def forward(self, x, return_emb=False, register_hook=False):

        h = x
        h = self._fc1(h)
        h = self.layer1(h)
        h = self.layer2(h)
        h = h.mean(dim=1)
        logits = self._fc2(h)

        if return_emb:
            emb = torch.clone(h)
            return logits, emb
        else:
            return logits
