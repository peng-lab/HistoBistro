"""
Mostly copy-paste but modified from PyTorch library.
https://github.com/graphdeeplearning/graphtransformer/blob/main/layers/graph_transformer_layer.py
(from Daniel Reisenbüchler)
"""

from math import pi, log

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import dgl.function as fn

#nystrom-attention dependencies
from math import ceil
from torch import nn, einsum
from einops import rearrange, reduce, repeat
import positional_encodings as penc

from functools import wraps
import sklearn.neighbors as n

from einops.layers.torch import Reduce
from scipy import sparse as sp




       

class LAMIL(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self._fc1 = nn.Sequential(nn.Linear(2048, 512, bias=True), nn.ReLU())
        self.num_classes = num_classes
        self.gt1 = GraphTransformerLayer(in_dim=512, out_dim=512, num_heads=8)
        self.gt2 = GraphTransformerLayer(in_dim=512, out_dim=512, num_heads=8)
        self._fc2 = nn.Linear(512, self.num_classes, bias=True)
 

    def forward(self, h, g1, g2, return_emb=False):
        
        
        h = self._fc1(h)
        
        h = self.gt1(g1, h)
        
        h = self.gt2(g2, h)
        
        h = h.mean(dim=1)

        
        if return_emb:
            emb = torch.clone(h)
 
        logits = self._fc2(h)
        
        if return_emb:
            return logits, emb
        else:
            return logits

    


"""
    Graph Transformer Layer
    
"""

"""
    Util functions
"""
def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}
    return func

def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}

    return func


"""
    Single Attention Head
"""

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        
        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        
    
    def propagate_attention(self, g):
        # Compute attention score
        
        
        
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score')) #, edges)
         
        g.apply_edges(scaled_exp('score', np.sqrt(self.out_dim)))
        
        # Send weighted values to target nodes
        eids = g.edges()
        
        
        g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'att'))

        
        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))

        g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))

    
    def forward(self, g, h):
        
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        
        # Reshaping into [num_nodes, num_heads, feat_dim] to 
        # get projections for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)
        
        self.propagate_attention(g)
        
        head_out = g.ndata['wV']/g.ndata['z']
        
        return head_out
    

class GraphTransformerLayer(nn.Module):
    """
        Param: 
    """
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=False, residual=True, use_bias=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm        
        
        self.attention = MultiHeadAttentionLayer(in_dim, out_dim//num_heads, num_heads, use_bias)
        
        self.O = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)
            

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(out_dim)
            
        
    def forward(self, g, h):
        h_in1 = h # for first residual connection
        
        # multi-head attention out
        attn_out = self.attention(g, h)
        
        h = attn_out.view(-1, self.out_channels)
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        h = self.O(h)
        
        if self.residual:
            h = h_in1 + h # residual connection
        
        if self.layer_norm:
            h = self.layer_norm1(h)
                 

        return h
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)
    
    