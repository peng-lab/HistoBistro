import torch
import torch.nn as nn


class CoordinateEmbedding(nn.Module):
    """
    Coordinate Positional Embedding
    """
    def __init__(self, dim_coords, dim_model):
        super().__init__()
        self.linear = nn.Linear(dim_coords, dim_model)
    def forward(self, coords):
        return self.linear(coords.float())


tokens = torch.rand(1, 3, 2048)
pos_emb = CoordinateEmbedding(2, 2048)
coords = torch.tensor([[5124, 22454], [5127, 22855], [6127, 32855]])
tokens += pos_emb(coords)
