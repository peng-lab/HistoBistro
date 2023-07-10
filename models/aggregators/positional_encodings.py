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
        coords = torch.stack(coords, dim=-1)
        return self.linear(coords.float())


tokens = torch.rand(1, 3, 2048)
pos_emb = CoordinateEmbedding(2, 2048)
x = torch.tensor([[5124, 5127, 6127]])
y = torch.tensor([[22454, 22855, 32855]])
tokens += pos_emb((x, y))
