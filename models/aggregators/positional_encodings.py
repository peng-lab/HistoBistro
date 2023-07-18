import numpy as np
import torch
import torch.nn as nn


class CoordinateEmbedding(nn.Module):
    """
    Coordinate Positional Embedding
    """
    def __init__(self, dim_coords: int, dim_model: int):
        super().__init__()
        self.linear = nn.Linear(dim_coords, dim_model)
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.linear(coords.float())
    

class PositionalEncoding2D(nn.Module):
    """
    calculates sin / cos positional embeddings from absolute coordinates (x, y).
    x coords are mapped to first half of channel size, y coords are mapped to second half of channel size.
    adapted to histopathology format from
    https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/torch_encodings.py
    # TODO could be extended with different embeddings like RoFormer
    """
    def __init__(self, channels: int):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        :param coords: A 3d tensor of size (batch_size, n, 2)
        :return: positional encodings of size (batch_size, n, d)
        """
        if len(coords.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")
        bs, n, _ = coords.shape
        
        # scale coords (to start from (0,0))
        coords -= coords.min()

        pos_x, pos_y = coords[:, :, 0], coords[:, :, 1]
        # outer product indpendent of batch size b
        sin_inp_x = torch.einsum("bi,bj->bij", pos_x, self.inv_freq.unsqueeze(0).to(coords.device))
        sin_inp_y = torch.einsum("bi,bj->bij", pos_y, self.inv_freq.unsqueeze(0).to(coords.device))
        emb_x = self.get_emb(sin_inp_x)
        emb_y = self.get_emb(sin_inp_y)
        emb = torch.zeros((bs, n, 2*self.channels), device=coords.device)
        emb[:, :, :self.channels] = emb_x
        emb[:, :, self.channels:2*self.channels] = emb_y

        return emb
    
    @staticmethod
    def get_emb(sin_inp: torch.Tensor) -> torch.Tensor:
        """
        Gets a base embedding for one dimension with sin and cos intertwined
        """
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return torch.flatten(emb, -2, -1)


class ConcatEmbedding(nn.Module):
    """
    Concatenate learned positional embedding to feature vector
    """
    def __init__(self, dim_coords: int, dim_emb: int):
        super().__init__()
        self.embedding = nn.Linear(dim_coords, dim_emb)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # center coordinates around 0
        coords = coords.float()
        coords -= coords.mean(dim=1, keepdim=True)

        return self.embedding(coords.float())


device = torch.device("cpu") # torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokens = torch.rand(1, 3, 2048, device=device)
# pos_emb = CoordinateEmbedding(2, 2048)
# pos_emb = PositionalEncoding2D(2048)
# pos_emb = LearnedPositionalEmbedding(2048)
pos_emb = ConcatEmbedding(2, 2)
coords = torch.tensor([[[5124, 22454], [5127, 22855], [6127, 32855]]], device=device)
# tokens += pos_emb(coords)
tokens = torch.cat((tokens, pos_emb(coords)), dim=-1)
print(pos_emb(coords))
print(tokens.shape)
