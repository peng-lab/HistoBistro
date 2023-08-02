import random

import dgl
import torch

from attentionmil import AttentionMIL
from lamil import LAMIL
from perceiver import Perceiver
from transformer import Transformer
from transmil import TransMIL


def test_attentionmil():
    attentionmil = AttentionMIL(num_classes=2, num_features=1024)
    input = torch.rand(1, 1, 1024)
    output = attentionmil(input)
    assert torch.equal(torch.tensor(output.size()), torch.tensor([1, 2]))


# TODO: Make LAMIL work!
def test_lamil():
    batch_size = 1 
    num_tiles = 1000 
    tile_dim = 1024 
    num_classes = 4
    tile_coords = torch.tensor([(random.random(), random.random()) for _ in range(num_tiles)]) 
    knn1, knn2 = 16, 64 
    g1, g2 = dgl.knn_graph(tile_coords, knn1), dgl.knn_graph(tile_coords, knn2) 
    wsi = torch.randn(batch_size, num_tiles, tile_dim) 
    
    lamil = LAMIL(num_classes=num_classes)
    output = lamil(wsi, g1, g2)
    assert torch.equal(torch.tensor(output.size()), torch.tensor([1, 2]))


def test_perceiver():
    perceiver = Perceiver(num_classes=2)
    input = torch.rand(1, 1, 2048)
    output = perceiver(input)
    assert torch.equal(torch.tensor(output.size()), torch.tensor([1, 2]))


def test_transformer():
    transformer = Transformer(num_classes=2)
    input = torch.rand(1, 1, 2048)
    output = transformer(input)
    assert torch.equal(torch.tensor(output.size()), torch.tensor([1, 2]))


def test_transmil():
    transmil = TransMIL(num_classes=2)
    input = torch.rand(1, 1, 1024)
    output = transmil(input)
    assert torch.equal(torch.tensor(output.size()), torch.tensor([1, 2]))


if __name__ == "__main__":

    test_attentionmil()
    #test_lamil()
    test_perceiver()
    test_transformer()
    test_transmil()
    
   


