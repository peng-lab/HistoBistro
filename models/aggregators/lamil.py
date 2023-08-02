import torch
import torch.nn as nn

from models.aggregators import BaseAggregator
from models.aggregators.model_utils import GraphTransformerLayer


class LAMIL(BaseAggregator):
    def __init__(self, num_classes):
        super(BaseAggregator, self).__init__()
        self.num_classes = num_classes
        self._fc1 = nn.Sequential(nn.Linear(2048, 512, bias=True), nn.ReLU())
        self._fc2 = nn.Linear(512, self.num_classes, bias=True)
        self.gt1 = GraphTransformerLayer(in_dim=512, out_dim=512, num_heads=8)
        self.gt2 = GraphTransformerLayer(in_dim=512, out_dim=512, num_heads=8)

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