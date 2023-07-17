import numpy as np
import torch
import torch.nn as nn
from models.aggregators import BaseAggregator
from models.aggregators.model_utils import PPEG, NystromTransformerLayer


class TransMIL(BaseAggregator):
    def __init__(self, num_classes, input_dim=1024, **kwargs):
        super(BaseAggregator, self).__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.num_classes = num_classes
        self.layer1 = NystromTransformerLayer(dim=512)
        self.layer2 = NystromTransformerLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.num_classes)

    def forward(self, x, coords=None):

        h = x  #[B, n, 1024]

        h = self._fc1(h)  #[B, n, 512]

        #----> padding
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  #[B, N, 512]

        #----> cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)

        #----> first translayer
        h = self.layer1(h)  #[B, N, 512]

        #----> ppeg
        h = self.pos_layer(h, _H, _W)  #[B, N, 512]

        #----> second translayer
        h = self.layer2(h)  #[B, N, 512]

        #----> cls_token
        h = self.norm(h)[:, 0]

        #----> predict
        logits = self._fc2(h)  #[B, n_classes]

        return logits
