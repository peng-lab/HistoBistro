from typing import Optional

import torch
import torch.nn as nn

from models.aggregators import BaseAggregator
from models.aggregators.model_utils import MILAttention


class AttentionMIL(BaseAggregator):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        encoder: Optional[nn.Module] = None,
        attention: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        **kwargs
    ) -> None:
        """Create a new attention MIL model.
        Args:
            n_feats:  The nuber of features each bag instance has.
            n_out:  The number of output layers of the model.
            encoder:  A network transforming bag instances into feature vectors.
        """
        super(BaseAggregator, self).__init__()
        self.encoder = encoder or nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU()
        )
        self.attention = attention or MILAttention(256)
        self.head = head or nn.Sequential(
            nn.Flatten(),
            # nn.BatchNorm1d(256),
            nn.Dropout(),
            nn.Linear(256, num_classes)
        )

    def forward(self, bags, coords=None, tiles=None, **kwargs):
        assert bags.ndim == 3
        if tiles is not None:
            assert bags.shape[0] == tiles.shape[0]
        else:
            tiles = torch.tensor([bags.shape[1]],
                                 device=bags.device).unsqueeze(0)

        embeddings = self.encoder(bags)

        # mask out entries if tiles < num_tiles
        masked_attention_scores = self._masked_attention_scores(
            embeddings, tiles
        )
        weighted_embedding_sums = (masked_attention_scores * embeddings).sum(-2)

        scores = self.head(weighted_embedding_sums)

        return scores

    def _masked_attention_scores(self, embeddings, tiles):
        """Calculates attention scores for all bags.
        Returns:
            A tensor containingtorch.concat([torch.rand(64, 256), torch.rand(64, 23)], -1)
             *  The attention score of instance i of bag j if i < len[j]
             *  0 otherwise
        """
        bs, bag_size = embeddings.shape[0], embeddings.shape[1]
        attention_scores = self.attention(embeddings)

        # a tensor containing a row [0, ..., bag_size-1] for each batch instance
        idx = (torch.arange(bag_size).repeat(bs, 1).to(attention_scores.device))

        # False for every instance of bag i with index(instance) >= lens[i]
        attention_mask = (idx < tiles).unsqueeze(-1)

        masked_attention = torch.where(
            attention_mask, attention_scores,
            torch.full_like(attention_scores, -1e10)
        )

        return torch.softmax(masked_attention, dim=1)
