import torch
from einops import rearrange, repeat
from einops.layers.torch import Reduce
from torch import nn

from models.aggregators import BaseAggregator
from models.aggregators.model_utils import (FeedForward, PerceiverAttention,
                                            PreNorm, cache_fn, fourier_encode)


class Perceiver(BaseAggregator):
    def __init__(
        self,
        *,
        num_freq_bands=6,
        depth=2,  
        max_freq=10.,
        input_dim=512,
        input_axis=1,
        num_latents=256,
        latent_dim=512,
        cross_heads=1,
        latent_heads=8,
        cross_dim_head=64,
        latent_dim_head=64,
        num_classes=1000,
        attn_dropout=0.1,
        ff_dropout=0.,
        weight_tie_layers=False,
        fourier_encode_data=True,
        self_per_cross_attn=1,
        final_classifier_head=True
    ):
        """The shape of the final attention mechanism will be:
        depth * (cross attention -> self_per_cross_attn * self attention)
        Args:
          num_freq_bands: Number of freq bands, with original value (2 * K + 1)
          depth: Depth of net.
          max_freq: Maximum frequency, hyperparameter depending on how
              fine the data is.
          freq_base: Base for the frequency
          input_channels: Number of channels for each token of the input.
          input_axis: Number of axes for input data (2 for images, 3 for video)
          num_latents: Number of latents, or induced set points, or centroids.
              Different papers giving it different names.
          latent_dim: Latent dimension.
          cross_heads: Number of heads for cross attention. Paper said 1.
          latent_heads: Number of heads for latent self attention, 8.
          cross_dim_head: Number of dimensions per cross attention head.
          latent_dim_head: Number of dimensions per latent self attention head.
          num_classes: Output number of classes.
          attn_dropout: Attention dropout
          ff_dropout: Feedforward dropout
          weight_tie_layers: Whether to weight tie layers (optional).
          fourier_encode_data: Whether to auto-fourier encode the data, using
              the input_axis given. defaults to True, but can be turned off
              if you are fourier encoding the data yourself.
          self_per_cross_attn: Number of self attention blocks per cross attn.
          final_classifier_head: mean pool and project embeddings to number of classes (num_classes) at the end
        """
        super(BaseAggregator, self).__init__()
        self.input_axis = input_axis
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands

        self._fc1 = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU()
        )  # adapted to 2048 from RetCCL features

        self.fourier_encode_data = fourier_encode_data
        fourier_channels = (
            input_axis * ((num_freq_bands * 2) + 1)
        ) if fourier_encode_data else 0
        input_dim = fourier_channels + input_dim

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        get_cross_attn = lambda: PreNorm(
            latent_dim,
            PerceiverAttention(
                latent_dim,
                input_dim,
                heads=cross_heads,
                dim_head=cross_dim_head,
                dropout=attn_dropout
            ),
            context_dim=input_dim
        )
        get_cross_ff = lambda: PreNorm(
            latent_dim, FeedForward(latent_dim, dropout=ff_dropout)
        )
        get_latent_attn = lambda: PreNorm(
            latent_dim,
            PerceiverAttention(
                latent_dim,
                heads=latent_heads,
                dim_head=latent_dim_head,
                dropout=attn_dropout
            )
        )
        get_latent_ff = lambda: PreNorm(
            latent_dim, FeedForward(latent_dim, dropout=ff_dropout)
        )

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(
            cache_fn,
            (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff)
        )

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for _ in range(self_per_cross_attn):
                self_attns.append(
                    nn.ModuleList(
                        [
                            get_latent_attn(**cache_args),
                            get_latent_ff(**cache_args)
                        ]
                    )
                )

            self.layers.append(
                nn.ModuleList(
                    [
                        get_cross_attn(**cache_args),
                        get_cross_ff(**cache_args), self_attns
                    ]
                )
            )

        self.to_logits = nn.Sequential(
            Reduce('b n d -> b d', 'max'), nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes)
        ) if final_classifier_head else nn.Identity()

    def forward(self, data, mask=None, return_embeddings=False):

        ###fully connected layer
        data = self._fc1(data)

        b, *axis, _, device = *data.shape, data.device
        assert len(
            axis
        ) == self.input_axis, 'input data must have the right number of axis'

        if self.fourier_encode_data:
            # calculate fourier encoded positions in the range of [-1, 1], for all axis

            axis_pos = list(
                map(
                    lambda size: torch.
                    linspace(-1., 1., steps=size, device=device), axis
                )
            )
            pos = torch.stack(torch.meshgrid(*axis_pos, indexing='ij'), dim=-1)
            enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
            enc_pos = repeat(enc_pos, '... -> b ...', b=b)

            data = torch.cat((data, enc_pos), dim=-1)

        # concat to channels of data and flatten axis

        data = rearrange(data, 'b ... d -> b (...) d')

        x = repeat(self.latents, 'n d -> b n d', b=b)

        # layers

        for cross_attn, cross_ff, self_attns in self.layers:
            x = cross_attn(x, context=data, mask=mask) + x
            x = cross_ff(x) + x

            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x

        # allow for fetching embeddings

        if return_embeddings:
            return x

        # to logits

        return self.to_logits(x)