import torch
from .common import FeedForward
from timm.models.layers import DropPath
import numpy as np
from einops import rearrange


class DecoderBlock(torch.nn.Module):
    def __init__(self, dimensionality, heads, h_size, dropout_rate, drop_path_rate):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dimensionality)
        self.norm2 = torch.nn.LayerNorm(dimensionality)

        self.attention = torch.nn.MultiheadAttention(
            embed_dim=dimensionality,
            num_heads=heads,
            dropout=dropout_rate,
            batch_first=True)

        self.mlp = FeedForward(dimensionality=dimensionality,
                               h_size=h_size, dropout_rate=dropout_rate)
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0.0 else torch.nn.Identity()

    def forward(self, inputs, mask=None, return_attention=False):
        x = self.norm1(inputs)
        attention_output = self.attention(x, x, x)
        if return_attention:
            return attention_output[1]
        x = inputs + self.drop_path(attention_output[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    # Load the pretrained weights
    def load(self, weights):
        pass


class Decoder(torch.nn.Module):
    def __init__(self, number_of_classes, input_shape, patch_size, dimensionality,
                 heads, h_size, dropout, layers, drop_path_rate):
        super().__init__()

        # The number of classes K
        self.number_of_classes = number_of_classes
        self.d_model = dimensionality
        self.scale = dimensionality ** -0.5
        self.patch_size = patch_size

        # The grid shape
        self.grid_shape = (input_shape[0]//self.patch_size,
                           input_shape[1]//self.patch_size)

        self.tokenizer = torch.nn.Parameter(torch.randn(
            [1, dimensionality, number_of_classes]))

        dpr = np.linspace(0.0, drop_path_rate, layers)
        self.blocks = torch.nn.ModuleList(
            [DecoderBlock(dimensionality=dimensionality, heads=heads, h_size=h_size,
                          dropout_rate=dropout, drop_path_rate=dpr[i]) for i in range(layers)]
        )

        self.mask_norm = torch.nn.LayerNorm(number_of_classes)

    def forward(self, inputs):
        x = inputs

        for blk in self.blocks:
            x = blk(x)

        masks = torch.matmul(x, self.tokenizer)

        masks = self.mask_norm(masks)

        masks = rearrange(masks, "b (h w) n -> b n h w", h=self.grid_shape[0])

        return masks

    # Load the pretrained weights
    def load(self, weights):
        pass
