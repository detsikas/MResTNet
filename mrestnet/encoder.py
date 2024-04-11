import torch
from timm.models.layers import DropPath
from .common import FeedForward
import numpy as np


class EncoderBlock(torch.nn.Module):
    def __init__(self, dimensionality, num_heads, dff, dropout_rate=0.1, drop_path_rate=0.1):
        super(EncoderBlock, self).__init__()

        self.layer_norm_1 = torch.nn.LayerNorm(dimensionality)
        self.layer_norm_2 = torch.nn.LayerNorm(dimensionality)

        self.self_attention = torch.nn.MultiheadAttention(
            embed_dim=dimensionality,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True)

        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0.0 else torch.nn.Identity()

        self.ffn = FeedForward(dimensionality=dimensionality,
                               h_size=dff, dropout_rate=dropout_rate)

    def forward(self, inputs):
        y = self.layer_norm_1(inputs)
        y = self.self_attention(y, y, y)

        x = inputs + self.drop_path(y[0])
        x = x + self.drop_path(self.ffn(self.layer_norm_2(x)))

        return x


class Encoder(torch.nn.Module):
    def __init__(self, layers, dimensionality, heads, h_size, dropout_rate=0.1, drop_path_rate=0.1):
        super(Encoder, self).__init__()

        self.dimensionality = dimensionality
        self.layers = layers

        self.layer_norm = torch.nn.LayerNorm(dimensionality)

        dpr = np.linspace(0.0, drop_path_rate, layers)
        self.enc_layers = torch.nn.ModuleList(
            [EncoderBlock(dimensionality=dimensionality,
                          num_heads=heads,
                          dff=h_size,
                          dropout_rate=dropout_rate,
                          drop_path_rate=dpr[i])
             for i in range(layers)]
        )

        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, inputs):
        x = self.dropout(inputs)

        for i in range(self.layers):
            x = self.enc_layers[i](x)

        x = self.layer_norm(x)

        return x
