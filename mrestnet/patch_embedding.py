import torch
from timm.models.layers import trunc_normal_


class PatchEmbedding(torch.nn.Module):
    def __init__(self, image_size, patch_size, channels, dimensionality):
        super().__init__()
        self.image_size = image_size
        self.dimensionality = dimensionality
        self.patch_size = patch_size

        # Create patches and embeddings in one step
        self.embedding = torch.nn.Conv2d(
            in_channels=channels, out_channels=dimensionality, kernel_size=patch_size, stride=patch_size)
        self.cls_token = torch.nn.Parameter(
            torch.zeros([1, 1, dimensionality]))

        trunc_normal_(self.cls_token, std=0.02)

    def forward(self, inputs):
        x = self.embedding(inputs).flatten(2).transpose(1, 2)
        # x = torch.flatten(x, start_dim=2, end_dim=3)
        # Repeat the class token across the batch
        cls_token = torch.repeat_interleave(
            self.cls_token, repeats=inputs.shape[0], dim=0)
        x = torch.cat((cls_token, x), 1)

        return x

