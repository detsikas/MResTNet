from .encoder import Encoder
from .decoder import Decoder
from .patch_embedding import PatchEmbedding
import torch
import numpy as np
import math
import scipy.ndimage
from timm.models.layers import trunc_normal_
from . import loader
from .dmva_decoder import DMVADecoder
from .scaler import Scaler
from .fuser import Fuser


# The transformer encoder-decoder model
class Transformer(torch.nn.Module):
    def __init__(self, input_channels, number_of_classes, input_shape, patch_size, encoder_layers, encoder_heads, decoder_heads,
                 decoder_layers, model_dimensionality, h_size):
        super(Transformer, self).__init__()

        self.input_shape_ = input_shape
        self.patch_size = patch_size
        self.n_cls = number_of_classes

        # Patches and patch embeddings
        self.patch_embedding = PatchEmbedding(
            image_size=input_shape, patch_size=patch_size, channels=input_channels, dimensionality=model_dimensionality)

        # Positional embeddings
        number_of_patches = input_shape[0] * \
            input_shape[1]//(patch_size*patch_size)
        self.positional_embedding_shape = [
            1, number_of_patches + 1, model_dimensionality]

        self.positional_embedding = torch.nn.Parameter(
            torch.randn(self.positional_embedding_shape))
        trunc_normal_(self.positional_embedding, std=0.02)

        self.encoder = Encoder(layers=encoder_layers, dimensionality=model_dimensionality,
                               heads=encoder_heads, h_size=h_size,
                               dropout_rate=0.0, drop_path_rate=0.1)

        # The Decoder that is specific for the downstream task (in our case semantic segmentation)
        self.transformer_decoder = Decoder(number_of_classes=number_of_classes, input_shape=input_shape, patch_size=patch_size, layers=decoder_layers,
                                           dimensionality=model_dimensionality,
                                           heads=decoder_heads, h_size=h_size,
                                           drop_path_rate=0.0, dropout=0.1)
        self.unet_decoder = DMVADecoder(number_of_classes=number_of_classes, input_shape=input_shape, patch_size=patch_size,
                                        dimensionality=model_dimensionality, with_dropout=True, activation='relu')

        self.scaler = Scaler(in_channels=3)
        self.scaler_activation = torch.nn.Sigmoid()

        self.unet_norm = torch.nn.LayerNorm(number_of_classes)
        self.fuser = Fuser(number_of_classes=number_of_classes)

    def forward(self, inputs):
        scale = self.scaler(inputs)
        scale = torch.nn.functional.interpolate(
            scale, scale_factor=self.patch_size, mode="bilinear")
        scale = self.scaler_activation(scale)

        x = self.patch_embedding(inputs)

        # The x shape is now (batch, sequence length h*w/(patch_size*patch_size), model dimensionality)
        # Adding the positional embeddings
        x = x+self.positional_embedding

        # Going through the encoder
        x = self.encoder(x)

        # Removing the class token
        x = x[:, 1:]

        # x is now encoded with shape (batch_size, target_length, model_dimensionality)
        # Decoding to create the segmentations masks
        xt = self.transformer_decoder(x)

        # x is now decoded with shape (batch_size, )
        # Restoring to the model input image size
        xt = torch.nn.functional.interpolate(
            xt, scale_factor=self.patch_size, mode="bilinear")

        xu = self.unet_decoder(x)

        xu *= scale
        xt *= (1-scale)
        # xu = self.unet_norm(xu)

        masks = torch.cat((xt, xu), 1)
        masks = self.fuser(masks)
        # masks = xu+xt

        # xu = self.unet_norm(xu)

        # masks = torch.cat((xt, xu), 1)
        # masks = self.fuser(masks)
        # masks = xu+xt

        # Apply softmax to the output
        # masks = torch.nn.functional.softmax(masks, dim=1)

        return masks

    # Load the pretrained weights
    def load_pretrained(self, weights):
        state_dict = self.state_dict()
        loader.load_conv2d(state_dict, weights,
                           'patch_embedding.embedding', 'embedding')
        loader.load_parameter(state_dict, weights,
                              'patch_embedding.cls_token', 'cls')
        loader.load_parameter_with_func(state_dict, weights, 'positional_embedding',
                                        'Transformer/posembed_input/pos_embedding', self.interpolate_embeddings)

        # Encoder
        loader.load_layer_norm(
            state_dict, weights, 'encoder.layer_norm', 'Transformer/encoder_norm')
        for i in range(self.encoder.layers):
            loader.load_layer_norm(
                state_dict, weights, f'encoder.enc_layers.{i}.layer_norm_1', f'Transformer/encoderblock_{i}/LayerNorm_0')
            loader.load_layer_norm(
                state_dict, weights, f'encoder.enc_layers.{i}.layer_norm_2', f'Transformer/encoderblock_{i}/LayerNorm_2')
            loader.load_multihead_self_attention(
                state_dict, weights, f'encoder.enc_layers.{i}.self_attention', f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1')
            loader.load_dense(
                state_dict, weights, f'encoder.enc_layers.{i}.ffn.dense_0', f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_0')
            loader.load_dense(
                state_dict, weights, f'encoder.enc_layers.{i}.ffn.dense_1', f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_1')

        # Verify that all pretrained parameters have been loaded
        '''
        if len(weights) > 0:
            print('Some keys have not been assigned')
            from pprint import pprint
            pprint(list(weights.keys()))
            sys.exit(0)
        '''

    def interpolate_embeddings(self, pretrained_pos_embeddings):
        posemb_tok, posemb_grid = (
            pretrained_pos_embeddings[:, :1],
            pretrained_pos_embeddings[0, 1:],
        )

        gs_old = int(math.sqrt(len(posemb_grid)))
        gs_new = (self.input_shape_[0]//self.patch_size,
                  self.input_shape_[1]//self.patch_size)

        if gs_old == gs_new[0] and gs_old == gs_new[1]:
            return pretrained_pos_embeddings

        # Reshape the pretrained embeddings to a grid
        posemb_grid = posemb_grid.reshape((1, gs_old, gs_old, -1))

        # Interpolate to the new grid size
        zoom = (1, gs_new[0] / gs_old, gs_new[1] / gs_old, 1)
        posemb_grid = scipy.ndimage.zoom(posemb_grid, zoom, order=1)

        # Flatten again the new grid
        posemb_grid = posemb_grid.reshape(1, gs_new[0] * gs_new[1], -1)

        # Put back the class token
        posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)

        return posemb
