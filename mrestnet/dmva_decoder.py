import torch
from .convolutional_blocks import MultiResBlock
from einops import rearrange


class DMVADecoder(torch.nn.Module):
    def __init__(self, number_of_classes, dimensionality, input_shape,
                 patch_size, with_dropout=False, activation='relu'):
        super(DMVADecoder, self).__init__()

        filters = dimensionality
        self.with_dropout = with_dropout

        # The grid shape
        self.grid_shape = (input_shape[0]//patch_size,
                           input_shape[1]//patch_size)

        # self.upsampling_4 = tf.keras.layers.UpSampling2D(4)
        # self.upsampling_8 = tf.keras.layers.UpSampling2D(8)
        # self.upsampling_16 = tf.keras.layers.UpSampling2D(16)

        # filters //= 2
        # self.conv2d_transpose_1 = tf.keras.layers.Conv2DTranspose(
        #    filters=filters, kernel_size=1, strides=2, padding='same')

        self.multires_block_1 = MultiResBlock(
            in_channels=dimensionality, filters=filters, activation=activation, dilation_rate=1, alpha=1.0)
        # self.concatenate_1 = tf.keras.layers.Concatenate()
        if with_dropout:
            self.dropout_1 = torch.nn.Dropout(0.2)

        filters //= 2
        # self.conv2d_transpose_2 = tf.keras.layers.Conv2DTranspose(
        #    filters=filters, kernel_size=1, strides=2, padding='same')
        self.multires_block_2 = MultiResBlock(
            in_channels=self.multires_block_1.output_filters, filters=filters, activation=activation, dilation_rate=1, alpha=1.0)
        # self.concatenate_1 = tf.keras.layers.Concatenate()
        if with_dropout:
            self.dropout_2 = torch.nn.Dropout(0.2)

        filters //= 2
        # self.conv2d_transpose_3 = tf.keras.layers.Conv2DTranspose(
        #    filters=filters, kernel_size=1, strides=2, padding='same')
        self.multires_block_3 = MultiResBlock(
            in_channels=self.multires_block_2.output_filters, filters=filters, activation=activation, dilation_rate=1, alpha=1.0)
        # self.concatenate = tf.keras.layers.Concatenate()
        if with_dropout:
            self.dropout_3 = torch.nn.Dropout(0.1)

        filters //= 2
        # self.conv2d_transpose_4 = tf.keras.layers.Conv2DTranspose(
        #    filters=filters, kernel_size=1, strides=2, padding='same')
        self.multires_block_4 = MultiResBlock(
            in_channels=self.multires_block_3.output_filters, filters=filters, activation=activation, dilation_rate=1, alpha=1.0)
        # self.concatenate_1 = tf.keras.layers.Concatenate()
        if with_dropout:
            self.dropout_4 = torch.nn.Dropout(0.1)

        # Output
        self.output_layer = torch.nn.Conv2d(in_channels=self.multires_block_4.output_filters,
                                            out_channels=number_of_classes, kernel_size=1, padding='same')

    def forward(self, inputs):
        x = rearrange(inputs, "b (h w) d-> b d h w", h=self.grid_shape[0])

        x = torch.nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear")
        # skip_connection = self.upsampling_2(self.reshape(inputs[-2]))
        x = self.multires_block_1(x)
        # x = self.concatenate([x, skip_connection])
        if self.with_dropout:
            x = self.dropout_1(x)

        x = torch.nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear")
        # skip_connection = self.upsampling_4(self.reshape(inputs[-3]))
        x = self.multires_block_2(x)
        # x = self.concatenate([x, skip_connection])
        if self.with_dropout:
            x = self.dropout_2(x)

        x = torch.nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear")
        # skip_connection = self.upsampling_8(self.reshape(inputs[-4]))
        x = self.multires_block_3(x)
        # x = self.concatenate([x, skip_connection])
        if self.with_dropout:
            x = self.dropout_3(x)

        x = torch.nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear")
        # skip_connection = self.upsampling_16(self.reshape(inputs[-5]))
        x = self.multires_block_4(x)
        # x = self.concatenate([x, skip_connection])
        if self.with_dropout:
            x = self.dropout_4(x)

        x = self.output_layer(x)

        return x
