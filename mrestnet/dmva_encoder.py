import torch
import sys
from .convolutional_blocks import ConvBlock, MultiResBlock


class ResPathBlock(torch.nn.Module):
    def __init__(self, in_channels, filters, activation):
        super(ResPathBlock, self).__init__()
        self.activation = activation
        self.sc_conv_block = ConvBlock(
            in_channels=in_channels, filters=filters, kernel_size=1, activation=None)
        self.conv_block = ConvBlock(
            in_channels=in_channels, filters=filters, activation=activation)
        if activation == 'leaky_relu':
            self.activation_layer = torch.nn.LeakyReLU()
        elif activation == 'relu':
            self.activation_layer = torch.nn.ReLU()
        elif activation is None:
            self.activation_layer = None
        else:
            print('Bad extivation')
            sys.exit(0)
        self.bn_layer = torch.nn.BatchNorm2d(num_features=filters)

    def forward(self, inputs):
        shortcut = self.sc_conv_block(inputs)
        rx = self.conv_block(inputs)
        rx += shortcut
        if self.activation is not None:
            rx = self.activation_layer(rx)

        return self.bn_layer(rx)


class ResPath(torch.nn.Module):
    def __init__(self, in_channels, filters, length, activation='relu'):
        super(ResPath, self).__init__()
        self.length = length
        self.blocks = [ResPathBlock(in_channels=in_channels,
                                    filters=filters,
                                    activation=activation)
                       for i in range(length)]

    def forward(self, inputs):
        x = inputs
        for i in range(self.length):
            x = self.blocks[i](x)
        return x


class DMVAEncoder(torch.nn.Module):
    def __init__(self, in_channels, starting_filters=16, with_dropout=False, activation='relu'):
        super(DMVAEncoder, self).__init__()
        layer_filters = starting_filters
        self.with_dropout = with_dropout
        self.pooling_layer = torch.nn.MaxPool2d(2)
        self.mres_1 = MultiResBlock(
            in_channels=in_channels, filters=layer_filters, activation=activation)
        if with_dropout:
            self.dp_1 = torch.nn.Dropout(0.1)
            self.dp_2 = torch.nn.Dropout(0.1)
            self.dp_3 = torch.nn.Dropout(0.2)
            self.dp_4 = torch.nn.Dropout(0.2)
            self.dp_5 = torch.nn.Dropout(0.2)
        # self.res_path_1 = ResPath(
        #    name='res_path_1', filters=layer_filters, length=4)

        layer_filters *= 2
        self.mres_2 = MultiResBlock(
            in_channels=self.mres_1.output_filters, filters=layer_filters, activation=activation)
        # self.res_path_2 = ResPath(
        #    name='res_path_2', filters=layer_filters, length=3)

        layer_filters *= 2
        self.mres_3 = MultiResBlock(
            in_channels=self.mres_2.output_filters, filters=layer_filters, activation=activation)
        # self.res_path_3 = ResPath(
        #    name='res_path_3', filters=layer_filters, length=2)

        layer_filters *= 2
        self.mres_4 = MultiResBlock(
            in_channels=self.mres_3.output_filters, filters=layer_filters, activation=activation, dilation_rate=2)
        # self.res_path_4 = ResPath(
        #    name='res_path_4', filters=layer_filters, length=1)

        layer_filters *= 2
        self.mres_5 = MultiResBlock(
            in_channels=self.mres_4.output_filters, filters=layer_filters, activation=activation, dilation_rate=2)
        # self.res_path_5 = ResPath(
        #    name='res_path_5', filters=layer_filters, length=1)
        self.output_channels = self.mres_5.output_filters

    def forward(self, inputs):
        x = self.mres_1(inputs)
        if self.with_dropout:
            x = self.dp_1(x)
        # sc_1 = self.res_path_1(x)

        x = self.pooling_layer(x)
        x = self.mres_2(x)
        if self.with_dropout:
            x = self.dp_2(x)
        # sc_2 = self.res_path_2(x)

        x = self.pooling_layer(x)
        x = self.mres_3(x)
        if self.with_dropout:
            x = self.dp_3(x)
        # sc_3 = self.res_path_3(x)

        x = self.pooling_layer(x)
        x = self.mres_4(x)
        if self.with_dropout:
            x = self.dp_4(x)
        # sc_4 = self.res_path_4(x)

        x = self.pooling_layer(x)
        x = self.mres_5(x)
        if self.with_dropout:
            x = self.dp_5(x)
        # sc_5 = self.res_path_5(x)

        return x  # , sc_1, sc_2, sc_3, sc_4, sc_5]
