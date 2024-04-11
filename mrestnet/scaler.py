import torch
import sys


class Scaler(torch.nn.Module):
    def __init__(self, in_channels, activation='relu', starting_filters=16):
        super(Scaler, self).__init__()
        filters = starting_filters
        self.conv_1 = ConvBlock(in_channels=in_channels,
                                filters=filters, activation=activation)
        self.dropout_1 = torch.nn.Dropout(0.1)

        filters *= 2
        self.max_pool = torch.nn.MaxPool2d(2)
        self.conv_2a = ConvBlock(in_channels=self.conv_1.out_channels,
                                 filters=filters, activation=activation)
        self.conv_2b = ConvBlock(in_channels=self.conv_2a.out_channels,
                                 filters=filters, activation=activation)
        self.dropout_2 = torch.nn.Dropout(0.3)

        filters *= 2
        self.conv_3a = ConvBlock(in_channels=self.conv_2b.out_channels,
                                 filters=filters, activation=activation)
        self.conv_3b = ConvBlock(in_channels=self.conv_3a.out_channels,
                                 filters=filters, activation=activation)
        self.dropout_3 = torch.nn.Dropout(0.3)

        filters *= 2
        self.conv_4a = ConvBlock(in_channels=self.conv_3b.out_channels,
                                 filters=filters, activation=activation)
        self.conv_4b = ConvBlock(in_channels=self.conv_4a.out_channels,
                                 filters=filters, activation=activation)
        self.dropout_4 = torch.nn.Dropout(0.3)

        '''
        self.flatten = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(64)
        self.dense_2 = tf.keras.layers.Dense(10)
        self.dense_3 = tf.keras.layers.Dense(1, activation='tanh') <--relu
        '''
        self.output_layer = torch.nn.Conv2d(in_channels=self.conv_4b.out_channels,
                                            out_channels=1, kernel_size=1, padding='same')

    def forward(self, inputs):
        x = self.conv_1(inputs)
        x = self.dropout_1(x)

        x = self.max_pool(x)
        x = self.conv_2a(x)
        x = self.conv_2b(x)
        x = self.dropout_2(x)

        x = self.max_pool(x)
        x = self.conv_3a(x)
        x = self.conv_3b(x)
        x = self.dropout_3(x)

        x = self.max_pool(x)
        x = self.conv_4a(x)
        x = self.conv_4b(x)
        x = self.dropout_4(x)

        x = self.max_pool(x)

        '''
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        '''
        x = self.output_layer(x)

        return x

    def load(self, weights):
        pass


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, filters, activation, kernel_size=3, dilation_rate=1):
        super(ConvBlock, self).__init__()
        self.activation = activation
        self.out_channels = filters
        self.conv_layer = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=filters, kernel_size=kernel_size, dilation=dilation_rate, padding='same')
        self.bn_layer = torch.nn.BatchNorm2d(num_features=filters)
        if activation == 'relu':
            self.activation_layer = torch.nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation_layer = torch.nn.LeakyReLU()
        elif activation is None:
            self.activation_layer = None
        else:
            print('Bad activation')
            sys.exit(0)

    def forward(self, inputs):
        x = self.conv_layer(inputs)
        x = self.bn_layer(x)
        if self.activation is not None:
            x = self.activation_layer(x)
        return x
