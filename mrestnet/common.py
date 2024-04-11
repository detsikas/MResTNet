import torch


# The feedforward layer and the top of the encoder and decoder blocks
class FeedForward(torch.nn.Module):
    def __init__(self, dimensionality, h_size, dropout_rate=0.1):
        super().__init__()

        self.dense_0 = torch.nn.Linear(
            in_features=dimensionality, out_features=h_size)
        self.activation_0 = torch.nn.GELU()
        self.dropout_0 = torch.nn.Dropout(dropout_rate)
        self.dense_1 = torch.nn.Linear(
            in_features=h_size, out_features=dimensionality)
        self.dropout_1 = torch.nn.Dropout(dropout_rate)

    def forward(self, inputs):
        x = self.dense_0(inputs)
        x = self.activation_0(x)
        x = self.dropout_0(x)
        x = self.dense_1(x)
        x = self.dropout_1(x)
        return x
