import torch


class Fuser(torch.nn.Module):
    def __init__(self, number_of_classes):
        super(Fuser, self).__init__()

        self.layer_1 = torch.nn.Conv2d(
            in_channels=2*number_of_classes, out_channels=2*number_of_classes, kernel_size=3, padding='same')
        self.bn_1 = torch.nn.BatchNorm2d(num_features=2*number_of_classes)

        self.layer_2 = torch.nn.Conv2d(in_channels=2*number_of_classes,
                                       out_channels=number_of_classes, kernel_size=1, padding='same')

    def forward(self, inputs):
        x = self.layer_1(inputs)
        x = self.bn_1(x)
        x = self.layer_2(x)

        return x
