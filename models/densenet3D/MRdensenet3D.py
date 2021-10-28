from models.densenet3D.densenet3D import DenseNet
import torch.nn as nn
import numpy as np

def print_num_of_parameters(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    print('Number of parameters: ' + str(sum([np.prod(p.size()) for p in model_parameters])))


class MRDenseNet3D(nn.Module):
    """Densenet-BC model class
    Args:
        growth_rate (int) - how many filters to add each layer (k in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self,
                 n_input_channels=3,
                 conv1_t_size=3,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0,
                 num_classes=1000):

        super().__init__()
        self.features = DenseNet(
                 n_input_channels,
                 conv1_t_size,
                 conv1_t_stride,
                 no_max_pool,
                 growth_rate,
                 block_config,
                 num_init_features,
                 bn_size,
                 drop_rate,
                 num_classes).features

        self.avg = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Linear(1024, 2)

    def forward(self, x):
        x0 = x[0]  # (1, 3, 224, 224, 23)
        x1 = x[1]
        x0 = x0.permute(0, 1, 4, 2, 3)  # (1, 3, 23, 224, 224)
        x1 = x1.permute(0, 1, 4, 2, 3)

        x0 = self.features(x0)
        x1 = self.features(x1)  # (1, 1024, 1, 7, 7)

        x0 = nn.ReLU(inplace=True)(x0)
        x1 = nn.ReLU(inplace=True)(x1)

        x0 = self.avg(x0)
        x1 = self.avg(x1)

        x = (x0 - x1).view(x0.shape[0], -1)

        #x = nn.ReLU(inplace=True)(x)

        out = self.classifier(x)
        return out, (x0, x1)


if __name__ == '__main__':
    import torch
    net = MRDenseNet3D()
    print_num_of_parameters(net.features)