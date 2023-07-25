import timm
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models


class BITSiamnese(nn.Module):
    def __init__(self):
        super().__init__()

        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        if 1:
            self.bit = timm.create_model('resnetv2_50x1_bitm', pretrained=True, num_classes=2)
            self.bit.head.fc = nn.Identity()
            self.bit.head.flatten = nn.Identity()

            for param in self.bit.parameters():
                param.requires_grad = False

            if 1:
                for param in self.bit.stages[3].blocks[2].parameters():
                    param.requires_grad = True
            else:
                for param in self.bit.stem.parameters():
                    param.requires_grad = True

                for param in self.bit.stages[0].parameters():
                    param.requires_grad = True
            self.classifier = nn.Linear(2048 * 1, 2)
        else:
            self.bit = models.vgg11(pretrained=True).features
            for param in self.bit[:16].parameters():
                param.requires_grad = False

            self.classifier = nn.Linear(512*1, 2)

    def forward(self, x):
        x0 = x[0]  #(B, C, H, W, Z)
        x1 = x[1]
        x0 = x0.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]
        x1 = x1.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]
        x0 = self.bit(x0)
        x1 = self.bit(x1)
        out = self.avg(x1) - self.avg(x0)  # (B*23, 512, 1, 1)
        out = out[:, :, 0, 0]
        out, _ = torch.max(out, 0)
        out = out.unsqueeze(0)
        #out = out.view(1, -1)
        out = self.classifier(out)
        return out,


def print_num_of_parameters(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    print('Number of parameters: ' + str(sum([np.prod(p.size()) for p in model_parameters])))


if __name__ == "__main__":
    g = BITSiamnese()
    print(g([torch.rand(1, 3, 256, 256, 23), torch.rand(1, 3, 256, 256, 23)])[0].shape)
    print_num_of_parameters(g)

