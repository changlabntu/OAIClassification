import torch
from models.MRPretrained import MRPretrained
import torch.nn as nn


class MRPretrainedTwoStream(MRPretrained):
    def __init__(self, *args, **kwargs):
        super(MRPretrainedTwoStream, self).__init__(*args, **kwargs)

    def chain_multiply(self, x):
        x = 1 - x.unsqueeze(1).unsqueeze(2)
        return torch.chain_matmul(*x)

    def forward(self, x):  # (1, 3, 224, 224, 23)
        # dummies
        out = None  # output of the model
        features = None  # features we want to further analysis
        # reshape
        x0 = x[0]
        x1 = x[1]
        B = x0.shape[0]
        x0 = x0.permute(0, 4, 1, 2, 3)  # (B, 23, 3, 224, 224)
        x0 = x0.reshape(B * x0.shape[1], x0.shape[2], x0.shape[3], x0.shape[4])  # (B*23, 3, 224, 224)
        x1 = x1.permute(0, 4, 1, 2, 3)  # (B, 23, 3, 224, 224)
        x1 = x1.reshape(B * x1.shape[1], x1.shape[2], x1.shape[3], x1.shape[4])  # (B*23, 3, 224, 224)
        # features
        x0 = self.features(x0)  # (B*23, 512, 7, 7)
        x1 = self.features(x1)  # (B*23, 512, 7, 7)

        if self.fuse == 'cat':  # concatenate across the slices
            x0 = self.avg(x0)  # (B*23, 512, 1, 1)
            x1 = self.avg(x1)  # (B*23, 512, 1, 1)
            x0 = x0.view(B, x0.shape[0] // B, x0.shape[1], x0.shape[2], x0.shape[3])  # (B, 23, 512, 1, 1)
            x1 = x1.view(B, x1.shape[0] // B, x1.shape[1], x1.shape[2], x1.shape[3])  # (B, 23, 512, 1, 1)
            x0cat = x0.view(B, x0.shape[1] * x0.shape[2], x0.shape[3], x0.shape[4])  # (B, 23*512, 1, 1)
            x1cat = x1.view(B, x1.shape[1] * x1.shape[2], x1.shape[3], x1.shape[4])  # (B, 23*512, 1, 1)

            out = self.classifier_cat(x1cat - x0cat)  # (Classes)
            out = out[:, :, 0, 0]

            features = (x0cat, x1cat)

        if self.fuse == 'max':  # max-pooling across the slices
            x = self.avg(x1) - self.avg(x0)  # (B*23, 512, 1, 1)
            x = x.view(B, x.shape[0] // B, x.shape[1], x.shape[2], x.shape[3]) # (B, 23, 512, 1, 1)
            features, _ = torch.max(x, 1)  # (B, 512, 1, 1)
            out = self.classifier(features)  # (Classes)
            out = out[:, :, 0, 0]

        if self.fuse == 'chain0':
            x = x1 - x0  # (23, 512, 7, 7)
            x = self.avg(x)  # (23, 512, 1, 1)
            x = self.classifier(x)
            x = nn.ReLU()(x[:, :, 0, 0])
            x0 = 1 - x[:, 0].unsqueeze(1).unsqueeze(2)
            x1 = 1 - x[:, 1].unsqueeze(1).unsqueeze(2)
            x0 = torch.chain_matmul(*x0)
            x1 = torch.chain_matmul(*x1)
            out = torch.cat([x0, x1], 1)

        if self.fuse == 'chain1':  # best so far original
            x0 = self.classifier(x0)
            x1 = self.classifier(x1)
            x = nn.ReLU()(x0 - x1)  # (23, 2, 7, 7)
            x = self.avg(x)[:, :, 0, 0]
            x0 = self.chain_multiply(x[:, 0])
            x1 = self.chain_multiply(x[:, 1])
            out = torch.cat([x0, x1], 1)

        if self.fuse == 'chain2':  # best so far
            # x = x1 - x0  # (23, 512, 7, 7)
            x0 = self.classifier(x0)
            x1 = self.classifier(x1)
            x = nn.ReLU()(x0 - x1)  # (23, 2, 7, 7)

            a = self.avg(x)[:, :, 0, 0]
            b = self.avg(x.permute(2, 1, 0, 3))[:, :, 0, 0]
            c = self.avg(x.permute(3, 1, 0, 2))[:, :, 0, 0]

            # chain left and right knees separately
            a0 = self.chain_multiply(a[:, 0])
            a1 = self.chain_multiply(a[:, 1])

            # chain the difference
            a2 = self.chain_multiply(a[:, 1] - a[:, 0])
            out = torch.cat([a0, a1], 1)

        return out, features


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.backbone = 'vgg11'
    parser.pretrained = False
    parser.n_classes = 2
    parser.fuse = 'cat'

    mr1 = MRPretrained(parser)
    mr2 = MRPretrainedTwoStream(parser)
    out1 = mr1(torch.rand(4, 3, 224, 224, 23))
    out2 = mr2([torch.rand(4, 3, 224, 224, 23), torch.rand(4, 3, 224, 224, 23)])