import torch
from models.MRPretrained import MRPretrained
import torch.nn as nn
import copy


class MRPretrainedRepeat(MRPretrained):
    def __init__(self, *args, **kwargs):
        super(MRPretrainedRepeat, self).__init__(*args, **kwargs)

        # repeat encoders
        if self.args_m.repeat > 0:
            for r in range(self.args_m.repeat):
                setattr(self, 'encoder' + str(r), self.get_encoder(self.args_m))
            #self.encoders = []
            #for r in range(self.args_m.repeat):
            #    self.encoders.append(self.get_encoder(self.args_m))

        if self.fuse == 'simple3':
            self.simplel = nn.Conv2d(self.fmap_c, 1, 1, 1, 0)
            self.simplem = nn.Conv2d(self.fmap_c, 1, 1, 1, 0)
            self.final = nn.Conv2d(2 * self.args_m.repeat, self.args_m.n_classes, 1, 1, 0)

            self.bn0 = nn.BatchNorm2d(2 * self.args_m.repeat)
            self.relu0 = nn.ReLU()

        if self.fuse == 'simple4':
            self.simplel0 = nn.Conv2d(self.fmap_c, 1, 1, 1, 0)
            self.simplel1 = nn.Conv2d(self.fmap_c, 1, 1, 1, 0)
            self.simplel2 = nn.Conv2d(self.fmap_c, 1, 1, 1, 0)
            self.simplel3 = nn.Conv2d(self.fmap_c, 1, 1, 1, 0)
            self.simplel4 = nn.Conv2d(self.fmap_c, 1, 1, 1, 0)
            self.simplem0 = nn.Conv2d(self.fmap_c, 1, 1, 1, 0)
            self.simplem1 = nn.Conv2d(self.fmap_c, 1, 1, 1, 0)
            self.simplem2 = nn.Conv2d(self.fmap_c, 1, 1, 1, 0)
            self.simplem3 = nn.Conv2d(self.fmap_c, 1, 1, 1, 0)
            self.simplem4 = nn.Conv2d(self.fmap_c, 1, 1, 1, 0)
            #self.simple3 = nn.Conv2d(2 * self.args_m.repeat, self.args_m.n_classes, 1, 1, 0)

            self.final = nn.Conv2d(2 * (self.args_m.repeat - 1), self.args_m.n_classes, 1, 1, 0)

            self.bn0 = nn.BatchNorm2d(2 * self.args_m.repeat)

        if self.fuse == 'simple5':
            self.simplel0 = nn.Conv2d(self.fmap_c, 1, 1, 1, 0)
            self.simplel1 = nn.Conv2d(self.fmap_c, 1, 1, 1, 0)
            self.simplel2 = nn.Conv2d(self.fmap_c, 1, 1, 1, 0)
            self.simplel3 = nn.Conv2d(self.fmap_c, 1, 1, 1, 0)
            self.simplel4 = nn.Conv2d(self.fmap_c, 1, 1, 1, 0)
            #self.simple3 = nn.Conv2d(2 * self.args_m.repeat, self.args_m.n_classes, 1, 1, 0)

            self.final = nn.Conv2d(1 * (self.args_m.repeat - 1), self.args_m.n_classes, 1, 1, 0)

            self.bn0 = nn.BatchNorm2d(2 * self.args_m.repeat)

    def chain_multiply(self, x):
        x = 1 - x.unsqueeze(1).unsqueeze(2)
        return torch.chain_matmul(*x)

    def forward(self, x):  # (1, C, 224, 224, 23)
        # reshape
        x0 = x[0]
        x1 = x[1]
        B = x0.shape[0]
        x0 = x0.permute(0, 4, 1, 2, 3)  # (B, 23, C, 224, 224)
        x0 = x0.reshape(B * x0.shape[1], x0.shape[2], x0.shape[3], x0.shape[4])  # (B*23, C, 224, 224)
        x1 = x1.permute(0, 4, 1, 2, 3)  # (B, 23, C, 224, 224)
        x1 = x1.reshape(B * x1.shape[1], x1.shape[2], x1.shape[3], x1.shape[4])  # (B*23, C, 224, 224)

        # features over channel
        n_channels = x1.shape[1]
        x0 = torch.split(x0, split_size_or_sections=1, dim=1)  # (B*23, 1, 224, 224) X C
        x1 = torch.split(x1, split_size_or_sections=1, dim=1)  # (B*23, 1, 224, 224) X C

        #f0 = [self.encoders[i](x0[i].repeat(1, 3, 1, 1)) for i in range(n_channels)]  # (B*23, 512, 7, 7) X C
        #f1 = [self.encoders[i](x1[i].repeat(1, 3, 1, 1)) for i in range(n_channels)]  # (B*23, 512, 7, 7) X C

        f0 = [getattr(self, 'encoder' + str(i))(x0[i].repeat(1, 3, 1, 1)) for i in range(n_channels)]  # (B*23, 512, 7, 7) X C
        f1 = [getattr(self, 'encoder' + str(i))(x1[i].repeat(1, 3, 1, 1)) for i in range(n_channels)]  # (B*23, 512, 7, 7) X C

        if self.fuse == 'simple2':  # classify per lateral and medial region then max pooling

            f = [self.avg(f0[i]) - self.avg(f1[i]) for i in range(n_channels)]   # (B*23, 512, 1, 1) X C
            f = [x.view(B, x.shape[0] // B, x.shape[1], x.shape[2], x.shape[3]) for x in f]  # (B, 23, 512, 1, 1) X C

            xl = [y[:, :13, :, :, :] for y in f]  # (B, 12, 512, 1, 1) X C
            xm = [y[:, 13:, :, :, :] for y in f]  # (B, 11, 512, 1, 1) X C

            xl = [torch.max(y, 1)[0] for y in xl]  # (B, 512, 1, 1) X C
            xm = [torch.max(y, 1)[0] for y in xm]  # (B, 512, 1, 1) X C

            xl = [self.simplel(y) for y in xl]  # (B, 2, 1, 1) X C
            xm = [self.simplem(y) for y in xm]  # (B, 2, 1, 1) X C

            xl = torch.stack(xl, 4)  # (B, 2, 1, C)
            xm = torch.stack(xm, 4)  # (B, 2, 1, C)

            if 0:
                x = xl.mean(3) + xm.mean(3)

            features = xl.sum(3) + xm.sum(3)
            x = self.simple1(features)   # (B, 2, 1, 1)
            out = x[:, :, 0, 0]

        if self.fuse == 'simple3':  # classify per lateral and medial region then max pooling

            f = [self.avg(f0[i]) - self.avg(f1[i]) for i in range(n_channels)]   # (B*23, 512, 1, 1) X C
            f = [x.view(B, x.shape[0] // B, x.shape[1], x.shape[2], x.shape[3]) for x in f]  # (B, 23, 512, 1, 1) X C

            xl = [y[:, :13, :, :, :] for y in f]  # (B, 13, 512, 1, 1) X C
            xm = [y[:, 13:, :, :, :] for y in f]  # (B, 10, 512, 1, 1) X C

            xl = [torch.max(y, 1)[0] for y in xl]  # (B, 512, 1, 1) X C
            xm = [torch.max(y, 1)[0] for y in xm]  # (B, 512, 1, 1) X C

            xl = [self.simplel(y) for y in xl]  # (B, 1, 1, 1) X C
            xm = [self.simplem(y) for y in xm]  # (B, 1, 1, 1) X C

            features = torch.cat(xl + xm, 1)  # (B, 2*C, 1, 1)
            features = self.relu0(features)

            x = self.final(features)   # (B, 2, 1, 1)
            out = x[:, :, 0, 0]

        if self.fuse == 'simple4':  # classify per lateral and medial region then max pooling

            f = [self.avg(f0[i]) - self.avg(f1[i]) for i in range(n_channels)]   # (B*23, 512, 1, 1) X C
            f = [x.view(B, x.shape[0] // B, x.shape[1], x.shape[2], x.shape[3]) for x in f]  # (B, 23, 512, 1, 1) X C

            xl = [y[:, :13, :, :, :] for y in f]  # (B, 13, 512, 1, 1) X C
            xm = [y[:, 13:, :, :, :] for y in f]  # (B, 10, 512, 1, 1) X C

            xl = [torch.max(y, 1)[0] for y in xl]  # (B, 512, 1, 1) X C
            xm = [torch.max(y, 1)[0] for y in xm]  # (B, 512, 1, 1) X C

            xl0 = self.simplel0(xl[0])
            xl1 = self.simplel1(xl[1])
            xl2 = self.simplel2(xl[2])
            xl3 = self.simplel3(xl[3])
            xl4 = self.simplel4(xl[4])

            xm0 = self.simplem0(xm[0])
            xm1 = self.simplem1(xm[1])
            xm2 = self.simplem2(xm[2])
            xm3 = self.simplem3(xm[3])
            xm4 = self.simplem4(xm[4])

            xl = [xl0, xl1, xl2, xl3]#, xl4]
            xm = [xm0, xm1, xm2, xm3]#, xm4]

            #features = torch.cat(xl + xm, 1)  # (B, 2*C, 1, 1)

            features = torch.cat(xl + xm, 1)
            x = self.final(features)  # (B, 2, 1, 1)
            out = x[:, :, 0, 0]
            #print(features.shape)

        if self.fuse == 'simple5':  # classify per lateral and medial region then max pooling

            f = [self.avg(f0[i]) - self.avg(f1[i]) for i in range(n_channels)]   # (B*23, 512, 1, 1) X C
            f = [x.view(B, x.shape[0] // B, x.shape[1], x.shape[2], x.shape[3]) for x in f]  # (B, 23, 512, 1, 1) X C

            xeff = f[0][:, :, :, :, :]

            xl = [y[:, :, :, :, :] for y in f]  # (B, 13, 512, 1, 1) X C

            xl = [torch.max(y, 1)[0] for y in xl]  # (B, 512, 1, 1) X C

            xl0 = self.simplel0(xl[0])
            xl1 = self.simplel1(xl[1])
            xl2 = self.simplel2(xl[2])
            xl3 = self.simplel3(xl[3])
            xl4 = self.simplel4(xl[4])

            xl = [xl0, xl1, xl2, xl3]#, xl4]

            features = torch.cat(xl, 1)
            x = self.final(features)  # (B, 2, 1, 1)
            out = x[:, :, 0, 0]
            #print(features.shape)

        return out, features


if __name__ == '__main__':
    import argparse
    from models.MRPretrained import print_num_of_parameters

    parser = argparse.ArgumentParser()
    parser.backbone = 'alexnet'
    parser.pretrained = False
    parser.n_classes = 2
    parser.fuse = 'simple2'
    parser.repeat = 7

    mr2 = MRPretrainedRepeat(parser)


    #out1 = mr1(torch.rand(4, 3, 224, 224, 23))
    out2 = mr2([torch.rand(4, 7, 224, 224, 23), torch.rand(4, 7, 224, 224, 23)])

    print_num_of_parameters(mr2)