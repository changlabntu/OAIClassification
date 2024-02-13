import pytorch_lightning as pl
import time, torch
import numpy as np
import torch.nn as nn
import os
import tifffile as tiff
from torch.optim import lr_scheduler
from engine.base import BaseModel


# from pytorch_lightning.utilities import rank_zero_only

class TripletCenterLoss(nn.Module):
    def __init__(self, margin=0, num_classes=2):
        super(TripletCenterLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.centers = nn.Parameter(torch.randn(num_classes, 512))

    def forward(self, inputs, targets):
        batch_size = inputs.size(0)
        print(batch_size)
        targets_expand = targets.view(batch_size, 1).expand(batch_size, inputs.size(1))
        centers_batch = self.centers.gather(0, targets_expand)  # centers batch

        # compute pairwise distances between input features and corresponding centers
        centers_batch_bz = torch.stack([centers_batch] * batch_size)
        inputs_bz = torch.stack([inputs] * batch_size).transpose(0, 1)
        dist = torch.sum((centers_batch_bz - inputs_bz) ** 2, 2).squeeze()
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        print(dist)
        # for each anchor, find the hardest positive and negative
        mask = targets.expand(batch_size, batch_size).eq(targets.expand(batch_size, batch_size).t())
        dist_ap, dist_an = [], []
        for i in range(batch_size):  # for each sample, we compute distance
            dist_ap.append(dist[i][mask[i]].max())  # mask[i]: positive samples of sample i
            dist_an.append(dist[i][mask[i] == 0].min())  # mask[i]==0: negative samples of sample i

        for x in dist_ap:
            print(x.shape)
        for x in dist_an:
            print(x.shape)
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # generate a new label y
        # compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        # y_i = 1, means dist_an > dist_ap + margin will casuse loss be zero
        loss = self.ranking_loss(dist_an, dist_ap, y)

        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)  # normalize data by batch size
        return loss, prec


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=2, feat_dim=512, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

def lambda_rule(epoch):
    # lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
    n_epochs_decay = 50
    n_epochs = 101
    epoch_count = 0
    lr_l = 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay + 1)
    return lr_l


class LitModel(BaseModel):
    def __init__(self, args, train_loader, eval_loader, net, loss_function, metrics):
        super().__init__(args, train_loader, eval_loader, net, loss_function, metrics)

        self.net.projection = nn.Linear(512, 32).cuda()

        self.triple = nn.TripletMarginLoss()
        self.center = CenterLoss(feat_dim=32)

        # update the parameters for the optimizer
        self.optimizer = self.configure_optimizers()


    def training_step(self, batch, batch_idx=0):
        # training_step defined the train loop. It is independent of forward
        imgs = batch['img']
        labels = batch['labels']

        # repeat part
        if len(imgs) == 2:
            imgs[0] = imgs[0].repeat(1, 3, 1, 1, 1)
            imgs[1] = imgs[1].repeat(1, 3, 1, 1, 1)

        output, features = self.net(imgs)
        loss, _ = self.loss_function(output, labels)

        # contrastive loss
        featuresA = []
        featuresB = []
        if 0:
            if labels[0] == 1:
                featuresA.append(features[1][:1, ::])
                featuresB.append(features[0][:1, ::])
            else:
                featuresA.append(features[0][:1, ::])
                featuresB.append(features[1][:1, ::])

            if labels[1] == 1:
                featuresA.append(features[1][1:, ::])
                featuresB.append(features[0][1:, ::])
            else:
                featuresA.append(features[0][1:, ::])
                featuresB.append(features[1][1:, ::])
        else:
            for i in range(len(labels)):
                if labels[i] == 1:
                    featuresA.append(features[1][i:i+1, ::])
                    featuresB.append(features[0][i:i+1, ::])
                else:
                    featuresA.append(features[0][i:i+1, ::])
                    featuresB.append(features[1][i:i+1, ::])

        featuresA = torch.cat(featuresA, dim=0)
        featuresB = torch.cat(featuresB, dim=0)

        featuresA = self.net.projection(featuresA)
        featuresB = self.net.projection(featuresB)

        loss_t = 0
        loss_t += self.triple(featuresA[0, :], featuresA[1, :], featuresB[0, :])
        loss_t += self.triple(featuresB[0, :], featuresB[1, :], featuresA[0, :])

        loss_center = self.center(torch.cat([f for f in [featuresA, featuresB]], dim=0),
                                  torch.FloatTensor([1] * featuresA.shape[0] + [0] * featuresA.shape[0]).cuda())

        self.log('train_loss', loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        self.log('t', loss_t, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        self.log('c', loss_center, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)

        return 1 * loss #+ loss_t + loss_center

    # @rank_zero_only
    def validation_step(self, batch, batch_idx=0):
        if 1:#self.trainer.global_rank == 0:
            imgs = batch['img']
            labels = batch['labels']

            # repeat part
            if len(imgs) == 2:
                imgs[0] = imgs[0].repeat(1, 3, 1, 1, 1)
                imgs[1] = imgs[1].repeat(1, 3, 1, 1, 1)

            output, features = self.net(imgs)

            loss, _ = self.loss_function(output, labels)
            if not self.args.legacy:
                self.log('val_loss', loss, on_step=False, on_epoch=True,
                         prog_bar=True, logger=True, sync_dist=True)

            # metrics
            self.all_label.append(labels.cpu())
            self.all_out.append(output.cpu().detach())
            self.all_loss.append(loss.detach().cpu().numpy())

            return loss
        else:
            return 0
