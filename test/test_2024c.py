import time, os, glob
import torch
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn
from utils.make_config import *
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from dotenv import load_dotenv
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import tifffile as tiff
from utils.metrics_classification import ClassificationLoss, GetAUC
from loaders.data_multi import MultiData as Dataset
from dotenv import load_dotenv
import argparse
from loaders.data_multi import PairedData, PairedData3D
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

metrics = GetAUC()
load_dotenv('.env')


def args_train():
    parser = argparse.ArgumentParser()

    # projects
    parser.add_argument('--prj', type=str, default='', help='name of the project')

    # training modes
    parser.add_argument('--legacy', action='store_true', dest='legacy', default=False, help='legacy pytorch')
    parser.add_argument('--par', dest='parallel', action="store_true", help='run in multiple gpus')
    # training parameters
    parser.add_argument('-e', '--epochs', dest='epochs', default=101, type=int,
                        help='number of epochs')
    parser.add_argument('-b', '--batch-size', dest='batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--bu', '--batch-update', dest='batch_update', default=1, type=int, help='batch to update')
    parser.add_argument('--lr', '--learning-rate', dest='lr', default=0.0005, type=float, help='learning rate')

    parser.add_argument('-w', '--weight-decay', dest='weight_decay', default=0.005, type=float, help='weight decay')
    # optimizer
    parser.add_argument('--op', dest='op', default='sgd', type=str, help='type of optimizer')

    # models
    parser.add_argument('--fuse', dest='fuse', default='')
    parser.add_argument('--backbone', dest='backbone', default='vgg11')
    parser.add_argument('--pretrained', dest='pretrained', default=True)
    parser.add_argument('--freeze', action='store_true', dest='freeze', default=False)
    parser.add_argument('--classes', dest='n_classes', default=2, type=int, help='number of classes')
    parser.add_argument('--repeat', type=int, default=0, help='repeat the encoder N time')

    # misc
    parser.add_argument('--mode', type=str, default='dummy')
    parser.add_argument('--port', type=str, default='dummy')

    parser.add_argument('--dataset', type=str, default='womac4')
    parser.add_argument('--load3d', action='store_true', dest='load3d', default=True)
    parser.add_argument('--direction', type=str, default='a_b', help='a2b or b2a')
    parser.add_argument('--flip', action='store_true', dest='flip', default=False, help='image flip left right')
    parser.add_argument('--resize', type=int, default=0)
    parser.add_argument('--cropsize', type=int, default=0)
    parser.add_argument('--n01', action='store_true', dest='n01', default=False)
    parser.add_argument('--trd', type=float, dest='trd', help='threshold of images', default=0)
    parser.add_argument('--preload', action='store_true', help='preload the data once to cache')
    parser.add_argument('--split', type=str, default=None)
    parser.add_argument('--fold', type=int, default=None)

    return parser


def list_to_tensor(ax, repeat=True, norm=None):
    ax = torch.cat([torch.from_numpy(x / 1).unsqueeze(2) for x in ax], 2).unsqueeze(0).unsqueeze(1)
    # normalize ax to -1 to 1
    ax = normalize(ax, norm=norm)
    if repeat:
        ax = ax.repeat(1, 3, 1, 1, 1)
    return ax


def normalize(x, norm=None):
    if norm == '01':
        x = (x - x.min()) / (x.max() - x.min())
    elif norm == '11':
        x = (x - x.min()) / (x.max() - x.min())
        x = x * 2 - 1
    else:
        x = (x - x.min()) / (x.max() - x.min())
        if len(x.shape) > 4:
            all = []
            for i in range(x.shape[4]):
                #all.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(x[:, :, :, :, i]))
                all.append(transforms.Normalize(0.485, 0.229)(x[:, :, :, :, i]))
            x = torch.stack(all, 4)
        else:
            #x = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(x)
            x = transforms.Normalize(0.485, 0.229)(x)
    return x


def flip_by_label(x, label):
    y = []
    for i in range(x.shape[0]):
        if label[i] == 1:
            y.append(torch.flip(x[i, :], [0]))
        else:
            y.append(x[i, :])
    return torch.stack(y, 0)


def manual_classifier_auc(xA, xB, labels, L):
    out = net.classifier((xB - xA).unsqueeze(2).unsqueeze(3).cuda()).detach().cpu()[:, :, 0, 0]
    print('AUC0=  ' + str(metrics(labels[:L], flip_by_label((out[:L, :]), 1 - labels[:L]))[0]))
    out = net.classifier(-(xB - xA).unsqueeze(2).unsqueeze(3).cuda()).detach().cpu()[:, :, 0, 0]
    print('AUC1=  ' + str(metrics(labels[:L], flip_by_label((out[:L, :]), labels[:L]))[0]))


def test_eval_set():
    # Forward
    outAB = []
    xA = []
    xB = []

    #for data in tqdm(eval_loader):
    print('testing over the eval set...')
    for i in tqdm(range(len(eval_set))):
        data = eval_set.__getitem__(i)
        (a, b) = data['img']
        a = a.repeat(1, 3, 1, 1, 1)
        b = b.repeat(1, 3, 1, 1, 1)
        imgs = (b.cuda(), a.cuda())
        out, (xB_, xA_) = net(imgs)
        outAB.append(out.detach().cpu())
        del out

        xA.append(xA_.detach().cpu())
        xB.append(xB_.detach().cpu())

    (outAB, xA, xB) = (torch.cat(x) for x in (outAB, xA, xB))
    # AUC
    manual_classifier_auc(xA, xB, labels, L=667)

def test_diffusion():
    print('testing over the diffusion set...')
    xAD = []
    xBD = []
    xD = []
    for i in tqdm(range(len(dlist) // 23)):
        (ax, bx, dx) = ([tiff.imread(x) for x in y[i * 23:(i + 1) * 23]] for y in (alist, blist, dlist))
        (ax, bx, dx) = (list_to_tensor(x, repeat=True, norm=norm_method).float() for x in (ax, bx, dx))

        imgs = (ax.cuda(), bx.cuda())
        out, (xA_, xB_) = net(imgs)
        xAD.append(xA_.detach().cpu())
        xBD.append(xB_.detach().cpu())
        del xA_, xB_, out

        imgs = (dx.cuda(), bx.cuda())
        out, (xD_, xB_) = net(imgs)
        xD.append(xD_.detach().cpu())
        del xD_, xB_, out

    (xAD, xBD, xD) = (torch.cat(x) for x in (xAD, xBD, xD))
    # AUC
    manual_classifier_auc(xAD, xBD, labels, L=200)
    # Probability of A vs B
    outADD = nn.Softmax(dim=1)(net.classifier((xAD - xD).unsqueeze(2).unsqueeze(3).cuda()).detach().cpu()[:, :, 0, 0])


def test_gan():
    # GAN data
    log_root = '/media/ExtHDD01/logs/womac4/'
    ckpt = 'global1_cut1/nce4_down4_0011_ngf32_proj32_zcrop16/checkpoints/net_g_model_epoch_200.pth'
    net_gan = torch.load(log_root + ckpt, map_location='cpu').cuda()
    alpha = 1
    gpu = True


    def get_xy(ax):
        if gpu:
            ax = ax.cuda()
        mask = net_gan(ax, alpha=alpha)['out0'].detach().cpu()
        mask = nn.Sigmoid()(mask)
        ax = torch.multiply(mask, ax.detach().cpu())
        return ax, mask


    print('testing over the GAN set...')
    all_mask = []
    all_ax = []
    all_bx = []
    all_gx = []

    for i in tqdm(range(len(dlist) // 23)):
        ax = [tiff.imread(x) for x in alist[i * 23:(i + 1) * 23]]
        bx = [tiff.imread(x) for x in blist[i * 23:(i + 1) * 23]]
        ax = list_to_tensor(ax, repeat=True, norm=norm_method).float()
        bx = list_to_tensor(bx, repeat=True, norm=norm_method).float()

        gx, mask = get_xy(ax.permute(4, 1, 2, 3, 0)[:,:1,:,:,0])
        gx = gx.permute(1, 2, 3, 0).unsqueeze(0).repeat(1, 3, 1, 1, 1)
        mask = mask.permute(1, 2, 3, 0).unsqueeze(0).repeat(1, 3, 1, 1, 1)
        gx_mask = torch.multiply(ax, mask)
        all_ax.append(ax)
        all_bx.append(bx)
        all_gx.append(gx)
        all_mask.append(mask)

    xAG = []
    xBG = []
    xG = []
    for i in tqdm(range(len(dlist) // 23)):
        imgs = (all_ax[i].cuda(), all_bx[i].cuda())
        out, (xAG_, xBG_) = net(imgs)
        xAG.append(xAG_.detach().cpu())
        xBG.append(xBG_.detach().cpu())
        del out, xAG_, xBG_
        #out_ab.append(out.detach().cpu())

        imgs = (all_gx[i].cuda(), all_gx[i].cuda())
        out, (xG_, _) = net(imgs)
        xG.append(xG_.detach().cpu())
        del out, xG_
        #out_ag.append(out.detach().cpu())

    (xAG, xBG, xG) = (torch.cat(x) for x in (xAG, xBG, xG))


    # Probability of A vs B
    manual_classifier_auc(xAG, xBG, labels, L=200)
    outAGG = nn.Softmax(dim=1)(net.classifier((xAG - xG).unsqueeze(2).unsqueeze(3).cuda()).detach().cpu()[:, :, 0, 0])

    #out_ag_mask = []
    #imgs = (ax.cuda(), gx_mask.cuda())
    #out, (xB, xA) = net(imgs)
    #out_ac_mask.append(out.detach().cpu())


    plt.scatter(outADD[:,0],outAGG[:,0]);plt.xlim(0,1);plt.ylim(0,1);plt.show()


parser = args_train()
args = parser.parse_args()

load_dotenv('env/.t09b')
x = pd.read_csv('env/womac4_moaks.csv')
labels = (x.loc[x['SIDE'] == 'RIGHT']['V$$WOMKP#']).values > (x.loc[x['SIDE'] == 'LEFT']['V$$WOMKP#']).values


# Data
eval_set = Dataset(root=os.environ.get('DATASET') + args.dataset + '/val/',
                   path='ap_bp', opt=args, labels=[(int(x),) for x in labels], mode='test', index=None)

eval_loader = DataLoader(eval_set, batch_size=1, shuffle=False, num_workers=4, drop_last=False)

# Extra Raw Data
# diffusion data
data_root = '/home/ghc/Dataset/paired_images/womac4/fullXXX/'
alist = sorted(glob.glob(data_root + 'a/*'))
blist = sorted(glob.glob(data_root + 'b/*'))
dlist = sorted(glob.glob(data_root + '003b/*'))


# Model
ckpt_path = '/media/ExtHDD01/logscls/'
#ckpt = sorted(glob.glob('/home/ghc/Dropbox/TheSource/scripts/OAI_classification/checkpoints/siamese/vgg19max2/*.pth'))[12-2]
#ckpt = sorted(glob.glob('/home/ghc/Dropbox/TheSource/scripts/OAI_classification/checkpoints/siamese/alexmax2/*.pth'))[10-2]
#ckpt = sorted(glob.glob('/home/ghc/Dropbox/TheSource/scripts/OAI_classification/checkpoints/alexmaxA/*.pth'))[9-2]
#ckpt = sorted(glob.glob('/home/ghc/Dropbox/TheSource/scripts/OAI_classification/checkpoints/alexmaxA%/*.pth'))[7-2]
#ckpt = sorted(glob.glob(ckpt_path + 'siamese/vgg11max2%test2/checkpoints/*.pth'))[14-2]
#ckpt = sorted(glob.glob(ckpt_path + 'siamese/vgg19max2/checkpoints/*.pth'))[12-2]
#ckpt = sorted(glob.glob(ckpt_path + 'contrastive/vgg11max2_n01_cls/checkpoints/*.pth'))[12]
#ckpt = sorted(glob.glob(ckpt_path + 'siamesetwo/vgg11max2_n01/checkpoints/*.pth'))[22]

for i in [8, 9]:
    ckpt = sorted(glob.glob(ckpt_path + 'siamese/vgg19max2/checkpoints/*.pth'))[i]
    net = torch.load(ckpt, map_location='cpu')
    net = net.cuda()
    net = net.eval()

    # NORMALIZATION
    args.n01 = False
    norm_method = None
    test_eval_set()


