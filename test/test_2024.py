import time, os, glob
import torch
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn
from utils.make_config import *
from engine.lightning_siamese import LitClassification
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

    return parser

from dotenv import load_dotenv
import argparse
from loaders.data_multi import PairedData, PairedData3D
load_dotenv('env/.t09b')
x = pd.read_csv('env/womac4_moaks.csv')
labels = (x.loc[x['SIDE'] == 'RIGHT']['V$$WOMKP#']).values > (x.loc[x['SIDE'] == 'LEFT']['V$$WOMKP#']).values

parser = args_train()
# Model
ckpt = sorted(glob.glob('/home/ghc/Dropbox/TheSource/scripts/OAI_classification/checkpoints/siamese/vgg19max2/*.pth'))[12-2]
#ckpt = sorted(glob.glob('/home/ghc/Dropbox/TheSource/scripts/OAI_classification/checkpoints/siamese/alexmax2/*.pth'))[10-2]
#ckpt = sorted(glob.glob('/home/ghc/Dropbox/TheSource/scripts/OAI_classification/checkpoints/alexmaxA/*.pth'))[9-2]
#ckpt = sorted(glob.glob('/home/ghc/Dropbox/TheSource/scripts/OAI_classification/checkpoints/alexmaxA%/*.pth'))[7-2]
net = torch.load(ckpt, map_location='cpu')
net = net.cuda()
net = net.train()

# Data
from loaders.data_multi import MultiData as Dataset

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

args = parser.parse_args()

eval_set = Dataset(root=os.environ.get('DATASET') + args.dataset + '/fullXXX/',
                   path='ap_bp', opt=args, labels=[(int(x),) for x in labels], mode='test', index=None)

eval_loader = DataLoader(eval_set, batch_size=1, shuffle=False, num_workers=4, drop_last=False)

if 0:
    # Forward
    all_out = []
    all_label = []
    all_xA = []
    all_xB = []
    for data in tqdm(eval_loader):
        (a, b) = data['img']
        label = data['labels']
        a = a.repeat(1, 3, 1, 1, 1)
        b = b.repeat(1, 3, 1, 1, 1)
        if label[0] == 1:
            imgs = (b.cuda(), a.cuda())
            out, (xB, xA) = net(imgs)
        else:
            imgs = (a.cuda(), b.cuda())
            out, (xA, xB) = net(imgs)

        all_out.append(out.detach().cpu().numpy())
        all_label.append(label[0].detach().cpu().numpy())
        all_xA.append(xA.detach().cpu())
        all_xB.append(xB.detach().cpu())

    all_out = np.concatenate(all_out, 0)
    all_label = np.concatenate(all_label, 0)
    all_xA = torch.cat(all_xA, 0)
    all_xB = torch.cat(all_xB, 0)

    if len(all_xA.shape) < 4:
        all_xA = all_xA.unsqueeze(2).unsqueeze(3)
        all_xB = all_xB.unsqueeze(2).unsqueeze(3)



    loss_function = ClassificationLoss()
    # AUC from classifier

    print('AUC=  ' + str(metrics(all_label[:667], all_out[:667, :])[0]))

    # manual calculated classifier
    flip = ((torch.from_numpy(labels)) / 1 - 0.5) * 2
    w = torch.multiply((all_xB - all_xA).squeeze(), flip.unsqueeze(1).repeat(1, all_xA.shape[1]))
    out = net.classifier(w.unsqueeze(2).unsqueeze(3).cuda()).detach().cpu()[:, :, 0, 0]
    print('AUC=  ' + str(metrics(all_label[:667], out[:667, :])[0]))

    # By difference to average(No Pain)
    all_xB_mean = torch.mean(all_xB[667:, ::], 0).unsqueeze(0).repeat(667 * 2, 1, 1, 1)
    new_test = torch.cat([all_xA[:667, ::], all_xB[:667, ::]], 0)
    out = net.classifier((new_test - all_xB_mean).cuda()).detach().cpu()[:, :, 0, 0]
    print('AUC=  ' + str(metrics(np.array([0]*667 + [1] * 667), out)[0]))


    # SVM
    X_train = torch.cat([all_xA[667:,:,0,0], all_xB[667:,:,0,0]], 0).numpy()
    y_train = np.array([0]*1558 + [1] * 1558)
    X_test = torch.cat([all_xA[:667,:,0,0], all_xB[:667,:,0,0]], 0).numpy()
    y_test =np.array([0]*667 + [1] * 667)
    # Create a SVM Classifier
    clf = svm.SVC(kernel='poly', probability=True)  # Linear Kernel
    # Train the model using the training sets
    clf.fit(X_train, y_train)
    # Predict the response for test dataset
    y_pred = clf.predict(X_test)
    # Evaluate accuracy
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    # Compute AUC Score
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(['Acc contra: ' + "{0:0.3f}".format(accuracy_score(y_test, y_pred)) +
           ' AUC contra: ' + "{0:0.3f}".format(roc_auc_score(y_test, y_pred_proba))])


# umap
if 0:
    import umap
    import matplotlib.pyplot as plt
    reducer = umap.UMAP()
    e = reducer.fit_transform(torch.cat([all_xA, all_xB], 0).squeeze())
    L = 2225
    plt.figure(figsize=(10, 8))
    plt.scatter(e[:667, 0], e[:667, 1], s=0.5 * np.ones(667))
    plt.scatter(e[L:L + 667, 0], e[L:L + 667, 1], s=0.5 * np.ones(667))
    #plt.scatter(e[667:2225, 0], e[667:2225, 1], s=0.5 * np.ones(1558))
    #plt.scatter(e[L+667:L + 2225, 0], e[L+667:L + 2225, 1], s=0.5 * np.ones(1558))
    plt.show()


def list_to_tensor(ax, repeat=True):
    ax = torch.cat([torch.from_numpy(x / 1).unsqueeze(2) for x in ax], 2).unsqueeze(0).unsqueeze(1)
    if repeat:
        ax = ax.repeat(1, 3, 1, 1, 1)
    ax = ax / ax.max()
    return ax

# diffusion data
import matplotlib.pyplot as plt
data_root = '/home/ghc/Dataset/paired_images/womac4/fullXXX/'
alist = sorted(glob.glob(data_root + 'ap/*'))
blist = sorted(glob.glob(data_root + 'bp/*'))
clist = sorted(glob.glob(data_root + '003b/*'))

out_cb = []
out_ab = []
out_ac = []
for i in tqdm(range(len(clist) // 23)):
    ax = [tiff.imread(x) for x in alist[i * 23:(i + 1) * 23]]
    bx = [tiff.imread(x) for x in blist[i * 23:(i + 1) * 23]]
    cx = [tiff.imread(x) for x in clist[i * 23:(i + 1) * 23]]
    ax = list_to_tensor(ax).float()
    bx = list_to_tensor(bx).float()
    cx = list_to_tensor(cx).float()

    #ax = ax[:, :, 64:-64, 64:-64, :]
    #bx = bx[:, :, 64:-64, 64:-64, :]
    #cx = cx[:, :, 64:-64, 64:-64, :]

    if labels[i]:
        imgs = (bx.cuda(), cx.cuda())
    else:
        imgs = (cx.cuda(), bx.cuda())
    out, (xB, xA) = net(imgs)
    out_cb.append(out.detach().cpu())

    if labels[i]:
        imgs = (bx.cuda(), ax.cuda())
    else:
        imgs = (ax.cuda(), bx.cuda())
    out, (xB, xA) = net(imgs)
    out_ab.append(out.detach().cpu())

    if labels[i]:
        imgs = (cx.cuda(), ax.cuda())
    else:
        imgs = (ax.cuda(), cx.cuda())
    out, (xB, xA) = net(imgs)
    out_ac.append(out.detach().cpu())


out_ab = torch.cat(out_ab, 0)
out_cb = torch.cat(out_cb, 0)
out_ac = torch.cat(out_ac, 0)

ab = nn.Softmax(dim=1)(out_ab)
cb = nn.Softmax(dim=1)(out_cb)
ac = nn.Softmax(dim=1)(out_ac)

print('AUC=  ' + str(metrics(labels[:200], out_ab)[0]))

ab2 = np.array([ab[i, int(labels[i] / 1)] for i in range(200)])
cb2 = np.array([cb[i, int(labels[i] / 1)] for i in range(200)])
ac2 = np.array([ac[i, int(labels[i] / 1)] for i in range(200)])


# GAN data
log_root = '/media/ExtHDD01/logs/womac4/'
ckpt = 'global1_cut1/nce4_down4_0011_ngf32_proj32_zcrop16/checkpoints/net_g_model_epoch_200.pth' # 0.92
net_gan = torch.load(log_root + ckpt, map_location='cpu').cuda()
alpha = 1
gpu = True


def get_xy(ax):
    if gpu:
        ax = ax.cuda()
    mask = net_gan(ax, alpha=alpha)['out0'].detach().cpu()
    mask = nn.Sigmoid()(mask)
    ax = torch.multiply(mask, ax.detach().cpu())
    return ax

out_cb = []
out_ab = []
out_ac = []
for i in tqdm(range(len(clist) // 23)):
    ax = [tiff.imread(x) for x in alist[i * 23:(i + 1) * 23]]
    bx = [tiff.imread(x) for x in blist[i * 23:(i + 1) * 23]]
    ax = list_to_tensor(ax, repeat=True).float()
    bx = list_to_tensor(bx, repeat=True).float()

    cx = get_xy(ax.permute(4, 1, 2, 3, 0)[:,:1,:,:,0]).permute(1, 2, 3, 0).unsqueeze(0).repeat(1, 3, 1, 1, 1)

    #ax = ax[:, :, 64:-64, 64:-64, :]
    #bx = bx[:, :, 64:-64, 64:-64, :]
    #cx = cx[:, :, 64:-64, 64:-64, :]

    if labels[i]:
        imgs = (bx.cuda(), cx.cuda())
    else:
        imgs = (cx.cuda(), bx.cuda())
    out, (xB, xA) = net(imgs)
    out_cb.append(out.detach().cpu())

    if labels[i]:
        imgs = (bx.cuda(), ax.cuda())
    else:
        imgs = (ax.cuda(), bx.cuda())
    out, (xB, xA) = net(imgs)
    out_ab.append(out.detach().cpu())

    if labels[i]:
        imgs = (cx.cuda(), ax.cuda())
    else:
        imgs = (ax.cuda(), cx.cuda())
    out, (xB, xA) = net(imgs)
    out_ac.append(out.detach().cpu())

out_ab = torch.cat(out_ab, 0)
out_cb = torch.cat(out_cb, 0)
out_ac = torch.cat(out_ac, 0)
ab = nn.Softmax(dim=1)(out_ab)
cb = nn.Softmax(dim=1)(out_cb)
ac = nn.Softmax(dim=1)(out_ac)
ab3 = np.array([ab[i, int(labels[i] / 1)] for i in range(200)])
cb3 = np.array([cb[i, int(labels[i] / 1)] for i in range(200)])
ac3 = np.array([ac[i, int(labels[i] / 1)] for i in range(200)])