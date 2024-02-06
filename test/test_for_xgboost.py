import time, os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from utils.make_config import *
from engine.lightning_siamese import LitClassification
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from dotenv import load_dotenv
import numpy as np
import argparse
import pandas as pd
from loaders.data_multi import PairedData, PairedData3D
from utils.imagesc import imagesc

# args
load_dotenv('env/.t09')
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='womac3')
parser.add_argument('--bysubject', action='store_true', dest='bysubject', default=False)
#parser.add_argument('--direction', type=str, default='effusion/aeffpainYphiB_effusion/beffpainYphiB', help='a2b or b2a')
parser.add_argument('--direction', type=str, default='effusion/aeffphifixB_effusion/beffphifixB', help='a2b or b2a')
parser.add_argument('--flip', action='store_true', dest='flip', default=False, help='image flip left right')
parser.add_argument('--resize', type=int, default=0)
parser.add_argument('--cropsize', type=int, default=0)
parser.add_argument('--n01', action='store_true', dest='n01', default=True)
parser.add_argument('--trd', type=float, dest='trd', help='threshold of images', default=0)
parser.add_argument('--gray', action='store_true', dest='gray', default=False, help='dont copy img to 3 channel')
parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--port', type=str, default='dummy')

args = parser.parse_args()
args.resize = 384
args.cropsize = 256
args.n01 = True

# CUDA_VISIBLE_DEVICES=1 python train.py --backbone alexnet --fuse simple3
# --direction effusion/aeffpainYphiB_effusion/beffpainYphiB --repeat 5  --prj effpainYphiBsplitM

# Model
args.direction = 'effusion/aeffpainYphiB_effusion/beffpainYphiB'
model_name = 'results/simple5a_0.893.pth'
net = torch.load(model_name)
net = net.eval()

# Data
df = pd.read_csv('/home/ghc/Dropbox/TheSource/scripts/OAI_pipelines/meta/subjects_unipain_womac3.csv')
train_index = [x for x in range(df.shape[0]) if not df['has_moaks'][x]]
eval_index = [x for x in range(df.shape[0]) if df['has_moaks'][x]]
# train_index = range(213, 710)
# eval_index = range(0, 213)
# train_index = range(497)
# eval_index = range(497, 710)
df = pd.read_csv('/media/ExtHDD01/OAI/OAI_extracted/OAI00womac3/OAI00womac3.csv')
labels = [(x,) for x in df.loc[df['SIDE'] == 1, 'P01KPN#EV'].astype(np.int8)]
train_set = PairedData3D(root=os.environ.get('DATASET') + args.dataset + '/full/',
                         path=args.direction, opt=args, labels=labels, mode='train', filenames=True, index=train_index)
eval_set = PairedData3D(root=os.environ.get('DATASET') + args.dataset + '/full/',
                        path=args.direction, opt=args, labels=labels, mode='test', filenames=True, index=eval_index)
full_set = PairedData3D(root=os.environ.get('DATASET') + args.dataset + '/full/',
                        path=args.direction, opt=args, labels=labels, mode='test', filenames=True, index=range(len(labels)))

# Testing
dataloader = DataLoader(full_set, batch_size=1, shuffle=False, num_workers=4, drop_last=False)

# Forward
all_out = []
all_label = []
all_features = []
for i, ((a, b), (label, ), filenames) in enumerate(dataloader):
    print(i)
    if label[0] == 1:
        imgs = (b.cuda(), a.cuda())
    else:
        imgs = (a.cuda(), b.cuda())
    out, features = net(imgs)
    all_out.append(out.detach().cpu().numpy())
    all_label.append(label.detach().cpu().numpy())
    all_features.append(features.detach().cpu().numpy())
    #all_x0.append(x0.detach().cpu().numpy())
    #all_x1.append(x1.detach().cpu().numpy())

all_out = np.concatenate(all_out, 0)
all_label = np.concatenate(all_label, 0)
all_features = np.concatenate(all_features, 0)
#all_x0 = np.concatenate(all_x0, 0)
#all_x1 = np.concatenate(all_x1, 0)

# Metrics
from utils.metrics_classification import ClassificationLoss, GetAUC
loss_function = ClassificationLoss()
metrics = GetAUC()
print('AUC=  ' + str(metrics(all_label, all_out)[0]))

np.save(model_name.replace('.pth', '.npy'), all_features)