import time, os
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torchvision
import argparse
from utils.make_config import *
from engine.lightning_siamese import LitClassification
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter

from dotenv import load_dotenv
import argparse
from loaders.data_multi import PairedData, PairedData3D

from train import split_N_fold
import glob


def this_is_for_something_else():
    images = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/womac4/full/a/*'))
    tini = time.time()
    subjects = sorted(list(set([x.replace('_' + x.split('_')[-1], '') for x in images])))
    sub = dict()
    for s in subjects:
        sub[s] = sorted([x for x in images if x.replace('_' + x.split('_')[-1], '') == s])
    print(time.time() - tini)

def MOAKS_get_vars(categories, ver):
    moaks_summary = pd.read_excel(os.path.join(os.path.expanduser('~'), 'Dropbox',
                                               'TheSource/OAIDataBase/OAI_Labels/MOAKS/KMRI_SQ_MOAKS_variables_summary.xlsx'))
    moaks_variables = moaks_summary.loc[moaks_summary['CATEGORY'].isin(categories), 'VARIABLE']
    l = list(moaks_variables.values)
    return [x.replace('$$', ver) for x in l]

load_dotenv('env/.t09')

# additional arguments for dataset
# Data
parser = argparse.ArgumentParser()  # add_help=False)
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
parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--port', type=str, default='dummy')


args = parser.parse_args()

from loaders.data_multi import MultiData as Dataset
from utils.metrics_classification import ClassificationLoss, GetAUC
metrics = GetAUC()

x = pd.read_csv('env/womac4_moaks.csv')
labels = (x.loc[x['SIDE'] == 'RIGHT']['V$$WOMKP#']).values > (x.loc[x['SIDE'] == 'LEFT']['V$$WOMKP#']).values
labels = [(int(x),) for x in labels]

#pain_moaks = (x.loc[(x['V$$WOMKP#'] > 0) & (~x['READPRJ'].isna())].index.values // 2)

knee_painful = x.loc[(x['V$$WOMKP#'] > 0)].reset_index()
ID_has_eff = x.loc[~x['V$$MEFFWK'].isna()]['ID'].unique()
pmeffid = knee_painful.loc[knee_painful['ID'].isin(ID_has_eff)].index.values

moaks = True
sig = False

for epoch in range(10, 101, 10):
    outall = []
    labelall = []
    if moaks:
        fold = 1
    else:
        fold = 10

    for i in range(fold):
        if moaks:
            train_index = [y for y in range(x.shape[0] // 2) if y not in pmeffid]
            eval_index = [y for y in range(x.shape[0] // 2) if y in pmeffid]
        else:
            train_index, eval_index = split_N_fold(L=x.shape[0] // 2, fold=fold, split=i)#split_five_fold(x, str(i))
        print((min(eval_index), max(eval_index)))

        # net_name = sorted(glob.glob('checkpoints/vgg19cat/m/' + str(i) + '/' + str(epoch) + '*'))[-1]
        net_name = sorted(glob.glob('checkpoints/moakseff/m/' + str(epoch) + '*'))[-1]
        if sig:
            args.direction = 'pain0323/sigA_pain0323/sigB'
            net_name = sorted(glob.glob('checkpoints/moakseff/sig/' + str(epoch) + '*'))[-1]
        eval_set = Dataset(root=os.environ.get('DATASET') + args.dataset + '/full/',
                           path=args.direction, opt=args, labels=labels, mode='test', index=eval_index)

        net = torch.load(net_name, map_location='cpu').cuda().eval()

        for i in range(len(eval_set)):
            print(i)
            batch = eval_set.__getitem__(i)
            imgs = [x.cuda().unsqueeze(0).repeat(1, 3, 1, 1, 1).cuda()[:, :, 64:-64, 64:-64, :] for x in batch['img']]
            #imgs = [x.cuda().unsqueeze(0).repeat(1, 3, 1, 1, 1).cuda()[:, :, :, :, :] for x in batch['img']]
            out = net(imgs)
            outall.append(out[0].cpu().detach().numpy())
            labelall.append(batch['labels'][0])

    # results
    o = np.concatenate(outall, 0)
    l = np.array(labelall)
    print(metrics(l, o))
    if sig:
        np.save('out/sig_'+ str(epoch) + '.npy', o)
    else:
        np.save('out/m_'+ str(epoch) + '.npy', o)
