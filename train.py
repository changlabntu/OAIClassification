import time, os
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torchvision
import argparse
from utils.make_config import *
from engine.lightning_classification import LitClassification
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
import torchio as tio
load_dotenv('env/.t09')


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


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
    parser.add_argument('--lr', '--learning-rate', dest='lr', default=0.001, type=float, help='learning rate')

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


def train(net, args, train_set, eval_set, loss_function, metrics):
    # Data Loader
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True, pin_memory=True)

    train_loader.__code__ = ''

    # preload
    if args.preload:
        tini = time.time()
        print('Preloading...')
        for i, x in enumerate(tqdm(train_loader)):
            pass
        for i, x in enumerate(tqdm(eval_loader)):
            pass
        print('Preloading time: ' + str(time.time() - tini))

    # freezing parameters
    if args.freeze:
        net.par_freeze = [y for x in [list(x.parameters()) for x in [getattr(net, 'features')]] for y in x]
    else:
        net.par_freeze = []

    """ cuda """
    if args.legacy:
        net = net.cuda()
        net = nn.DataParallel(net)

    """ training class """
    ln_classification = LitClassification(args=args,
                                          train_loader=train_loader,
                                          eval_loader=eval_loader,
                                          net=net,
                                          loss_function=loss_function,
                                          metrics=metrics)

    """ vanilla pytorch mode"""
    if args.legacy:
        # Use pytorch without lightning
        ln_classification.overall_loop()
    else:
        # Use pytorch lightning for training, you can ignore it
        checkpoint_callback = ModelCheckpoint(
            dirpath='checkpoints/' + args.prj + '/',
            filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}',
            verbose=False,
            monitor='val_loss',
            mode='min'
        )

        # we can use loggers (from TensorBoard) to monitor the progress of training
        tb_logger = pl_loggers.TensorBoardLogger('logs/' + args.prj + '/', default_hp_metric=False)
        trainer = pl.Trainer(gpus=-1, strategy='ddp',
                             max_epochs=args.epochs, progress_bar_refresh_rate=20, logger=tb_logger,
                             accumulate_grad_batches=args.batch_update,
                             callbacks=[checkpoint_callback],
                             auto_lr_find=True)

        # if lr == 0  run learning rate finder
        if args.lr == 0:
            trainer.tune(ln_classification, train_loader, eval_loader)
        else:
            trainer.fit(ln_classification, train_loader, eval_loader)


def split_five_fold(x, split):
    N = x.shape[0] // 10
    split5 = (list(range(N)), list(range(N, 2 * N)), list(range(2 * N, 3 * N)), list(range(3 * N, 4 * N)),
              list(range(4 * N, 5 * N)))
    if split == '0':
        eval_index = split5[0]
        train_index = split5[1] + split5[2] + split5[3] + split5[4]
    elif split == '1':
        eval_index = split5[1]
        train_index = split5[0] + split5[2] + split5[3] + split5[4]
    elif split == '2':
        eval_index = split5[2]
        train_index = split5[0] + split5[1] + split5[3] + split5[4]
    elif split == '3':
        eval_index = split5[3]
        train_index = split5[0] + split5[1] + split5[2] + split5[4]
    elif split == '4':
        eval_index = split5[4]
        train_index = split5[0] + split5[1] + split5[2] + split5[3]
    return train_index, eval_index


def split_moaks(x, split):
    moaks_id = x.loc[(~x['READPRJ'].isna())]['ID'].unique()
    eval_index = [y // 2 for y in (x.loc[(~x['ID'].isin(moaks_id)) & (x['SIDE'] == 'LEFT')]).index.values]
    train_index_all = [y // 2 for y in (x.loc[(x['ID'].isin(moaks_id)) & (x['SIDE'] == 'LEFT')]).index.values]
    N = len(train_index_all) // 5
    if split == '0':
        train_index = train_index_all[N:]
    if split == '1':
        train_index = train_index_all[:N] + train_index_all[2 * N:]
    if split == '2':
        train_index = train_index_all[:2 * N] + train_index_all[3 * N:]
    if split == '3':
        train_index = train_index_all[:3 * N] + train_index_all[4 * N:]
    if split == '4':
        train_index = train_index_all[:4 * N]
    return train_index, eval_index


if __name__ == "__main__":
    parser = args_train()

    from dotenv import load_dotenv
    import argparse
    from loaders.data_multi import PairedData, PairedData3D

    # additional arguments for dataset
    # Data
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

    args = parser.parse_args()

    args.resize = 384
    args.cropsize = 256

    #df = pd.read_csv('/home/ghc/Dropbox/TheSource/scripts/OAI_pipelines/meta/subjects_unipain_womac3.csv')
    #train_index = [x for x in range(df.shape[0]) if not df['has_moaks'][x]]
    #eval_index = [x for x in range(df.shape[0]) if df['has_moaks'][x]]
    #train_index = range(213, 710)
    #eval_index = range(0, 213)
    #train_index = range(497)
    #eval_index = range(497, 710)

    #df = pd.read_csv('/media/ExtHDD01/OAI/OAI_extracted/OAI00womac3/OAI00womac3.csv')
    #labels = [(x, ) for x in df.loc[df['SIDE'] == 1, 'P01KPN#EV'].astype(np.int8)]

    #x = pd.read_csv('/home/ghc/Dropbox/TheSource/scripts/OAI_API/meta/womac4min0.csv')
    #labels = x.loc[x['SIDE'] == 'RIGHT']['V$$WOMKPR'] > x.loc[x['SIDE'] == 'RIGHT']['V$$WOMKPL']
    x = pd.read_csv('env/womac4_moaks.csv')
    labels = (x.loc[x['SIDE'] == 'RIGHT']['V$$WOMKP#']).values > (x.loc[x['SIDE'] == 'LEFT']['V$$WOMKP#']).values
    labels = [(int(x), ) for x in labels]
    has_moaks = (~x['READPRJ'].isna())

    if args.split is not None:
        args.prj = args.prj + '/' + args.split + '/'
        train_index, eval_index = split_moaks(x, args.split)

    from loaders.data_multi import MultiData as Dataset
    train_set = Dataset(root=os.environ.get('DATASET') + args.dataset + '/full/',
                        path=args.direction, opt=args, labels=labels, mode='train', index=train_index)
    print(len(train_set))
    eval_set = Dataset(root=os.environ.get('DATASET') + args.dataset + '/full/',
                       path=args.direction, opt=args, labels=labels, mode='test', index=eval_index)
    print(len(eval_set))

# Networks
if args.backbone == 'densenet3D':
    from models.densenet3D.MRdensenet3D import MRDenseNet3D
    net = MRDenseNet3D()
else:
    if args.repeat > 0:
        from models.MRPretrainedRepeat import MRPretrainedRepeat
        net = MRPretrainedRepeat(args_m=args)
    else:
        from models.MRPretrainedSiamese import MRPretrainedSiamese
        net = MRPretrainedSiamese(args_m=args)

    #from models.gfnet.gfnet0112 import GFSiamnese
    #net = GFSiamnese()

#from models.BiT import BITSiamnese
#net = BITSiamnese()

# Performance
from utils.metrics_classification import ClassificationLoss, GetAUC
loss_function = ClassificationLoss()
metrics = GetAUC()

os.makedirs(os.path.join('checkpoints', args.prj), exist_ok=True)

args.not_tracking_hparams = ['mode', 'port', 'parallel', 'epochs', 'legacy']
train(net, args, train_set, eval_set, loss_function, metrics)

# CUDA_VISIBLE_DEVICES=1 python train.py --backbone vgg11 --fuse cat -w 0 --prj womac4check
# CUDA_VISIBLE_DEVICES=0 python train.py --backbone alexnet --fuse simple3 --direction effusion/aeffpainYphiB_effusion/beffpainYphiB --repeat 5