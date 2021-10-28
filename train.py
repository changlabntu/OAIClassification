import time, os
from torch.utils.data import DataLoader
import torch.nn as nn
import argparse
from utils.make_config import *
from engine.lightning_classification import LitClassification
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from dotenv import load_dotenv
load_dotenv('.env')


def args_train():
    parser = argparse.ArgumentParser()

    # projects
    parser.add_argument('--prj', type=str, default='', help='name of the project')

    # training modes
    parser.add_argument('--legacy', action='store_true', dest='legacy', default=False, help='legacy pytorch')
    parser.add_argument('--par', dest='parallel', action="store_true", help='run in multiple gpus')
    # training parameters
    parser.add_argument('-e', '--epochs', dest='epochs', default=100, type=int,
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

    # misc
    parser.add_argument('--mode', type=str, default='dummy')
    parser.add_argument('--port', type=str, default='dummy')
    args = parser.parse_args()

    args.not_tracking_hparams = ['mode', 'port', 'parallel', 'epochs', 'legacy', 'prj']

    return args


def train(net, args, train_set, eval_set, loss_function, metrics):
    # Data Loader
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    train_loader.__code__ = ''

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
        tb_logger = pl_loggers.TensorBoardLogger('logs/' + args.prj + '/')
        trainer = pl.Trainer(gpus=1, distributed_backend='ddp',
                             max_epochs=100, progress_bar_refresh_rate=20, logger=tb_logger,
                             accumulate_grad_batches=args.batch_update,
                             #callbacks=[checkpoint_callback],
                             auto_lr_find=True)

        # if lr == 0  run learning rate finder
        if args.lr == 0:
            trainer.tune(ln_classification, train_loader, eval_loader)
        else:
            trainer.fit(ln_classification, train_loader, eval_loader)


if __name__ == "__main__":
    args = args_train()

    # Dataset
    from loaders.OAI_pain_loader import OAIFromFolder as OAIData
    source = ('/media/ExtHDD01/OAI/OAI_extracted/OAI00UniPain3/Npy/SAG_IW_TSE_LEFT_cropped/*',
              '/media/ExtHDD01/OAI/OAI_extracted/OAI00UniPain3/Npy/SAG_IW_TSE_RIGHT_cropped/*')
    train_set = OAIData(index=range(213, 710), source=source, threshold=800)
    eval_set = OAIData(index=range(213), source=source, threshold=800)

    # Networks
    if args.backbone == 'densenet3D':
        from models.densenet3D.MRdensenet3D import MRDenseNet3D
        net = MRDenseNet3D()
    else:
        from models.MRPretrainedTwoStream import MRPretrainedTwoStream
        net = MRPretrainedTwoStream(args_m=args)

    # Performance
    from utils.metrics_classification import ClassificationLoss, GetAUC
    loss_function = ClassificationLoss()
    metrics = GetAUC()

    train(net, args, train_set, eval_set, loss_function, metrics)

    # CUDA_VISIBLE_DEVICES=0 python train.py --backbone vgg11 --fuse cat -w 0
