import time, os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from utils.make_config import *
from engine.lightning_classification import LitClassification
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from dotenv import load_dotenv
import numpy as np
load_dotenv('.env')

# Model
net = torch.load('checkpoints/20_0.84.pth')

# Data
from loaders.OAI_pain_loader import OAIFromFolder as OAIData
source = ('/media/ExtHDD01/OAI/OAI_extracted/OAI00UniPain3/Npy/SAG_IW_TSE_LEFT_cropped/*',
          '/media/ExtHDD01/OAI/OAI_extracted/OAI00UniPain3/Npy/SAG_IW_TSE_RIGHT_cropped/*')
train_set = OAIData(index=range(213, 710), source=source, threshold=800)
eval_set = OAIData(index=range(213), source=source, threshold=800)
eval_loader = DataLoader(eval_set, batch_size=1, shuffle=False, num_workers=4, drop_last=False)

# Forward
all_out = []
all_label = []
all_x0 = []
all_x1 = []
for i, (idx, (l, r), (label, )) in enumerate(eval_loader):
    imgs = (l.cuda(), r.cuda())
    out, (x0, x1) = net(imgs)
    all_out.append(out.detach().cpu().numpy())
    all_label.append(label.detach().cpu().numpy())
    all_x0.append(x0.detach().cpu().numpy())
    all_x1.append(x1.detach().cpu().numpy())

all_out = np.concatenate(all_out, 0)
all_label = np.concatenate(all_label, 0)
all_x0 = np.concatenate(all_x0, 0)
all_x1 = np.concatenate(all_x1, 0)

# Metrics
from utils.metrics_classification import ClassificationLoss, GetAUC
loss_function = ClassificationLoss()
metrics = GetAUC()
print('AUC=  ' + str(metrics(all_label, all_out)[0]))

if 0:
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import manifold, datasets

    # different between two knees
    X = (all_x1 - all_x0)[:, :, 0, 0]
    y = all_label

    pred = np.argmax(all_out, 1)
    correct = (pred == y).astype(np.int8)

    n_samples, n_features = X.shape

    # t-SNE
    X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(X)

    # Data Visualization
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # Normalize
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(correct[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.show()