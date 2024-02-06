import torch
import numpy as np
import tifffile as tiff
from utils.imagesc import imagesc
import glob

#net = torch.load('checkpoints/moakseff/m_pain/moaksid/100.pth', map_location='cpu').cuda()
net = torch.load('checkpoints/moaksid/mmax/moaksid/50.pth', map_location='cpu').cuda()
features = net.features[:-1]


l = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/womac4/full/a/*'))
w = net.classifier_cat.weight[:, :, 0, 0].detach().cpu().numpy()
#w = w[:, 512 * z:512 * (z + 1)]

x = tiff.imread(l[9256 - 1])
x = (x - x.min()) / (x.max() - x.min())
x = (x - 0.5) * 2

x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).cuda().float()
f = features(x).detach().cpu().numpy()

hmap = np.zeros((f.shape[2], f.shape[3]))

for c in range(f.shape[1]):
    hmap += f[0, c, :, :] * w[0, c]

imagesc(hmap)
