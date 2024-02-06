import torch
import numpy as np
import tifffile as tiff
from utils.imagesc import imagesc
import glob

net = torch.load('checkpoints/moakseff/m_pain/moaksid/30.pth', map_location='cpu').cuda().eval()
features = net.features
net = torch.load('checkpoints/moakseff/m/100.pth', map_location='cpu').cuda().eval()
features = net.features[:-1]

hmap_all = []
for z in range(23):
    l = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/womac4/full/a/*'))
    w = net.classifier_cat.weight[:, :, 0, 0].detach().cpu()
    w = w.view(w.shape[0], 23, 512).numpy()
    w = w[:, z, :]

    x = tiff.imread(l[1224 - 1])
    #imagesc(x)
    x = (x - x.min()) / (x.max() - x.min())
    x = (x - 0.5) * 2

    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).cuda().float()
    f = features(x).detach().cpu().numpy()

    hmap = np.zeros((f.shape[2], f.shape[3]))

    for c in range(f.shape[1]):
        hmap += f[0, c, :, :] * w[0, c]

    #hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min())
    hmap_all.append(hmap)

hmap_all = np.array(hmap_all)
tiff.imwrite('temp.tif', hmap_all)