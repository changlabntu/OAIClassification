import numpy as np
from collections import Counter
import glob, os, time

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import pytorch_lightning as pl


if 0:  # not in use
    def womac5():
        l = [x.split('/')[-1].split('.')[0] for x in sorted(glob.glob('/media/ExtHDD01/Dataset/OAI_extracted/OAI01WOMACDiff5Npy/SAG_IW_TSE_LEFT_cropped/*'))]
        r = [x.split('/')[-1].split('.')[0] for x in sorted(glob.glob('/media/ExtHDD01/Dataset/OAI_extracted/OAI01WOMACDiff5Npy/SAG_IW_TSE_RIGHT_cropped/*'))]
        lr = sorted(list(set(l) & set(r)))
        df = pd.read_csv('/home/ghc/Dropbox/TheSource/scripts/OAI_utilities/extracted_oai_info/V01_womac_diff5.csv')
        mri_right = ['/media/ghc/GHc_data1/OAI_extracted/OAI01WOMACDiff5Npy/SAG_IW_TSE_RIGHT_cropped/' + x + '.npy' for x in lr]
        mri_left = ['/media/ghc/GHc_data1/OAI_extracted/OAI01WOMACDiff5Npy/SAG_IW_TSE_LEFT_cropped/' + x + '.npy' for x in lr]
        labels = df.loc[df['ID'].isin(lr), 'label'].values.astype(np.int64)
        return mri_right, mri_left, labels


def load_OAI_var():
    all_path = os.path.join(os.path.expanduser('~'), 'Dropbox') + '/TheSource/OAIDataBase/OAI_Labels/'
    print(all_path)
    all_var = glob.glob(all_path + '*.npy')
    all_var.sort()
    v = dict()
    for var in all_var:
        name = var.split('/')[-1].split('.')[0]
        v[name] = np.load(var, allow_pickle=True)

    return v


def get_OAI_pain_labels():
    Labels = dict()
    v = load_OAI_var()

    FreL_uni = v['fre_pain_l'][np.searchsorted(v['ID_main'], v['ID_uni_fre_pain'])]
    FreR_uni = v['fre_pain_r'][np.searchsorted(v['ID_main'], v['ID_uni_fre_pain'])]

    quality = np.logical_and(v['fail_uni_l'] == 0, v['fail_uni_r'] == 0)

    select_condition = 'np.logical_and(quality == 1, abs(v["WOMP_uni_l"]-v["WOMP_uni_r"]) >= 3)'
    pick = eval(select_condition)

    Labels['label'] = FreR_uni[pick]
    Labels['ID_selected'] = v['ID_uni_fre_pain'][pick]

    return Labels['label']


class OAIFromFile(Dataset):
    def __init__(self, index):
        super().__init__()
        self.mri_left = np.load('/media/ExtHDD01/Dataset/OAI_uni_pain/'
                                'unilateral_pain_left_womac3.npy')
        self.mri_right = np.load('/media/ExtHDD01/Dataset/OAI_uni_pain/'
                                'unilateral_pain_right_womac3.npy')
        self.labels = get_OAI_pain_labels()
        self.index = index

    def __len__(self):
        return self.labels[self.index].shape[0]

    def __getitem__(self, idx):
        index = self.index[idx]
        label = self.labels[index]

        l = self.mri_left[index, ::]
        r = self.mri_right[index, ::]  # (1, 224, 224, 23)

        # copy channel
        l = np.concatenate([l] * 3, 0)
        r = np.concatenate([r] * 3, 0)

        l = torch.from_numpy(l).type(torch.FloatTensor)
        r = torch.from_numpy(r).type(torch.FloatTensor)

        oshape = l.shape
        l = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(l.view(l.shape[0], l.shape[1], -1))
        r = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(r.view(l.shape[0], l.shape[1], -1))
        l = l.view(oshape)
        r = r.view(oshape)

        return idx, (l, r), (label, )


class OAIFromFolder(Dataset):
    def __init__(self, index, source, threshold):
        super().__init__()
        self.mri_left = sorted(glob.glob(source[0]))
        self.mri_right = sorted(glob.glob(source[1]))
        self.labels = get_OAI_pain_labels()
        self.index = index
        self.threshold = threshold

    def __len__(self):
        return self.labels[self.index].shape[0]

    def __getitem__(self, idx):
        index = self.index[idx]
        label = self.labels[index]

        l = np.expand_dims(np.load(self.mri_left[index]), 0)
        r = np.expand_dims(np.load(self.mri_right[index]), 0)
        l[l >= self.threshold] = self.threshold
        r[r >= self.threshold] = self.threshold
        l = l / l.max()
        r = r / r.max()
        l = l.astype(np.float32)
        r = r.astype(np.float32)

        # copy channel
        l = np.concatenate([l] * 3, 0)
        r = np.concatenate([r] * 3, 0)
        l = torch.from_numpy(l)
        r = torch.from_numpy(r)

        return idx, (l, r), (label, )
