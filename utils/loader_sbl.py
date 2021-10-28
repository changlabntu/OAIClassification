import numpy as np
from torch.utils.data import Dataset


class LoaderSBL(Dataset):
    def __init__(self, X, clinical, labels):
        self.X = X
        self.clinical = clinical
        self.labels = labels

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):

        X = self.X[idx, :]
        X = np.concatenate([np.expand_dims(x, 0) for x in [X[:200], X[200:]]], 0)

        clinical = self.clinical[idx, :]

        label = self.labels[idx].astype(np.int64)
        return 0, (X, clinical), (label, )


