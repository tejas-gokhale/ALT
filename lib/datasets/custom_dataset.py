import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class cat_dataloaders():
    """
    Class to concatenate multiple dataloaders.
    adapted from https://discuss.pytorch.org/t/train-simultaneously-on-two-datasets/649/35
    """
    def __init__(self, dataloaders, proportions=[1,1]):
        self.dataloaders = dataloaders
        self.proportions = proportions        

    def __iter__(self):
        self.loader_iter = []
        for data_loader in self.dataloaders:
            self.loader_iter.append(iter(data_loader))
        return self

    def __getitem__(self):
        ## this is more generic -- for n dataloaders
        out = []
        for data_iter in self.loader_iter:
            out.append(next(data_iter))
        ## we only care about 2 datasets
        img1, label1 = next(self.loader_iter[0])
        img2, label2 = next(self.loader_iter[1])

        img_both = torch.cat([img1, img2], dim=0)
        label_both = torch.cat([label1, label2], dim=0)

        return img_both, label_both

    def __len__(self):
        L = 0
        for i, dd in enumerate(self.dataloaders):
            L += int(self.proportions[i] * len(dd))   ### AGAT ORIG
            # L += len(dd)
        return L


class AugData(torch.utils.data.Dataset):
    def __init__(self, x_aug, y_aug, transform=None):   
        self.x_aug = x_aug
        self.y_aug = y_aug
        self.transform = transform
        print(
            "x_aug.shape:{} Y_aug.shape:{}".format(
                x_aug.shape, y_aug.shape)
            )
    def __getitem__(self, index):
        img_aug = self.x_aug[index]
        label_aug = self.y_aug[index]

        if self.transform is not None:
            img_aug = self.transform(img_aug)

        return img_aug, label_aug
    def __len__(self):
        return len(self.x_aug)


class PairData(torch.utils.data.Dataset):
    def __init__(self, x_orig, x_aug, y, transform=None):   
        self.x_orig = x_orig
        self.x_aug = x_aug
        self.y = y
        self.transform = transform

        print(
            "x_orig.shape:{} x_aug.shape:{} y.shape: {}".format(
                x_orig.shape, x_aug.shape, y.shape)
            )

    def __getitem__(self, index):
        img_orig = self.x_orig[index]
        img_aug = self.x_aug[index]
        label = self.y[index]

        if self.transform is not None:
            img_orig = self.transform(img_orig)
            img_aug = self.transform(img_aug)
        return img_orig, img_aug, label

    def __len__(self):
        return len(self.x_aug)


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
