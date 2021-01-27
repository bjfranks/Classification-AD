from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import CIFAR100
from base.torchvision_dataset import TorchvisionDataset

import numpy as np
import torch
import torchvision.transforms as transforms
import random


class CIFAR100_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class: int = 0, data_augmentation: bool = False, normalize: bool = False,
                 size: int = 50000, outlier_exposure: bool = False, oe_n_classes: int = 100, seed: int = 0):
        super().__init__(root)

        self.image_size = (3, 32, 32)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.shuffle = True
        random.seed(seed)  # set seed

        if outlier_exposure:
            self.normal_classes = None
            self.outlier_classes = list(range(0, 100))
            self.known_outlier_classes = tuple(random.sample(self.outlier_classes, oe_n_classes))
            size = min(size, oe_n_classes*500)
            self.size = size
        else:
            # Define normal and outlier classes
            self.normal_classes = tuple([normal_class])
            self.outlier_classes = list(range(0, 100))
            self.outlier_classes.remove(normal_class)
            self.outlier_classes = tuple(self.outlier_classes)

        # CIFAR-100 preprocessing: feature scaling to [0, 1], data normalization, and data augmentation
        train_transform = []
        test_transform = []
        if data_augmentation:
            # only augment training data
            train_transform += [transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomCrop(32, padding=4)]
        train_transform += [transforms.ToTensor()]
        test_transform += [transforms.ToTensor()]
        if data_augmentation:
            train_transform += [transforms.Lambda(lambda x: x + 0.001 * torch.randn_like(x))]
        if normalize:
            train_transform += [transforms.Normalize((0.491373, 0.482353, 0.446667), (0.247059, 0.243529, 0.261569))]
            test_transform += [transforms.Normalize((0.491373, 0.482353, 0.446667), (0.247059, 0.243529, 0.261569))]
        train_transform = transforms.Compose(train_transform)
        test_transform = transforms.Compose(test_transform)

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        # Get train set
        train_set = MyCIFAR100(root=self.root, train=True, transform=train_transform, target_transform=target_transform,
                               download=True)

        if outlier_exposure:
            idx = np.argwhere(np.isin(np.array(train_set.targets), self.known_outlier_classes))

            # Select size many elements for training
            if self.size < len(idx):
                random.seed(seed)
                idx = idx[random.sample(range(len(idx)), self.size)]

            idx = idx.flatten().tolist()
            train_set.semi_targets[idx] = -1 * torch.ones(len(idx)).long()  # set outlier exposure labels

            # Subset train_set to selected classes
            self.train_set = Subset(train_set, idx)
            self.train_set.shuffle_idxs = False
            self.test_set = None
        else:
            # Subset train_set to normal_classes
            idx = np.argwhere(np.isin(np.array(train_set.targets), self.normal_classes))
            idx = idx.flatten().tolist()
            train_set.semi_targets[idx] = torch.zeros(len(idx)).long()
            self.train_set = Subset(train_set, idx)

            # Get test set
            self.test_set = MyCIFAR100(root=self.root, train=False, transform=test_transform,
                                       target_transform=target_transform, download=True)


class MyCIFAR100(CIFAR100):
    """
    Torchvision CIFAR100 class with additional targets for the outlier exposure setting and patch of __getitem__ method
    to also return the outlier exposure target as well as the index of a data sample.
    """

    def __init__(self, *args, **kwargs):
        super(MyCIFAR100, self).__init__(*args, **kwargs)

        self.semi_targets = torch.zeros(len(self.targets), dtype=torch.int64)
        self.shuffle_idxs = False

    def __getitem__(self, index):
        """Override the original method of the CIFAR100 class.
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, semi_target, index)
        """
        img, target, semi_target = self.data[index], self.targets[index], int(self.semi_targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, semi_target, index
