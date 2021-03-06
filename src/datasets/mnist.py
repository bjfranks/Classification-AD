from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import MNIST
from base.torchvision_dataset import TorchvisionDataset

import numpy as np
import torch
import torchvision.transforms as transforms


class MNIST_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class: int = 0):
        super().__init__(root)

        self.image_size = (1, 28, 28)

        # Define normal and outlier classes
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)
        self.outlier_classes = tuple(self.outlier_classes)

        # MNIST preprocessing: feature scaling to [0, 1]
        transform = transforms.ToTensor()
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        # Get train set
        train_set = MyMNIST(root=self.root, train=True, transform=transform, target_transform=target_transform,
                            download=True)

        # Subset train_set to normal_classes
        idx = np.argwhere(np.isin(train_set.targets.cpu().data.numpy(), self.normal_classes))
        idx = idx.flatten().tolist()
        train_set.semi_targets[idx] = torch.zeros(len(idx)).long()
        self.train_set = Subset(train_set, idx)

        # Get test set
        self.test_set = MyMNIST(root=self.root, train=False, transform=transform, target_transform=target_transform,
                                download=True)


class MyMNIST(MNIST):
    """
    Torchvision MNIST class with additional targets for the outlier exposure setting and patch of __getitem__ method
    to also return the outlier exposure target as well as the index of a data sample.
    """

    def __init__(self, *args, **kwargs):
        super(MyMNIST, self).__init__(*args, **kwargs)

        self.semi_targets = torch.zeros_like(self.targets)

    def __getitem__(self, index):
        """Override the original method of the MNIST class.
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, semi_target, index)
        """
        img, target, semi_target = self.data[index], int(self.targets[index]), int(self.semi_targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, semi_target, index
