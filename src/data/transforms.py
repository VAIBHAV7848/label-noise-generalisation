"""Data transforms and augmentations for vision datasets."""

import torchvision.transforms as transforms


def get_cifar_transforms(train: bool = True):
    """Return standard CIFAR-10 / CIFAR-100 transforms."""
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def get_mnist_transforms(train: bool = True):
    """Return standard MNIST transforms."""
    mean = (0.1307,)
    std = (0.3081,)

    if train:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
