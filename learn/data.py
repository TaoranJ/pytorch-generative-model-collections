# =============================================================================
# ================================== Import ===================================
# =============================================================================
import os
import torch
import torchvision as thv
from torchvision import transforms
import torchvision.datasets as dset
from torch.utils.data import DataLoader


# =============================================================================
# ================================= Dataset ===================================
# =============================================================================
def dataloader(args):
    """Return the dataloader for selected dataset.
    Now have:
    - MNIST
    - FashionMNIST
    - CIFAR10
    - CIFAR100
    - SVHN
    - CelebA (https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZ
      zg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ)
    - STL10
    - LSUN
    - Fake data

    Parameters
    ----------
    batch_size : int
        Minibatch size.
    dataset_name : str
        Name of the selected dataset.

    Returns
    -------
    tr_set:
        Dataloader for training set.
    te_set:
        Dataloader for test set.

    """

    # resize images or not
    if args.img_resize:
        transform3c = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.CenterCrop(args.img_size),  # if H != W
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))])
        transform1c = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.CenterCrop(args.img_size),  # if H != W
            transforms.ToTensor(), transforms.Normalize((.5), (.5))])
    else:
        transform3c = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((.5, .5, .5),
                                                              (.5, .5, .5))])
        transform1c = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((.5), (.5))])
    # create dataloaders
    datapath, dataset_name, batch_size = 'data', args.dataset, args.batch_size
    if dataset_name == 'mnist':  # handwritten digits, (1, 28, 28)
        tr_set = thv.datasets.MNIST(datapath, train=True, download=True,
                                    transform=transform1c)
        te_set = thv.datasets.MNIST(datapath, train=False, download=True,
                                    transform=transform1c)
    elif dataset_name == 'fashion-mnist':  # fashion (Zalando), (1, 28, 28)
        tr_set = thv.datasets.FashionMNIST(datapath, train=True, download=True,
                                           transform=transform1c)
        te_set = thv.datasets.FashionMNIST(datapath, train=False,
                                           download=True,
                                           transform=transform1c)
    elif dataset_name == 'cifar10':  # 10-class image recognition, (3, 32 32)
        tr_set = thv.datasets.CIFAR10(datapath, train=True, download=True,
                                      transform=transform3c)
        te_set = thv.datasets.CIFAR10(datapath, train=False, download=True,
                                      transform=transform3c)
    elif dataset_name == 'cifar100':  # 100-class image recognition, (3, 32 32)
        tr_set = thv.datasets.CIFAR100(datapath, train=True, download=True,
                                       transform=transform3c)
        te_set = thv.datasets.CIFAR100(datapath, train=False, download=True,
                                       transform=transform3c)
    elif dataset_name == 'svhn':  # digit recognition, (3, 32, 32)
        tr_set = thv.datasets.SVHN(os.path.join(datapath, 'SVHN'),
                                   split='train', download=True,
                                   transform=transform3c)
        te_set = thv.datasets.SVHN(os.path.join(datapath, 'SVHN'),
                                   split='test', download=True,
                                   transform=transform3c)
    elif dataset_name == 'celeba':  # celebrity face, (3, 218, 178)
        celeba = dset.ImageFolder(root='data/celeba', transform=transform3c)
        tr_len = int(len(celeba) * 0.8)
        te_len = len(celeba) - tr_len
        tr_set, te_set = torch.utils.data.random_split(celeba,
                                                       [tr_len, te_len])
    elif dataset_name == 'stl10':  # 10-class image recognition, (3, 96, 96)
        tr_set = thv.datasets.STL10(datapath, split='train', download=True,
                                    transform=transform3c)
        te_set = thv.datasets.STL10(datapath, split='test', download=True,
                                    transform=transform3c)
    elif dataset_name == 'lsun':
        tr_classes = [c + '_train' for c in args.lsun_classes.split(',')]
        te_classes = [c + '_test' for c in args.lsun_classes.split(',')]
        tr_set = dset.LSUN(root='data/lsun', classes=tr_classes)
        te_set = dset.LSUN(root='data/lsun', classes=te_classes)
    elif dataset_name == 'fake':
        tr_set = dset.FakeData(
                               image_size=(3, args.img_size, args.img_size),
                               transform=transforms.ToTensor())
        te_set = dset.FakeData(size=1024,
                               image_size=(3, args.img_size, args.img_size),
                               transform=transforms.ToTensor())
    tr_set = DataLoader(tr_set, batch_size=batch_size, shuffle=True,
                        drop_last=True)
    te_set = DataLoader(te_set, batch_size=batch_size, shuffle=True,
                        drop_last=True)
    args.img_channels = 1 if dataset_name in ['mnist', 'fashion-mnist'] else 3
    if not args.img_resize:  # use original size
        if dataset_name in ['mnist', 'fashion-mnist']:
            args.img_size = 28
        elif dataset_name in ['cifar10', 'cifar100', 'svhn']:
            args.img_size = 32
        elif dataset_name == 'celeba':
            args.img_size = [218, 178]
        elif dataset_name == 'stl10':
            args.img_size = 96
    return tr_set, te_set
