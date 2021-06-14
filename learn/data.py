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
def dataloader(batch_size, dataset_name, args):
    """Return the dataloader for selected datset.

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
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))])
        transform1c = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(), transforms.Normalize((.5), (.5))])
    else:
        transform3c = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((.5, .5, .5),
                                                              (.5, .5, .5))])
        transform1c = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((.5), (.5))])
    # create dataloaders
    datapath = 'data'
    if dataset_name == 'mnist':  # handwritten digits, 1 channel
        tr_set = DataLoader(
                thv.datasets.MNIST(datapath, train=True, download=True,
                                   transform=transform1c),
                batch_size=batch_size, shuffle=True, drop_last=True)
        te_set = DataLoader(
                thv.datasets.MNIST(datapath, train=False, download=True,
                                   transform=transform1c),
                batch_size=batch_size, shuffle=True, drop_last=True)
    elif dataset_name == 'fashion-mnist':  # fashion (Zalando), 1 channel
        tr_set = DataLoader(
                thv.datasets.FashionMNIST(datapath, train=True, download=True,
                                          transform=transform1c),
                batch_size=batch_size, shuffle=True, drop_last=True)
        te_set = DataLoader(
                thv.datasets.FashionMNIST(datapath, train=False, download=True,
                                          transform=transform1c),
                batch_size=batch_size, shuffle=True, drop_last=True)
    elif dataset_name == 'cifar10':  # 10-class image recognition, 3 channels
        tr_set = DataLoader(
                thv.datasets.CIFAR10(datapath, train=True, download=True,
                                     transform=transform3c),
                batch_size=batch_size, shuffle=True, drop_last=True)
        te_set = DataLoader(
                thv.datasets.CIFAR10(datapath, train=False, download=True,
                                     transform=transform3c),
                batch_size=batch_size, shuffle=True, drop_last=True)
    elif dataset_name == 'svhn':  # object (numbers) recognition, 3 channels
        tr_set = DataLoader(
                thv.datasets.SVHN(os.path.join(datapath, 'SVHN'),
                                  split='train', download=True,
                                  transform=transform3c),
                batch_size=batch_size, shuffle=True, drop_last=True)
        te_set = DataLoader(
                thv.datasets.SVHN(os.path.join(datapath, 'SVHN'), split='test',
                                  download=True, transform=transform3c),
                batch_size=batch_size, shuffle=True, drop_last=True)
    elif dataset_name == 'celeba':  # celebrity face, 3 channel
        # Download manually from here:
        # https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?
        # resourcekey=0-rJlzl934LzC-Xp28GeIBzQ
        celeba = dset.ImageFolder(root='data/celeba',
                                  transform=transform3c)
        tr_len = int(len(celeba) * 0.8)
        te_len = len(celeba) - tr_len
        tr_set, te_set = torch.utils.data.random_split(celeba,
                                                       [tr_len, te_len])
        tr_set = DataLoader(tr_set, batch_size=batch_size, shuffle=True,
                            drop_last=True)
        te_set = DataLoader(te_set, batch_size=batch_size, shuffle=True,
                            drop_last=True)
    elif dataset_name == 'stl10':  # 10-class image recognition, 3 channels
        tr_set = DataLoader(
            thv.datasets.STL10(datapath, split='train', download=True,
                               transform=transform3c),
            batch_size=batch_size, shuffle=True, drop_last=True)
        te_set = DataLoader(
            thv.datasets.STL10(datapath, split='test', download=True,
                               transform=transform3c),
            batch_size=batch_size, shuffle=True, drop_last=True)
    if dataset_name in ['mnist', 'fashion-mnist']:
        args.in_channels = 1
    else:
        args.in_channels = 3
    return tr_set, te_set
