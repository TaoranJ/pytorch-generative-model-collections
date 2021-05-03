# =============================================================================
# ================================== Import ===================================
# =============================================================================
import os
import torchvision as thv
from torchvision import transforms
from torch.utils.data import DataLoader


# =============================================================================
# ================================= Dataset ===================================
# =============================================================================
def dataloader(batch_size, dataset_name, datapath='data'):
    """Return the dataloader for selected datset.

    Parameters
    ----------
    batch_size : int
        Minibatch size.
    dataset_name : str
        Name of the selected dataset.
    datapath : str
        Where to put the dataset.

    Returns
    -------
    tr_set:
        Dataloader for training set.
    te_set:
        Dataloader for test set.

    """

    transform3c = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((.5, .5, .5),
                                                          (.5, .5, .5))])
    transform1c = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((.5), (.5))])
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
    elif dataset_name == 'svhn':  # object (numbers) recognition, 3 channels
        tr_set = DataLoader(
                thv.datasets.SVHN(os.path.join(datapath, 'SVHN'),
                                  split='train', download=True,
                                  transform=transform3c),
                batch_size=batch_size, shuffle=True, drop_last=True)
        te_set = DataLoader(
                thv.datasets.SVHN(datapath, split='test', download=True,
                                  transform=transform3c),
                batch_size=batch_size, shuffle=True, drop_last=True)
    elif dataset_name == 'celeba':  # celebrity face, 3 channel
        tr_set = DataLoader(
                thv.datasets.CelebA(datapath, split='train', download=True,
                                    transform=transform3c),
                batch_size=batch_size, shuffle=True, drop_last=True)
        te_set = DataLoader(
                thv.datasets.CelebA(datapath, split='test', download=True,
                                    transform=transform3c),
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
    elif dataset_name == 'stl10':  # 10-class image recognition, 3 channels
        tr_set = DataLoader(
            thv.datasets.STL10(datapath, split='train', download=True,
                               transform=transform3c),
            batch_size=batch_size, shuffle=True, drop_last=True)
        te_set = DataLoader(
            thv.datasets.STL10(datapath, split='test', download=True,
                               transform=transform3c),
            batch_size=batch_size, shuffle=True, drop_last=True)
    return tr_set, te_set
