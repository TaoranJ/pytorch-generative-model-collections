# =============================================================================
# ================================== Import ===================================
# =============================================================================
import torch
import torchvision as thv
from torchvision import transforms


# =============================================================================
# ================================= Dataset ===================================
# =============================================================================
def mnist_dataloader(batch_size, datapath='data'):
    """Return MNIST dataloader for training/validation set.

    Parameters
    ----------
    batch_size : int
        Minibatch size.
    datapath : str
        Where to put the dataset.

    Returns
    -------
    train_set:
        Dataloader for MNIST training set.
    val_set:
        Dataloader for MNIST validation set.
    """

    transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,))])
    train_set = thv.datasets.MNIST(datapath, download=True, train=True,
                                   transform=transform)
    val_set = thv.datasets.MNIST(datapath, download=True, train=False,
                                 transform=transform)
    train_set = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                            shuffle=True, drop_last=True)
    val_set = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                          shuffle=True, drop_last=True)
    return train_set, val_set
