# =============================================================================
# ================================== Import ===================================
# =============================================================================
import torch.nn as nn


class InfoGanMnistG(nn.Module):
    """Generator used in InfoGAN paper for MNIST dataset. Can be also applied
    to Fashion-MNIST dataset.

    Parameters
    ----------
    z_dim : int
        Dimension for the noise.
    img_channels : int
        # of in_channels of the images.

    """
    def __init__(self, z_dim, img_channels):
        super(InfoGanMnistG, self).__init__()

        # fc 1: (batch_size, z_dim -> 1024)
        self.fc1 = nn.Sequential(
                nn.Linear(z_dim, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU())
        # fc 2: (batch_size, 1024 -> 7 * 7 * 128)
        self.fc2 = nn.Sequential(
                nn.Linear(1024, 7 * 7 * 128),
                nn.BatchNorm1d(7 * 7 * 128),
                nn.ReLU())
        # un-conv 1: (batch_size, 128 -> 64, 7 -> 14, 7 -> 14)
        self.dconv1 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                   kernel_size=[4, 4], stride=2,
                                   padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU())
        # un-conv 2: (batch_size, 128 -> 1, 14 -> 28, 14 -> 28)
        self.dconv2 = nn.ConvTranspose2d(in_channels=64,
                                         out_channels=img_channels,
                                         kernel_size=[4, 4], stride=2,
                                         padding=1)

    def forward(self, x):
        """Forward propagation.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            Sampled noise, a tensor of shape (batch_size, z_dim).

        """

        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(x.size(0), 128, 7, 7)
        x = self.dconv1(x)
        x = self.dconv2(x)
        return x
