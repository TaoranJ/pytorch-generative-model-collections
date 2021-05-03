# =============================================================================
# ================================== Import ===================================
# =============================================================================
import torch
import torch.nn as nn


# =============================================================================
# ============================== Discriminators ===============================
# =============================================================================
class InfoGan1C28D(nn.Module):
    """Discriminator used in InfoGan paper for MNIST dataset. Can be also
    applied to Fashion-MNIST dataset.

    Parameters
    ----------
    img_channels : int
        Channels of the input images.
    output_dim : int
        Dimension of the output.

    """
    def __init__(self, img_channels, output_dim):
        super(InfoGan1C28D, self).__init__()

        # conv1 (batch_size, 1 -> 64, 28 -> 14, 28 -> 14)
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=img_channels, out_channels=64,
                          kernel_size=[4, 4], stride=2, padding=1),
                nn.LeakyReLU(0.2))
        # conv2 (batch_size, 64 -> 128, 14 -> 7, 14 -> 7)
        self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=128,
                          kernel_size=[4, 4], stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2))
        # fc1 (batch_size, 128 * 7 * 7) -> (batch_size, 1024)
        self.fc1 = nn.Sequential(
                nn.Linear(128 * 7 * 7, 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(0.2))
        # fc2 (batch_size, 1024) -> (batch_size, 1)
        self.fc2 = nn.Sequential(
                nn.Linear(1024, output_dim),
                nn.Sigmoid())

    def forward(self, x):
        """Forward propagation.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            Images, a tensor of shape (batch_size, 1, 28, 28).

        Returns
        -------
        x : :class:`torch.Tensor`
            Real/fake tensor of shape (batch_size, 1).

        """

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1(torch.flatten(x, start_dim=1))
        x = self.fc2(x)
        return x


class InfoGan3C32D(nn.Module):
    """Discriminator used in InfoGan paper for SVHN dataset.

    Parameters
    ----------
    img_channels : int
        Channels of the input images.
    output_dim : int
        Dimension of the output.

    """
    def __init__(self, img_channels, output_dim):
        super(InfoGan3C32D, self).__init__()

        # conv1 (batch_size, 3 -> 64, 32 -> 16, 32 -> 16)
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=img_channels, out_channels=64,
                          kernel_size=[4, 4], stride=2, padding=1),
                nn.LeakyReLU(0.2))
        # conv2 (batch_size, 64 -> 128, 16 -> 8, 16 -> 8)
        self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=128,
                          kernel_size=[4, 4], stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2))
        # conv3 (batch_size, 128 -> 256, 8 -> 4, 8 -> 4)
        self.conv3 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256,
                          kernel_size=[4, 4], stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2))
        # fc1 (batch_size, 256 * 4 * 4) -> (batch_size, 128)
        self.fc1 = nn.Sequential(
                nn.Linear(256 * 4 * 4, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.2))
        # fc2 (batch_size, 128) -> (batch_size, 1)
        self.fc2 = nn.Sequential(
                nn.Linear(128, output_dim),
                nn.Sigmoid())

    def forward(self, x):
        """Forward propagation.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            Images, a tensor of shape (batch_size, 3, 32, 32).

        Returns
        -------
        x : :class:`torch.Tensor`
            Real/fake tensor of shape (batch_size, 1).

        """

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc1(torch.flatten(x, start_dim=1))
        x = self.fc2(x)
        return x
