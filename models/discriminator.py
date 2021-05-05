# =============================================================================
# ================================== Import ===================================
# =============================================================================
import torch
import torch.nn as nn


# =============================================================================
# ================ Discriminator for 1 channel 28 x 28 images =================
# =============================================================================
class D_InfoGan_1C28(nn.Module):
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
        super(D_InfoGan_1C28, self).__init__()

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


class D_InfoGan_CGAN_1C28(D_InfoGan_1C28):
    """A CGAN compatible discriminator. Concat condition information before the
    first layer.

    Parameters
    ----------
    img_channels : int
        Channels of the input images.
    output_dim : int
        Dimension of the output.
    c_dim : int
        Condition dimension.

    """
    def __init__(self, img_channels, output_dim, c_dim):
        super(D_InfoGan_CGAN_1C28, self).__init__(img_channels, output_dim)

        # conv1 (batch_size, img_channels + c_dim -> 64, 28 -> 14, 28 -> 14)
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=img_channels + c_dim, out_channels=64,
                          kernel_size=[4, 4], stride=2, padding=1),
                nn.LeakyReLU(0.2))

    def forward(self, x, c):
        """Forward propagation.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            Images, a tensor of shape (batch_size, 1, 28, 28).
        c : :class:`torch.Tensor`
            Condition input, a tensor of shape (batch_size, 1, 28, 28).

        Returns
        -------
        x_ : :class:`torch.Tensor`
            Real/fake tensor of shape (batch_size, 1).

        """

        # (batch_size, in_channels + c_dim, h, w)
        x_ = torch.cat([x, c], 1)
        x_ = super().forward(x_)
        return x_


class D_InfoGan_3C32(nn.Module):
    """Discriminator used in InfoGan paper for SVHN dataset.

    Parameters
    ----------
    img_channels : int
        Channels of the input images.
    output_dim : int
        Dimension of the output.

    """
    def __init__(self, img_channels, output_dim):
        super(D_InfoGan_3C32, self).__init__()

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


class D_InfoGan_CGAN_3C32(D_InfoGan_3C32):
    """A CGAN compatible discriminator.

    Parameters
    ----------
    img_channels : int
        Channels of the input images.
    output_dim : int
        Dimension of the output.
    c_dim : int
        Condition dimension.

    """
    def __init__(self, img_channels, output_dim, c_dim):
        super(D_InfoGan_CGAN_3C32, self).__init__(img_channels, output_dim)

        # conv1 (batch_size, img_channels + c_dim -> 64, 28 -> 14, 28 -> 14)
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=img_channels + c_dim, out_channels=64,
                          kernel_size=[4, 4], stride=2, padding=1),
                nn.LeakyReLU(0.2))
