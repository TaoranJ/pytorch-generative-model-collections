# =============================================================================
# ================================== Import ===================================
# =============================================================================
import torch
import torch.nn as nn


# =============================================================================
# ============================== 28 x 28 Images ===============================
# =============================================================================
class D_InfoGan_28(nn.Module):
    """Discriminator used in InfoGan paper for MNIST dataset. Can be also
    applied to other 28 * 28 image dataset.

    Parameters
    ----------
    img_channels : int
        Channels of the input images.

    """
    def __init__(self, img_channels):
        super(D_InfoGan_28, self).__init__()

        self.conv = nn.Sequential(
                # (batch, img_channels -> 64, 28 -> 14, 28 -> 14)
                nn.Conv2d(in_channels=img_channels, out_channels=64,
                          kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                # (batch, 64 -> 128, 14 -> 7, 14 -> 7)
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4,
                          stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2))
        self.fc = nn.Sequential(
                # (batch, 128 * 7 * 7 -> 1024)
                nn.Linear(128 * 7 * 7, 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(0.2),
                # (batch, 1024 -> 1)
                nn.Linear(1024, 1),
                nn.Sigmoid())

    def forward(self, imgs):
        """Forward propagation.

        Parameters
        ----------
        imgs : :class:`torch.Tensor`
            Images, a tensor of shape (batch, img_channels, 28, 28).

        Returns
        -------
        pred : :class:`torch.Tensor`
            Real/fake tensor of shape (batch_size, 1).

        """

        assert imgs.size(2) == 28, 'This discriminator is hardcoded for '
        '28 * 28 imgs.'

        feature_map = self.conv(imgs)
        pred = self.fc(torch.flatten(feature_map, start_dim=1))
        return pred


class D_InfoGan_CGAN_28(D_InfoGan_28):
    """A CGAN compatible discriminator. Concat condition information before the
    first layer.

    Parameters
    ----------
    img_channels : int
        Channels of the input images.
    c_dim : int
        Size of condition vector.

    """
    def __init__(self, img_channels, c_dim):
        super(D_InfoGan_CGAN_28, self).__init__(img_channels + c_dim)

    def forward(self, imgs, condition_vector):
        """Forward propagation.

        Parameters
        ----------
        imgs : :class:`torch.Tensor`
            Images, a tensor of shape (batch, img_channels, 28, 28).
        condition_vector : :class:`torch.Tensor`
            Condition input, a tensor of shape (batch, c_dim, 28, 28).

        Returns
        -------
        pred : :class:`torch.Tensor`
            Real/fake tensor of shape (batch, 1).

        """

        # (batch, img_channels + c_dim, h, w)
        pred = super().forward(torch.cat([imgs, condition_vector], 1))
        return pred


# =============================================================================
# ============================== 32 x 32 Images ===============================
# =============================================================================
class D_InfoGan_32(nn.Module):
    """Discriminator used in InfoGan paper for SVHN dataset. Can be also
    applied to other 32 x 32 images.

    Parameters
    ----------
    img_channels : int
        Channels of the input images.

    """
    def __init__(self, img_channels):
        super(D_InfoGan_32, self).__init__()

        self.conv = nn.Sequential(
                # (batch, 3 -> 64, 32 -> 16, 32 -> 16)
                nn.Conv2d(in_channels=img_channels, out_channels=64,
                          kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                # (batch, 64 -> 128, 16 -> 8, 16 -> 8)
                nn.Conv2d(in_channels=64, out_channels=128,
                          kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                # (batch, 128 -> 256, 8 -> 4, 8 -> 4)
                nn.Conv2d(in_channels=128, out_channels=256,
                          kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2))
        self.fc = nn.Sequential(
                # (batch, 256 * 4 * 4 -> 128)
                nn.Linear(256 * 4 * 4, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.2),
                # (batch_size, 128 -> 1)
                nn.Linear(128, 1),
                nn.Sigmoid())

    def forward(self, imgs):
        """Forward propagation.

        Parameters
        ----------
        imgs : :class:`torch.Tensor`
            Images, a tensor of shape (batch, img_channels, 32, 32).

        Returns
        -------
        pred : :class:`torch.Tensor`
            Real/fake tensor of shape (batch, 1).

        """

        assert imgs.size(2) == 32, 'This discriminator is hardcoded for '
        '32 * 32 imgs.'

        feature_map = self.conv(imgs)
        pred = self.fc(torch.flatten(feature_map, start_dim=1))
        return pred


class D_InfoGan_CGAN_32(D_InfoGan_32):
    """A CGAN compatible discriminator. Concat condition information before the
    first layer.

    Parameters
    ----------
    img_channels : int
        Channels of the input images.
    c_dim : int
        Condition dimension.

    """
    def __init__(self, img_channels, c_dim):
        super(D_InfoGan_CGAN_32, self).__init__(img_channels + c_dim)

    def forward(self, imgs, condition_vector):
        """Forward propagation.

        Parameters
        ----------
        imgs : :class:`torch.Tensor`
            Images, a tensor of shape (batch, img_channels, 32, 32).
        condition_vector : :class:`torch.Tensor`
            Condition input, a tensor of shape (batch, c_dim, 32, 32).

        Returns
        -------
        pred : :class:`torch.Tensor`
            Real/fake tensor of shape (batch, 1).

        """

        # (batch, img_channels + c_dim, h, w)
        pred = super().forward(torch.cat([imgs, condition_vector], 1))
        return pred


# =============================================================================
# ============================== 64 x 64 Images ===============================
# =============================================================================
class D_DCGAN_64(nn.Module):
    """Discriminator for the DCGAN. This generator is hardcoded to process
    64 * 64 images. It could be easily expanded to 2^x * 2^x images by adding
    or removing deconvolutional layers with increasing/decreasing
    feature_map_dim * x.

    Parameters
    ----------
    img_channels : int
        In_channels of images.
    feature_map_dim: int
        Size of feature map in discriminator.

    """
    def __init__(self, img_channels, feature_map_dim):
        super(D_DCGAN_64, self).__init__()
        self.discriminator = nn.Sequential(
            # (batch, img_channels -> feature_map_dim, 64 -> 32, 64 -> 32)
            nn.Conv2d(in_channels=img_channels,
                      out_channels=feature_map_dim,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (batch, feature_map_dim -> * 2, 32 -> 16, 32 -> 16)
            nn.Conv2d(in_channels=feature_map_dim,
                      out_channels=feature_map_dim * 2,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (batch, feature_map_dim * 2 -> * 4, 16 -> 8, 16 -> 8)
            nn.Conv2d(in_channels=feature_map_dim * 2,
                      out_channels=feature_map_dim * 4,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (batch, feature_map_dim * 4 -> * 8, 8 -> 4, 8 -> 4)
            nn.Conv2d(in_channels=feature_map_dim * 4,
                      out_channels=feature_map_dim * 8,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (batch, feature_map_dim * 8 -> 1, 4 -> 1, 4 -> 1)
            nn.Conv2d(in_channels=feature_map_dim * 8,
                      out_channels=1, kernel_size=4, stride=1, padding=0,
                      bias=False),
            nn.Sigmoid())

    def forward(self, imgs):
        """Forward propagation.

        Parameters
        ----------
        imgs : :class:`torch.Tensor`
            Images, a tensor of shape (batch_size, img_channels, 64, 64).

        Returns
        -------
        pred : :class:`torch.Tensor`
            Real/fake tensor of shape (batch_size, 1).

        """

        assert imgs.size(2) == 64, 'This discriminator is hardcoded for '
        '64 * 64 imgs.'

        pred = self.discriminator(imgs)
        return pred.view(-1, 1)
