# =============================================================================
# ================================== Import ===================================
# =============================================================================
import torch
import torch.nn as nn


# =============================================================================
# ============================== 28 x 28 Images ===============================
# =============================================================================
class G_InfoGan_28(nn.Module):
    """Generator used in InfoGAN paper for MNIST dataset. Can be also applied
    to other 28 * 28 image dataset.

    Parameters
    ----------
    latent_dim : int
        Size of the latent vector.
    img_channels : int
        Number of channels of the images.

    """
    def __init__(self, latent_dim, img_channels):
        super(G_InfoGan_28, self).__init__()

        self.fc = nn.Sequential(
                # (batch, latent_dim -> 1024)
                nn.Linear(latent_dim, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                # (batch, 1024 -> 7 * 7 * 128)
                nn.Linear(1024, 7 * 7 * 128),
                nn.BatchNorm1d(7 * 7 * 128),
                nn.ReLU())
        self.dconv = nn.Sequential(
                # (batch, 128 -> 64, 7 -> 14, 7 -> 14)
                nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                   kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                # (batch, 128 -> img_channels, 14 -> 28, 14 -> 28)
                nn.ConvTranspose2d(in_channels=64, out_channels=img_channels,
                                   kernel_size=[4, 4], stride=2, padding=1),
                nn.Tanh())  # Tanh [-1, 1] range of transformed images

    def forward(self, latent_vector):
        """Forward propagation.

        Parameters
        ----------
        latent_vector : :class:`torch.Tensor`
            Sampled noise, a tensor of shape (batch, latent_dim).

        Returns
        -------
        imgs : :class:`torch.Tensor`
            Generated figures, a tensor of shape
            (batch, img_channels, 28, 28).

        """

        feature_map = self.fc(latent_vector)
        feature_map = feature_map.view(feature_map.size(0), 128, 7, 7)
        imgs = self.dconv(feature_map)
        return imgs


class G_InfoGan_CGAN_28(G_InfoGan_28):
    """A CGAN compatible generator.

    Only need to add condition tensor to the first layer of the G_InfoGAN_1C28.

    Parameters
    ----------
    latent_dim : int
        Size of the latent vector.
    img_channels : int
        # of in_channels of the images.
    c_dim : int
        Size of condition vector.

    """
    def __init__(self, latent_dim, img_channels, c_dim):
        super(G_InfoGan_CGAN_28, self).__init__(latent_dim + c_dim,
                                                img_channels)

    def forward(self, latent_vector, condition_vector):
        """Forward propagation.

        Parameters
        ----------
        latent_vector : :class:`torch.Tensor`
            Sampled noise, a tensor of shape (batch, latent_dim).
        condition_vector : :class:`torch.Tensor`
            Sampled noise, a tensor of shape (batch, c_dim).

        Returns
        -------
        imgs : :class:`torch.Tensor`
            Generated figures, a tensor of shape (batch, img_channels, 28, 28).

        """

        # (batch, latent_dim + c_dim)
        imgs = super().forward(torch.cat([latent_vector, condition_vector], 1))
        return imgs


# =============================================================================
# ============================== 32 x 32 Images ===============================
# =============================================================================
class G_InfoGan_32(nn.Module):
    """Generator used in InfoGAN paper for SVHN dataset. Can be also applied to
    other 32 x 32 images.

    Parameters
    ----------
    latent_dim : int
        Dimension for the noise.
    img_channels : int
        Number of in_channels of the images.

    """
    def __init__(self, latent_dim, img_channels):
        super(G_InfoGan_32, self).__init__()

        self.fc = nn.Sequential(
                # (batch, latent_dim -> 2 * 2 * 448)
                nn.Linear(latent_dim, 2 * 2 * 448),
                nn.BatchNorm1d(2 * 2 * 448),
                nn.ReLU())
        self.dconv = nn.Sequential(
                # (batch, 448 -> 256, 2 -> 4, 2 -> 4)
                nn.ConvTranspose2d(in_channels=448, out_channels=256,
                                   kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256), nn.ReLU(),
                # (batch, 256 -> 128, 4 -> 8, 4 -> 8)
                nn.ConvTranspose2d(in_channels=256, out_channels=128,
                                   kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                # (batch, 128 -> 64, 8 -> 16, 8 -> 16)
                nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                   kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                # (batch, 64 -> 3, 8 -> 32, 8 -> 32)
                nn.ConvTranspose2d(in_channels=64, out_channels=img_channels,
                                   kernel_size=4, stride=2, padding=1),
                nn.Tanh())

    def forward(self, latent_vector):
        """Forward propagation.

        Parameters
        ----------
        latent_vector : :class:`torch.Tensor`
            Sampled noise, a tensor of shape (batch, latent_dim).

        Returns
        -------
        imgs : :class:`torch.Tensor`
            Generated images, a tensor of shape (batch, img_channels, 32, 32).

        """

        feature_map = self.fc(latent_vector)
        feature_map = feature_map.view(feature_map.size(0), 448, 2, 2)
        imgs = self.dconv(feature_map)
        return imgs


class G_InfoGan_CGAN_32(G_InfoGan_32):
    """A CGAN compatible generator.

    Only need to add condition tensor to the first layer of the G_InfoGAN_3C32.

    Parameters
    ----------
    latent_dim : int
        Size of the latent vector.
    img_channels : int
        Number of in_channels of the images.
    c_dim : int
        Condition dimension.

    """
    def __init__(self, latent_dim, img_channels, c_dim):
        super(G_InfoGan_CGAN_32, self).__init__(latent_dim + c_dim,
                                                img_channels)

    def forward(self, latent_vector, condition_vector):
        """Forward propagation.

        Parameters
        ----------
        latent_vector : :class:`torch.Tensor`
            Sampled noise, a tensor of shape (batch, latent_dim).
        condition_vector : :class:`torch.Tensor`
            Sampled noise, a tensor of shape (batch, c_dim).

        Returns
        -------
        imgs : :class:`torch.Tensor`
            Generated figures, a tensor of shape (batch, img_channels, 28, 28).

        """

        # (batch, latent_dim + c_dim)
        imgs = super().forward(torch.cat([latent_vector, condition_vector], 1))
        return imgs


# =============================================================================
# ============================== 64 x 64 Images ===============================
# =============================================================================
class G_DCGAN_64(nn.Module):
    """Generator for the DCGAN. This generator is hardcoded to generate 64 * 64
    images. It could be easily expanded to 2^x * 2^x images by adding or
    removing deconvolutional layers with increasing/decreasing
    feature_map_dim * x.

    Parameters
    ----------
    latent_dim : int
        Size of latent vector.
    feature_map_dim: int
        Size of feature map in generator.
    img_channels : int
        In_channels of images.

    """
    def __init__(self, latent_dim, feature_map_dim, img_channels):
        super(G_DCGAN_64, self).__init__()
        self.generator = nn.Sequential(
                # (batch, latent_dim -> feature_map_dim * 8, 1 -> 4, 1 -> 4)
                nn.ConvTranspose2d(in_channels=latent_dim,
                                   out_channels=feature_map_dim * 8,
                                   kernel_size=4, stride=1, padding=0,
                                   bias=False),
                nn.BatchNorm2d(feature_map_dim * 8), nn.ReLU(True),
                # (batch, feature_map_dim * 8 -> * 4, 4 -> 8, 4 -> 8)
                nn.ConvTranspose2d(in_channels=feature_map_dim * 8,
                                   out_channels=feature_map_dim * 4,
                                   kernel_size=4, stride=2, padding=1,
                                   bias=False),
                nn.BatchNorm2d(feature_map_dim * 4),
                nn.ReLU(True),
                # (batch, feature_map_dim * 4 -> * 2, 8 -> 16, 8 -> 16)
                nn.ConvTranspose2d(in_channels=feature_map_dim * 4,
                                   out_channels=feature_map_dim * 2,
                                   kernel_size=4, stride=2, padding=1,
                                   bias=False),
                nn.BatchNorm2d(feature_map_dim * 2),
                nn.ReLU(True),
                # (batch, feature_map_dim * 2 -> * 1, 16 -> 32, 16 -> 32)
                nn.ConvTranspose2d(in_channels=feature_map_dim * 2,
                                   out_channels=feature_map_dim,
                                   kernel_size=4, stride=2, padding=1,
                                   bias=False),
                nn.BatchNorm2d(feature_map_dim),
                nn.ReLU(True),
                # (batch, feature_map_dim -> img_channels, 32 -> 64, 32 -> 64)
                nn.ConvTranspose2d(in_channels=feature_map_dim,
                                   out_channels=img_channels,
                                   kernel_size=4, stride=2, padding=1,
                                   bias=False),
                nn.Tanh()  # [-1, 1]
        )

    def forward(self, latent_vector):
        """Forward propagation of the generator.

        Parameters
        ----------
        latent_vector : :class:`torch.Tensor`
            Sampled noise, a tensor of shape (batch, latent_dim).

        Returns
        -------
        imgs : :class:`torch.Tensor`
            Generated images, a tensor of shape (batch, channels, H, W).

        """

        if len(latent_vector.size()) == 2:  # vector_dim * 1 * 1 images
            latent_vector = latent_vector.unsqueeze(-1).unsqueeze(-1)
        imgs = self.generator(latent_vector)
        return imgs
