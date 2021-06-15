# =============================================================================
# ================================== Import ===================================
# =============================================================================
import torch
import torch.nn as nn


# =============================================================================
# ================== Generator for 1 channel 28 x 28 images ===================
# =============================================================================
class G_InfoGan_1C28(nn.Module):
    """Generator used in InfoGAN paper for MNIST dataset. Can be also applied
    to Fashion-MNIST dataset.

    Parameters
    ----------
    z_dim : int
        Dimension for the noise.
    img_channels : int
        # of channels of the images.

    """
    def __init__(self, z_dim, img_channels):
        super(G_InfoGan_1C28, self).__init__()

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
                                   kernel_size=[4, 4], stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU())
        # un-conv 2: (batch_size, 128 -> 1, 14 -> 28, 14 -> 28)
        self.dconv2 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=64, out_channels=img_channels,
                                   kernel_size=[4, 4], stride=2, padding=1),
                nn.Tanh())  # Tanh [-1, 1] range of transformed images

    def forward(self, x):
        """Forward propagation.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            Sampled noise, a tensor of shape (batch_size, z_dim).

        Returns
        -------
        x : :class:`torch.Tensor`
            Generated figures, a tensor of shape (batch_size, 1, 28, 28).

        """

        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(x.size(0), 128, 7, 7)
        x = self.dconv1(x)
        x = self.dconv2(x)
        return x


class G_InfoGan_CGAN_1C28(G_InfoGan_1C28):
    """A CGAN compatible generator.

    Only need to add condition tensor to the first layer of the G_InfoGAN_1C28.

    Parameters
    ----------
    z_dim : int
        Dimension for the noise.
    img_channels : int
        # of in_channels of the images.
    c_dim : int
        Condition dimension.

    """
    def __init__(self, z_dim, img_channels, c_dim):
        super(G_InfoGan_CGAN_1C28, self).__init__(z_dim, img_channels)

        # fc 1: (batch_size, z_dim + c_dim -> 1024)
        self.fc1 = nn.Sequential(
                nn.Linear(z_dim + c_dim, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU())

    def forward(self, x, c):
        """Forward propagation.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            Sampled noise, a tensor of shape (batch_size, z_dim).

        Returns
        -------
        x_ : :class:`torch.Tensor`
            Generated figures, a tensor of shape (batch_size, 1, 28, 28).

        """

        # (batch_size, z_dim + c_dim)
        x_ = torch.cat([x, c], 1)
        x_ = super().forward(x_)
        return x_


# =============================================================================
# ================= Generator for 3 channels 32 x 32 images ===================
# =============================================================================
class G_InfoGan_3C32(nn.Module):
    """Generator used in InfoGAN paper for SVHN dataset.

    Parameters
    ----------
    z_dim : int
        Dimension for the noise.
    img_channels : int
        # of in_channels of the images.

    """
    def __init__(self, z_dim, img_channels):
        super(G_InfoGan_3C32, self).__init__()

        # fc 1: (batch_size, z_dim -> 2 * 2 * 448)
        self.fc1 = nn.Sequential(
                nn.Linear(z_dim, 2 * 2 * 448),
                nn.BatchNorm1d(2 * 2 * 448),
                nn.ReLU())
        # un-conv 1: (batch_size, 448 -> 256, 2 -> 4, 2 -> 4)
        self.dconv1 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=448, out_channels=256,
                                   kernel_size=[4, 4], stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU())
        # un-conv 2: (batch_size, 256 -> 128, 4 -> 8, 4 -> 8)
        self.dconv2 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=256, out_channels=128,
                                   kernel_size=[4, 4], stride=2, padding=1),
                nn.ReLU())
        # un-conv 3: (batch_size, 128 -> 64, 8 -> 16, 8 -> 16)
        self.dconv3 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                   kernel_size=[4, 4], stride=2, padding=1),
                nn.ReLU())
        # un-conv 4: (batch_size, 64 -> 3, 8 -> 32, 8 -> 32)
        self.dconv4 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=64, out_channels=3,
                                   kernel_size=[4, 4], stride=2, padding=1),
                nn.Tanh())

    def forward(self, x):
        """Forward propagation.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            Sampled noise, a tensor of shape (batch_size, z_dim).

        Returns
        -------
        x : :class:`torch.Tensor`
            Generated images, a tensor of shape (batch_size, 3, 32, 32).

        """

        x = self.fc1(x)
        x = x.view(x.size(0), 448, 2, 2)
        x = self.dconv1(x)
        x = self.dconv2(x)
        x = self.dconv3(x)
        x = self.dconv4(x)
        return x


class G_InfoGan_CGAN_3C32(G_InfoGan_3C32):
    """A CGAN compatible generator.

    Only need to add condition tensor to the first layer of the G_InfoGAN_3C32.

    Parameters
    ----------
    z_dim : int
        Dimension for the noise.
    img_channels : int
        # of in_channels of the images.
    c_dim : int
        Condition dimension.

    """
    def __init__(self, z_dim, img_channels, c_dim):
        super(G_InfoGan_CGAN_3C32, self).__init__(z_dim, img_channels)

        # fc 1: (batch_size, z_dim + condition_dim -> 2 * 2 * 448)
        self.fc1 = nn.Sequential(
                nn.Linear(z_dim + c_dim, 2 * 2 * 448),
                nn.BatchNorm1d(2 * 2 * 448),
                nn.ReLU())

    def forward(self, x, c):
        """Forward propagation.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            Sampled noise, a tensor of shape (batch_size, z_dim).

        Returns
        -------
        x_ : :class:`torch.Tensor`
            Generated images, a tensor of shape (batch_size, 3, 32, 32).

        """

        # (batch_size, z_dim + c_dim)
        x_ = torch.cat([x, c], 1)
        x_ = super().forward(x_)
        return x_


# =============================================================================
# ================= Generator for 3 channels 64 x 64 images ===================
# =============================================================================
class G_DCGAN_3C64(nn.Module):
    """Generator for the DCGAN. This generator is hardcoded to generate 64 * 64
    images. It could be easily expanded to 2^x * 2^x images by adding or
    removing deconvolutional layers with increasing/decreasing
    feature_map_dim * x.

    Parameters
    ----------
    latent_dim : int
        Length of latent vector.
    feature_map_dim: int
        Size of feature map in generator.
    img_channels : int
        In_channels of images.
    ngpu: int
        Number of GPUs to use.

    """

    def __init__(self, latent_dim, feature_map_dim, img_channels, ngpu):
        super(G_DCGAN_3C64, self).__init__()
        self.ngpu = ngpu
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
        if latent_vector.is_cuda and self.ngpu > 1:
            imgs = nn.parallel.data_parallel(self.generator, latent_vector,
                                             range(self.ngpu))
        else:
            imgs = self.generator(latent_vector)
        return imgs
