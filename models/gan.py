# =============================================================================
# =============================== Description =================================
# =============================================================================

"""Here we used the generator/discriminator structure defined in the InfoGan
paper."""

# =============================================================================
# ================================== Import ===================================
# =============================================================================
import torch
import torch.nn as nn


class Generator(nn.Module):
    """Generator architecture defined in the InfoGAN paper. Architecture may
    vary based on the dataset.

    Parameters
    ----------
    z_dim : int
        Dimension for the noise.
    img_channels : int
        # of in_channels of the images.
    dataset : str
        Dataset name.
    """
    def __init__(self, z_dim, img_channels, dataset='mnist'):
        super(Generator, self).__init__()

        self.dataset = dataset
        if dataset == 'mnist':
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

    def forward_mnist(self, x):
        """Forward propagation of MNIST generator.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            Noise tensor, of shape (batch_size, z_dim).

        """
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(x.size(0), 128, 7, 7)
        x = self.dconv1(x)
        x = self.dconv2(x)
        return x

    def forward(self, x):
        """Forward propagation.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            Sampled noise, a tensor of shape (batch_size, z_dim).
        """
        if self.dataset == 'mnist':
            return self.forward_mnist(x)


class Discriminator(nn.Module):
    """Discriminator architecture defined in the InfoGAN paper. Architecture
    may vary based on the dataset.

    Parameters
    ----------
    img_channels : int
        Channels of the input images.
    output_dim : int
        Dimension of the output.
    dataset : str
        Dataset name.
    """
    def __init__(self, img_channels, output_dim, dataset='mnist'):
        super(Discriminator, self).__init__()

        self.dataset = dataset
        if dataset == 'mnist':
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

    def forward_mnist(self, x):
        """Forward propagation of MNIST discriminator.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            MNIST images, a tensor of shape (batch_size, 1, 28, 28).

        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1(torch.flatten(x, start_dim=1))
        x = self.fc2(x)
        return x

    def forward(self, x):
        """Forward propagation.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            Image tensor.
        """
        if self.dataset == 'mnist':
            return self.forward_mnist(x)
