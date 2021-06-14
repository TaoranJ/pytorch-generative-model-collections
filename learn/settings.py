# =============================================================================
# ================================== Import ===================================
# =============================================================================
import os
import torch
import random
import argparse


# =============================================================================
# ============================= Argument Parser ===============================
# =============================================================================
class GeneralArgParser(object):
    """General setting options for the GAN architecture."""
    def __init__(self):
        super(GeneralArgParser, self).__init__()
        self.parser = argparse.ArgumentParser()

    def device_settings(self):
        """Device related settings."""

        self.parser.add_argument('--cuda', type=int, default=0,
                                 help='Which cuda to use.')
        self.parser.add_argument('--cpu', action='store_true',
                                 help='Use cpus.')
        self.parser.add_argument('--ngpu', type=int, default=1,
                                 help='Number of GPUs to use')

    def optim_settings(self):
        """Optimization related settings."""

        self.parser.add_argument('--lr-g', type=float, default=2e-4,
                                 help='Generator learning rate.')
        self.parser.add_argument('--lr-d', type=float, default=2e-4,
                                 help='Discriminator learning rate.')

        self.parser.add_argument('--epochs', type=int, default=25,
                                 help='Number of epochs to train for.')
        self.parser.add_argument('--batch-size', type=int, default=64,
                                 help='Minibatch size')

        self.parser.add_argument('--d-steps', type=int, default=1,
                                 help='Number of steps to train discriminator '
                                      'each iteration.')
        self.parser.add_argument('--g-steps', type=int, default=1,
                                 help='Number of steps to train generator '
                                      'each iteration.')
        # optimization settings - Adam
        self.parser.add_argument('--beta1', type=float, default=0.5,
                                 help='Beta 1 of Adam.')
        self.parser.add_argument('--beta2', type=float, default=0.999,
                                 help='Beta 2 of Adam.')

    def data_settings(self):
        """Dataset related settings."""

        self.parser.add_argument('--dataset', type=str, default='mnist',
                                 choices=['mnist', 'fashion-mnist', 'svhn',
                                          'celeba', 'cifar10', 'stl10'],
                                 help='Dataset to use.')
        self.parser.add_argument('--dataroot', required=False,
                                 help='path to dataset')
        self.parser.add_argument('--workers', type=int, default=2,
                                 help='Number of data loading workers')
        self.parser.add_argument('--classes', default='bedroom',
                                 help='Comma separated list of classes for '
                                      'the LSUN')
        self.parser.add_argument('--random-seed', type=int,
                                 help='Random seed for reproducibility.')
        self.parser.add_argument('--img-size', type=int, default=64,
                                 help='The height / width of the input image.')
        self.parser.add_argument('--img-resize', action='store_true',
                                 help='Resize images to img-size?')

    def model_settings(self):
        """Model related settings."""

        self.parser.add_argument('--z-dim', type=int, default=64,
                                 help='Size of latent vector.')

    def parse_args(self):
        """Parse command line arguments.

        Returns
        -------
        args : :class:`argparse.Namespace`
            Command line arguments.

        """

        self.device_settings()
        self.optim_settings()
        self.data_settings()
        self.model_settings()
        args = self.parser.parse_args()
        # other settings
        args.device = 'cuda:{}'.format(args.cuda) \
            if torch.cuda.is_available() else 'cpu'
        args.model_name = 'GAN'
        args.eval_dir = 'eval_{}'.format(args.model_name)
        try:
            os.makedirs(args.eval_dir)
        except OSError:
            pass
        if args.random_seed is None:
            args.random_seed = random.randint(1, 10000)
        return args


class ArgParserCGAN(GeneralArgParser):
    """Settings for CGAN model."""
    def __init__(self):
        super(ArgParserCGAN, self).__init__()

    def model_settings(self):
        """Model related settings."""

        super().model_settings()

        # general settings
        self.parser.add_argument('--c-dim', type=int, default=10,
                                 help='Size of condition vector. ')

    def parse_args(self):
        """Parse command line arguments.

        Returns
        -------
        args : :class:`argparse.Namespace`
            Command line arguments.

        """

        args = super().parse_args()
        args.model_name = 'CGAN'
        args.eval_dir = 'eval_{}'.format(args.model_name)
        try:
            os.makedirs(args.eval_dir)
        except OSError:
            pass
        if args.manual_seed is None:
            args.manual_seed = random.randint(1, 10000)
        return args


class ArgParserDCGAN(GeneralArgParser):
    """Settings for DCGAN model."""
    def __init__(self):
        super(ArgParserDCGAN, self).__init__()

    def model_settings(self):
        """Model related settings."""

        super().model_settings()
        self.parser.add_argument('--g-fm-dim', type=int, default=64,
                                 help='Size of feature map in generator. '
                                      'Used in DCGAN.')
        self.parser.add_argument('--d-fm-dim', type=int, default=64,
                                 help='Size of feature map in discriminator. '
                                      'Used in DCGAN.')

    def parse_args(self):
        """Parse command line arguments.

        Returns
        -------
        args : :class:`argparse.Namespace`
            Command line arguments.

        """

        args = super().parse_args()
        args.model_name = 'DCGAN'
        args.eval_dir = 'eval_{}'.format(args.model_name)
        try:
            os.makedirs(args.eval_dir)
        except OSError:
            pass
        if args.random_seed is None:
            args.random_seed = random.randint(1, 10000)
        return args
