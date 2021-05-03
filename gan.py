# =============================================================================
# ================================== Import ===================================
# =============================================================================
import torch
import argparse
import torch.optim as optim
from learn.train import train
from learn.data import dataloader
from models.generator import InfoGan1C28G, InfoGan3C32G
from models.discriminator import InfoGan1C28D, InfoGan3C32D


# =============================================================================
# =============================== Entry Point =================================
# =============================================================================
def parse_args():
    """Parse command line arguments.

    Returns
    -------
    args : :class:`argparse.Namespace`
        Command line arguments.

    """

    pparser = argparse.ArgumentParser()
    # device configuration
    pparser.add_argument('--cuda', type=int, default=0,
                         help='Which cuda to use.')
    # model settings
    pparser.add_argument('--z-dim', type=int, default=64, help='Noise dim.')
    # optimization settings
    pparser.add_argument('--lr-d', type=float, default=2e-4,
                         help='Discriminator learning rate.')
    pparser.add_argument('--lr-g', type=float, default=1e-3,
                         help='Generator learning rate.')
    pparser.add_argument('--batch-size', type=int, default=64,
                         help='Minibatch size')
    pparser.add_argument('--epochs', type=int, default=50, help='Epochs')
    # dataset settings
    pparser.add_argument('--dataset', type=str, default='mnist',
                         choices=['mnist', 'fashion-mnist', 'svhn', 'celeba',
                                  'cifar10', 'stl10'], help='Dataset to use.')
    args = pparser.parse_args()
    args.device = 'cuda:{}'.format(args.cuda) \
        if torch.cuda.is_available() else 'cpu'
    return args


def main(args):
    """Train/evaluate discriminator and generators.

    Parameters
    ----------
    args :
        Command line arguments.

    """

    if args.dataset in ['mnist', 'fashion-mnist']:
        G = InfoGan1C28G(z_dim=args.z_dim, img_channels=1).to(args.device)
        D = InfoGan1C28D(img_channels=1, output_dim=1).to(args.device)
    elif args.dataset in ['svhn', 'celeba']:
        G = InfoGan3C32G(z_dim=args.z_dim, img_channels=3).to(args.device)
        D = InfoGan3C32D(img_channels=3, output_dim=1).to(args.device)
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr_d)
    G_optimizer = optim.Adam(G.parameters(), lr=args.lr_g)
    tr_set, _ = dataloader(args.batch_size, args.dataset)
    train(D, G, D_optimizer, G_optimizer, tr_set, args)


if __name__ == "__main__":
    main(parse_args())
