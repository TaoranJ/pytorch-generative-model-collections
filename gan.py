# =============================================================================
# ================================== Import ===================================
# =============================================================================
import torch
import argparse
import torch.optim as optim
from learn.train import train
from learn.data import dataloader
from models.generator import InfoGanMnistG
from models.discriminator import InfoGanMnistD


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
                         choices=['mnist', 'fashion-mnist', 'cifar10',
                                  'svhn', 'stl10'],
                         help='Dataset to use.')
    args = pparser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args


def main(args):
    """Train/evaluate discriminator and generators.

    Parameters
    ----------
    args :
        Command line arguments.

    """

    if args.dataset in ['mnist', 'fashion-mnist']:
        G = InfoGanMnistG(z_dim=args.z_dim, img_channels=1).to(args.device)
        D = InfoGanMnistD(img_channels=1, output_dim=1).to(args.device)
        D_optimizer = optim.Adam(D.parameters(), lr=args.lr_d)
        G_optimizer = optim.Adam(G.parameters(), lr=args.lr_g)
        tr_set, _ = dataloader(args.batch_size, args.dataset)
        train(D, G, D_optimizer, G_optimizer, tr_set, args)


if __name__ == "__main__":
    main(parse_args())
