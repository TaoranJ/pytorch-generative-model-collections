# =============================================================================
# ================================== Import ===================================
# =============================================================================
import torch
import argparse
from train import train
import torch.optim as optim
from data import mnist_dataloader
from models.gan import Generator, Discriminator


if __name__ == "__main__":
    pparser = argparse.ArgumentParser()
    # some default values are determined based on InfoGAN paper
    pparser.add_argument('--lr-d', type=float, default=2e-4,
                         help='Discriminator learning rate.')
    pparser.add_argument('--lr-g', type=float, default=1e-3,
                         help='Generator learning rate.')
    pparser.add_argument('--batch-size', type=int, default=64,
                         help='Minibatch size')
    pparser.add_argument('--z-dim', type=int, default=64, help='Noise dim.')
    pparser.add_argument('--dataset', type=str, default='mnist',
                         help='MNIST.')
    pparser.add_argument('--epochs', type=int, default=50, help='Epochs')

    args = pparser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.dataset == 'mnist':
        G = Generator(z_dim=args.z_dim, img_channels=1,
                      dataset='mnist').to(args.device)
        D = Discriminator(img_channels=1, output_dim=1,
                          dataset='mnist').to(args.device)
        D_optimizer = optim.Adam(D.parameters(), lr=args.lr_d)
        G_optimizer = optim.Adam(G.parameters(), lr=args.lr_g)
        train_set, val_set = mnist_dataloader(args.batch_size)
        train(D, G, D_optimizer, G_optimizer, train_set, args)
