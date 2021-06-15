# =============================================================================
# ================================== Import ===================================
# =============================================================================
import torch
import random
import torch.nn as nn
import torch.optim as optim
from learn.data import dataloader
from learn.train import train_gan
from learn.utils import weights_init
import torch.backends.cudnn as cudnn
from learn.settings import ArgParserDCGAN
from models.generator import G_DCGAN_64
from models.discriminator import D_DCGAN_64


# =============================================================================
# =============================== Entry Point =================================
# =============================================================================
def main(args):
    """Train/evaluate discriminator and generators.

    Parameters
    ----------
    args : :class:`argparse.Namespace`
        Command line arguments.

    """

    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    cudnn.benchmark = True
    # datasets
    tr_set, _ = dataloader(args.batch_size, args.dataset, args)
    # models
    D = D_DCGAN_64(img_channels=args.img_channels,
                   feature_map_dim=args.d_fm_dim,
                   ngpu=args.ngpu).to(args.device)
    if (args.device.type == 'cuda') and (args.ngpu > 1):
        D = nn.DataParallel(D, list(range(args.ngpu)))
    D.apply(weights_init)
    G = G_DCGAN_64(latent_dim=args.z_dim, feature_map_dim=args.g_fm_dim,
                   img_channels=args.img_channels,
                   ngpu=args.ngpu).to(args.device)
    if (args.device.type == 'cuda') and (args.ngpu > 1):
        G = nn.DataParallel(G, list(range(args.ngpu)))
    G.apply(weights_init)
    # optimizers
    optimizerD = optim.Adam(D.parameters(), lr=args.lr_d,
                            betas=(args.beta1, args.beta2))
    optimizerG = optim.Adam(G.parameters(), lr=args.lr_g,
                            betas=(args.beta1, args.beta2))
    # training
    train_gan(D, G, optimizerD, optimizerG, tr_set, args)


if __name__ == "__main__":
    args = ArgParserDCGAN().parse_args()
    print(args)
    main(args)
