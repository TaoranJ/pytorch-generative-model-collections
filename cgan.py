"""

Run CGAN with this script.

Examples
--------

.. code-block:: bash

   python cgan.py --dataset mnist --epochs 5 --batch-size 128 --latent-dim 100

"""

# =============================================================================
# ================================== Import ===================================
# =============================================================================
import torch.nn as nn
import torch.optim as optim
from learn.data import dataloader
from learn.train import train_cgan
from learn.settings import ArgParserCGAN
from models.generator import G_InfoGan_CGAN_28, G_InfoGan_CGAN_32
from models.discriminator import D_InfoGan_CGAN_28, D_InfoGan_CGAN_32


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

    tr_set, _ = dataloader(args)
    if args.img_size == 28:
        G = G_InfoGan_CGAN_28(latent_dim=args.latent_dim,
                              img_channels=args.img_channels,
                              c_dim=args.c_dim).to(args.device)
        D = D_InfoGan_CGAN_28(img_channels=args.img_channels,
                              c_dim=args.c_dim).to(args.device)
    elif args.img_size == 32:
        G = G_InfoGan_CGAN_32(latent_dim=args.latent_dim,
                              img_channels=args.img_channels,
                              c_dim=args.c_dim).to(args.device)
        D = D_InfoGan_CGAN_32(img_channels=args.img_channels,
                              c_dim=args.c_dim).to(args.device)
    if (args.device.type == 'cuda') and (args.ngpu > 1):
        D = nn.DataParallel(D, list(range(args.ngpu)))
        G = nn.DataParallel(G, list(range(args.ngpu)))
    # optimizers
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr_d,
                             betas=(args.beta1, args.beta2))
    G_optimizer = optim.Adam(G.parameters(), lr=args.lr_g,
                             betas=(args.beta1, args.beta2))
    train_cgan(D, G, D_optimizer, G_optimizer, tr_set, args)


if __name__ == "__main__":
    args = ArgParserCGAN().parse_args()
    print(args)
    main(args)
