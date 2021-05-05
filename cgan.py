# =============================================================================
# ================================== Import ===================================
# =============================================================================
import torch.optim as optim
from learn.data import dataloader
from learn.train import train_cgan
from learn.utils import parse_args
from models.generator import G_InfoGan_CGAN_1C28
from models.discriminator import D_InfoGan_CGAN_1C28


# =============================================================================
# =============================== Entry Point =================================
# =============================================================================
def main(args):
    """Train/evaluate discriminator and generators.

    Parameters
    ----------
    args :
        Command line arguments.

    """

    if args.dataset in ['mnist', 'fashion-mnist']:
        G = G_InfoGan_CGAN_1C28(z_dim=args.z_dim, img_channels=1,
                                c_dim=args.c_dim).to(
                args.device)
        D = D_InfoGan_CGAN_1C28(img_channels=1, output_dim=1,
                                c_dim=args.c_dim).to(args.device)
    elif args.dataset in ['svhn', 'celeba']:
        pass
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr_d,
                             betas=(args.beta1, args.beta2))
    G_optimizer = optim.Adam(G.parameters(), lr=args.lr_g,
                             betas=(args.beta1, args.beta2))
    tr_set, _ = dataloader(args.batch_size, args.dataset)
    train_cgan(D, G, D_optimizer, G_optimizer, tr_set, args)


if __name__ == "__main__":
    args = parse_args()
    args.model_name = 'CGAN'
    main(args)
