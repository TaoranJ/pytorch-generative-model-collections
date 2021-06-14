# =============================================================================
# ================================== Import ===================================
# =============================================================================
import torch
import random
import torch.optim as optim
from learn.data import dataloader
from learn.train import train_gan
from learn.utils import weights_init
import torch.backends.cudnn as cudnn
from models.generator import G_DCGAN
from models.discriminator import D_DCGAN
from learn.settings import ArgParserDCGAN


# =============================================================================
# =============================== Entry Point =================================
# =============================================================================
def main(args):
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    cudnn.benchmark = True
    tr_set, te_set = dataloader(args.batch_size, args.dataset, args)
    D = D_DCGAN(img_channels=args.in_channels, feature_map_dim=args.d_fm_dim,
                ngpu=args.ngpu).to(args.device)
    D.apply(weights_init)
    G = G_DCGAN(latent_dim=args.z_dim, feature_map_dim=args.g_fm_dim,
                img_channels=args.in_channels, ngpu=args.ngpu).to(args.device)
    G.apply(weights_init)
    optimizerD = optim.Adam(D.parameters(), lr=args.lr_d,
                            betas=(args.beta1, args.beta2))
    optimizerG = optim.Adam(G.parameters(), lr=args.lr_g,
                            betas=(args.beta1, args.beta2))
    train_gan(D, G, optimizerD, optimizerG, tr_set, args)


# if args.dataset in ['imagenet', 'folder', 'lfw']:
# elif args.dataset == 'lsun':
#     classes = [c + '_train' for c in args.classes.split(',')]
#     dataset = dset.LSUN(root=args.dataroot, classes=classes,
#                         transform=transforms.Compose([
#                             transforms.Resize(args.imageSize),
#                             transforms.CenterCrop(args.imageSize),
#                             transforms.ToTensor(),
#                             transforms.Normalize((0.5, 0.5, 0.5),
#                                                  (0.5, 0.5, 0.5)),
#                         ]))
# elif args.dataset == 'fake':
#     dataset = dset.FakeData(image_size=(3, args.imageSize, args.imageSize),
#                             transform=transforms.ToTensor())


if __name__ == "__main__":
    args = ArgParserDCGAN().parse_args()
    print(args)
    main(args)
