# =============================================================================
# ================================== Import ===================================
# =============================================================================
import time
import torch
import datetime
import numpy as np
from torch import nn
from collections import defaultdict
from learn.utils import loss_monitor, concat_images, generate_animation


# =============================================================================
# ============================= Training Process ==============================
# =============================================================================
def train(D, G, D_optimizer, G_optimizer, train_set, args):
    """Training process for GAN.

    Parameters
    ----------
    D : :class:`models.[gan].Discriminator`
        Discriminator.
    G : :class:`models.[gan].Generator`
        Generator.
    D_optimizer : :class:`torch.optim`
        Optimizer for discriminator.
    G_optimizer : :class:`torch.optim`
        Optimizer for generator.
    train_set : :class:`torch.utils.data.dataloader.DataLoader`
        Training set.
    args : :class:`argparse.Namespace`
        Argument parser.

    Returns
    -------
    D : :class:`models.[gan].Discriminator`
        Discriminator.
    G : :class:`models.[gan].Generator`
        Generator.

    """

    # record generated imgs every epoch
    vis_imgs = []
    fixed_z = torch.rand((args.batch_size, args.z_dim)).to(args.device)
    # labels for real/fake images
    y_real = torch.ones(args.batch_size, 1).to(args.device)
    y_fake = torch.zeros(args.batch_size, 1).to(args.device)
    # loss function
    criterion = nn.BCELoss()

    start_time = time.time()
    per_batch_loss = defaultdict(list)
    for epoch in range(1, args.epochs + 1):
        per_epoch_loss = defaultdict(list)
        for (imgs, _) in train_set:
            D.train()
            G.train()

            # (batch_size, img_channels, img_h, img_w)
            real_imgs = imgs.to(args.device)
            z = torch.rand((args.batch_size, args.z_dim)).to(args.device)

            # train discriminator
            D_optimizer.zero_grad()
            # real imgs -> discriminator
            D_real = D(real_imgs)
            D_real_loss = criterion(D_real, y_real)
            # noise -> generator -> fake imgs -> discriminator
            fake_imgs = G(z)
            D_fake = D(fake_imgs.detach())
            D_fake_loss = criterion(D_fake, y_fake)
            # loss -> upgrade discriminator parameters
            D_loss = D_real_loss + D_fake_loss
            D_loss.backward()
            D_optimizer.step()
            per_batch_loss['D_loss'].append(D_loss.item())
            per_epoch_loss['D_loss'].append(D_loss.item())

            # train generator
            G_optimizer.zero_grad()
            # z -> fake imgs -> discriminator
            fake_imgs = G(z)
            D_fake = D(fake_imgs)
            G_loss = criterion(D_fake, y_real)
            per_batch_loss['G_loss'].append(G_loss.item())
            per_epoch_loss['G_loss'].append(G_loss.item())
            # loss -> upgrade generator parameters
            G_loss.backward()
            G_optimizer.step()

        print('Epochs: {:03d}/{:03d}, Elapsed time: {}, '
              'Avg_D_loss: {:.4f}, Avg_G_loss: {:.4f}'.format(
                  epoch, args.epochs,
                  datetime.timedelta(seconds=int(time.time() - start_time)),
                  np.mean(per_epoch_loss['D_loss']),
                  np.mean(per_epoch_loss['G_loss'])))
        per_epoch_loss.clear()

        # try the generator every epoch
        G.eval()
        with torch.no_grad():
            # un-transform (0.5, 0.5)
            fake_img = ((G(fixed_z) + 1) / 2 * 255).to('cpu').numpy().astype(
                    np.uint8)
            vis_imgs.append(concat_images(fake_img, 8))

    torch.save(G.state_dict(), 'G.pkl')
    torch.save(D.state_dict(), 'D.pkl')
    generate_animation('generator_result_animation', vis_imgs)
    loss_monitor(per_batch_loss, 'steps_vs_loss.png')

    return D, G
