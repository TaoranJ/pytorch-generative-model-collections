# =============================================================================
# ================================== Import ===================================
# =============================================================================
import time
import torch
import datetime
import numpy as np
from torch import nn
from collections import defaultdict
from learn.utils import loss_monitor, generator_vis, generate_animation


# =============================================================================
# ============================= Training Process ==============================
# =============================================================================
def train_gan(D, G, D_optimizer, G_optimizer, train_set, args):
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

    # test geneartor's capability every epoch
    vis_imgs = []
    fixed_z = torch.rand(args.batch_size, args.z_dim).to(args.device)
    # labels for real/fake images
    y_real = torch.ones(args.batch_size, 1).to(args.device)
    y_fake = torch.zeros(args.batch_size, 1).to(args.device)
    # loss function
    criterion = nn.BCELoss()
    # record loss for both discriminator and generator
    per_batch_loss, per_epoch_loss = defaultdict(list), defaultdict(list)

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = defaultdict(list)  # monitor loss every epoch
        # set both G and D to training mode
        D.train()
        G.train()
        for (imgs, _) in train_set:
            real_imgs = imgs.to(args.device)  # (batch_size, in_channel, h, w)

            # train discriminator
            D_optimizer.zero_grad()
            # real imgs -> discriminator
            D_real_loss = criterion(D(real_imgs), y_real)
            # noise -> generator -> fake imgs -> discriminator
            z = torch.rand((args.batch_size, args.z_dim)).to(args.device)
            D_fake_loss = criterion(D(G(z).detach()), y_fake)
            D_loss = D_real_loss + D_fake_loss
            D_loss.backward()
            D_optimizer.step()  # only update discriminator
            per_batch_loss['D_loss'].append(D_loss.item())
            epoch_loss['D_loss'].append(D_loss.item())

            # train generator
            G_optimizer.zero_grad()
            # z -> fake imgs -> discriminator
            z = torch.rand((args.batch_size, args.z_dim)).to(args.device)
            G_loss = criterion(D(G(z)), y_real)
            G_loss.backward()
            G_optimizer.step()  # only update generator
            per_batch_loss['G_loss'].append(G_loss.item())
            epoch_loss['G_loss'].append(G_loss.item())

        # record loss
        epoch_d_loss = np.mean(epoch_loss['D_loss'])
        epoch_g_loss = np.mean(epoch_loss['G_loss'])
        per_epoch_loss['D_loss'].append(epoch_d_loss)
        per_epoch_loss['G_loss'].append(epoch_g_loss)
        print('Epochs: {:03d}/{:03d}, Elapsed time: {}, '
              'Avg_D_loss: {:.4f}, Avg_G_loss: {:.4f}'.format(
                  epoch, args.epochs,
                  datetime.timedelta(seconds=int(time.time() - start_time)),
                  epoch_d_loss, epoch_g_loss))
        epoch_loss.clear()
        vis_imgs.append(generator_vis(G, fixed_z))

    torch.save({'generator': G.state_dict(),
                'discriminator': D.state_dict}, args.model_name + '.pt')
    generate_animation('generator_result_animation', vis_imgs, args)
    loss_monitor(per_batch_loss, 'loss_vs_steps.png', args)
    loss_monitor(per_epoch_loss, 'loss_vs_epochs.png', args)
    return D, G


def train_cgan(D, G, D_optimizer, G_optimizer, train_set, args):
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

    # test geneartor's capability every epoch
    vis_imgs = []
    fixed_z = torch.rand(args.batch_size, args.z_dim).to(args.device)
    fixed_c = torch.randint(low=0, high=args.c_dim - 1,
                            size=(args.batch_size, 1))
    print(fixed_c.squeeze(-1))
    fixed_c = torch.zeros(args.batch_size, args.c_dim).scatter_(
            1, fixed_c, 1).to(args.device)
    # labels for real/fake images
    y_real = torch.ones(args.batch_size, 1).to(args.device)
    y_fake = torch.zeros(args.batch_size, 1).to(args.device)
    # loss function
    criterion = nn.BCELoss()
    # record loss for both discriminator and generator
    per_batch_loss, per_epoch_loss = defaultdict(list), defaultdict(list)

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = defaultdict(list)  # monitor loss every epoch
        # set both G and D to training mode
        D.train()
        G.train()
        for (imgs, cnd) in train_set:
            # (batch_size, in_channel, h, w) and one-hot encoding labels
            real_imgs = imgs.to(args.device)
            cnd_vec = torch.zeros(args.batch_size, args.c_dim).scatter_(
                    1, cnd.unsqueeze(-1), 1).to(args.device)
            cnd_img = cnd_vec.unsqueeze(-1).unsqueeze(-1).expand(
                    args.batch_size, args.c_dim, imgs.size(2), imgs.size(3))

            # train discriminator
            D_optimizer.zero_grad()
            # real imgs -> discriminator
            D_real_loss = criterion(D(real_imgs, cnd_img), y_real)
            # noise -> generator -> fake imgs -> discriminator
            z = torch.rand((args.batch_size, args.z_dim)).to(args.device)
            D_fake_loss = criterion(D(G(z, cnd_vec).detach(), cnd_img), y_fake)
            D_loss = D_real_loss + D_fake_loss
            D_loss.backward()
            D_optimizer.step()  # only update discriminator
            per_batch_loss['D_loss'].append(D_loss.item())
            epoch_loss['D_loss'].append(D_loss.item())

            # train generator
            G_optimizer.zero_grad()
            # z -> fake imgs -> discriminator
            z = torch.rand((args.batch_size, args.z_dim)).to(args.device)
            G_loss = criterion(D(G(z, cnd_vec), cnd_img), y_real)
            G_loss.backward()
            G_optimizer.step()  # only update generator
            per_batch_loss['G_loss'].append(G_loss.item())
            epoch_loss['G_loss'].append(G_loss.item())

        # record loss
        epoch_d_loss = np.mean(epoch_loss['D_loss'])
        epoch_g_loss = np.mean(epoch_loss['G_loss'])
        per_epoch_loss['D_loss'].append(epoch_d_loss)
        per_epoch_loss['G_loss'].append(epoch_g_loss)
        print('Epochs: {:03d}/{:03d}, Elapsed time: {}, '
              'Avg_D_loss: {:.4f}, Avg_G_loss: {:.4f}'.format(
                  epoch, args.epochs,
                  datetime.timedelta(seconds=int(time.time() - start_time)),
                  epoch_d_loss, epoch_g_loss))
        epoch_loss.clear()
        vis_imgs.append(generator_vis(G, fixed_z, fixed_c))

    torch.save({'generator': G.state_dict(),
                'discriminator': D.state_dict}, args.model_name + '.pt')
    generate_animation('generator_result_animation', vis_imgs, args)
    loss_monitor(per_batch_loss, 'loss_vs_steps.png', args)
    loss_monitor(per_epoch_loss, 'loss_vs_epochs.png', args)
    return D, G


def train_gan_original(D, G, D_optimizer, G_optimizer, train_set, args):
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

    # test geneartor's capability every epoch
    vis_imgs = []
    fixed_z = torch.rand((args.batch_size, args.z_dim)).to(args.device)
    # labels for real/fake images
    y_real = torch.ones(args.batch_size, 1).to(args.device)
    y_fake = torch.zeros(args.batch_size, 1).to(args.device)
    # loss function
    criterion = nn.BCELoss()
    # record per k steps loss for both discriminator and generator
    per_k_steps_loss = defaultdict(list)

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = defaultdict(list)  # monitor loss every epoch

        data_samples, d_train_loss = [], []
        # set both G and D to training mode
        D.train()
        G.train()
        for d_k, (imgs, _) in enumerate(train_set, start=1):
            # train discriminator d_steps batches
            data_samples.append(imgs.to(args.device))
            if d_k % args.d_steps != 0:
                continue
            for real_imgs in data_samples:  # (batch_size, in_channel, h, w)
                D_optimizer.zero_grad()
                # real imgs -> discriminator
                D_real = D(real_imgs)
                D_real_loss = criterion(D_real, y_real)
                # noise -> generator -> fake imgs -> discriminator
                z = torch.rand((args.batch_size, args.z_dim)).to(args.device)
                fake_imgs = G(z).detach()
                D_fake = D(fake_imgs)
                D_fake_loss = criterion(D_fake, y_fake)
                # loss -> upgrade discriminator parameters
                D_loss = D_real_loss + D_fake_loss
                D_loss.backward()
                D_optimizer.step()  # only update discriminator
                d_train_loss.append(D_loss.item())
            per_k_steps_loss['D_loss'].append(np.mean(d_train_loss))
            epoch_loss['D_loss'].append(np.mean(d_train_loss))
            d_train_loss.clear()
            data_samples.clear()

            # train generator g_step times
            g_train_loss = []
            for _ in range(args.g_steps):
                G_optimizer.zero_grad()
                # z -> fake imgs -> discriminator
                z = torch.rand((args.batch_size, args.z_dim)).to(args.device)
                fake_imgs = G(z)
                D_fake = D(fake_imgs)
                G_loss = criterion(D_fake, y_real)
                # loss -> upgrade generator parameters
                G_loss.backward()
                G_optimizer.step()  # only update generator
                g_train_loss.append(G_loss.item())
            per_k_steps_loss['G_loss'].append(np.mean(g_train_loss))
            epoch_loss['G_loss'].append(np.mean(g_train_loss))

        print('Epochs: {:03d}/{:03d}, Elapsed time: {}, '
              'Avg_D_loss: {:.4f}, Avg_G_loss: {:.4f}'.format(
                  epoch, args.epochs,
                  datetime.timedelta(seconds=int(time.time() - start_time)),
                  np.mean(epoch_loss['D_loss']),
                  np.mean(epoch_loss['G_loss'])))
        epoch_loss.clear()
        vis_imgs.append(generator_vis(G, fixed_z))

    torch.save({'generator': G.state_dict(),
                'discriminator': D.state_dict}, 'model.pt')
    generate_animation('generator_result_animation', vis_imgs, args)
    loss_monitor(per_k_steps_loss, 'loss_vs_steps.png', args)

    return D, G
