# =============================================================================
# =============================== Description =================================
# =============================================================================

"""GAN paper proposed the GAN idea instead of a specific architecture to solve
some specific problems. So here we used the architecture in the InfoGAN
paper."""

# =============================================================================
# ================================== Import ===================================
# =============================================================================
import os
import time
import utils
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataloader import dataloader


class Generator(nn.Module):
    """InfoGAN generator."""
    def __init__(self, z_dim, img_channels, dataset='mnist'):
        super(Generator, self).__init__()

        if dataset == 'mnist':
            # fc 1: (batch_size, z_dim -> 1024)
            self.fc1 = nn.Sequential(
                    nn.Linear(z_dim, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU())
            # fc 2: (batch_size, 1024 -> 7 * 7 * 128)
            self.fc2 = nn.Sequential(
                    nn.Linear(1024, 7 * 7 * 128),
                    nn.BatchNorm1d(7 * 7 * 128),
                    nn.ReLU())
            # un-conv 1: (batch_size, 128 -> 64, 7 -> 14, 7 -> 14)
            self.dconv1 = nn.Sequential(
                    nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                       kernel_size=[4, 4], stride=2,
                                       padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU())
            # un-conv 2: (batch_size, 128 -> 1, 14 -> 28, 14 -> 28)
            self.dconv2 = nn.ConvTranspose2d(in_channels=64,
                                             out_channels=img_channels,
                                             kernel_size=[4, 4], stride=2,
                                             padding=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(x.size(0), 128, 7, 7)
        x = self.dconv1(x)
        x = self.dconv2(x)
        return x


class Discriminator(nn.Module):
    """InfoGAN's discriminator"""
    def __init__(self, img_channels, output_dim, dataset='mnist'):
        super(Discriminator, self).__init__()

        if dataset == 'mnist':
            # conv1 (batch_size, 1 -> 64, 28 -> 14, 28 -> 14)
            self.conv1 = nn.Sequential(
                    nn.Conv2d(in_channels=img_channels, out_channels=64,
                              kernel_size=[4, 4], stride=2, padding=1),
                    nn.LeakyReLU(0.2))
            # conv2 (batch_size, 64 -> 128, 14 -> 7, 14 -> 7)
            self.conv2 = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=128,
                              kernel_size=[4, 4], stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.2))
            # fc1 (batch_size, 128 * 7 * 7) -> (batch_size, 1024)
            self.fc1 = nn.Sequential(
                    nn.Linear(128 * 7 * 7, 1024),
                    nn.BatchNorm1d(1024),
                    nn.LeakyReLU(0.2))
            # fc2 (batch_size, 1024) -> (batch_size, 1)
            self.fc2 = nn.Sequential(
                    nn.Linear(1024, output_dim),
                    nn.Sigmoid())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1(torch.flatten(x, start_dim=1))
        x = self.fc2(x)
        return x


class GAN(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.sample_num = 100
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.input_size = args.input_size
        self.z_dim = 62

        # load dataset
        self.data_loader = dataloader(self.dataset, self.input_size,
                                      self.batch_size)
        data = self.data_loader.__iter__().__next__()[0]

        # networks init
        self.G = Generator(input_dim=self.z_dim, output_dim=data.shape[1],
                           input_size=self.input_size)
        self.D = Discriminator(input_dim=data.shape[1], output_dim=1,
                               input_size=self.input_size)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG,
                                      betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD,
                                      betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('-----------------------------------------------')

        # fixed noise
        self.sample_z_ = torch.rand((self.batch_size, self.z_dim))
        if self.gpu_mode:
            self.sample_z_ = self.sample_z_.cuda()

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.y_real_ = torch.ones(self.batch_size, 1)
        self.y_fake_ = torch.zeros(self.batch_size, 1)
        if self.gpu_mode:
            self.y_real_ = self.y_real_.cuda()
            self.y_fake_ = self.y_fake_.cuda()

        self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            for iter, (x_, _) in enumerate(self.data_loader):
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break

                z_ = torch.rand((self.batch_size, self.z_dim))
                if self.gpu_mode:
                    x_, z_ = x_.cuda(), z_.cuda()

                # update D network
                self.D_optimizer.zero_grad()

                D_real = self.D(x_)
                D_real_loss = self.BCE_loss(D_real, self.y_real_)

                G_ = self.G(z_)
                D_fake = self.D(G_)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)

                D_loss = D_real_loss + D_fake_loss
                self.train_hist['D_loss'].append(D_loss.item())

                D_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                G_ = self.G(z_)
                D_fake = self.D(G_)
                G_loss = self.BCE_loss(D_fake, self.y_real_)
                self.train_hist['G_loss'].append(G_loss.item())

                G_loss.backward()
                self.G_optimizer.step()

                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (iter + 1),
                           self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_loss.item()))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            with torch.no_grad():
                self.visualize_results((epoch+1))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
                                 self.epoch)
        utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

    def visualize_results(self, epoch, fix=True):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_)
        else:
            """ random noise """
            sample_z_ = torch.rand((self.batch_size, self.z_dim))
            if self.gpu_mode:
                sample_z_ = sample_z_.cuda()

            samples = self.G(sample_z_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))
