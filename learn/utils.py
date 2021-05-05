# =============================================================================
# ================================== Import ===================================
# =============================================================================
import torch
import imageio
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


# =============================================================================
# =========================== Result Visualization ============================
# =============================================================================
def generator_vis(G, *inputs):
    """Use generator to generate data on the noise signals."""

    G.eval()
    with torch.no_grad():
        fake_imgs = G(inputs[0]) if len(inputs) == 1 else G(*inputs)
        fake_imgs = fake_imgs.to('cpu').numpy()
        return concat_images(fake_imgs, 8)


def concat_images(images, imgs_per_row):
    """Contact one batch of generated results in one image.

    Parameters
    ----------
    images :
        A batch of generated images, of shape (batch_size, c, h, w).
    imgs_per_row : int
        # of images per row.

    Returns
    -------
    img : :class:`numpy.array`
        An image tensor of shape (h * batch_size, w * batch_size, channels) or
        (h * batch_size, w * batch_size) if channles = 1.

    """

    num_pics, c, h, w = images.shape
    column_length = int(np.ceil(num_pics / imgs_per_row))
    # img = np.zeros((c, h * column_length, w * imgs_per_row), dtype=np.uint8)
    img = np.zeros((c, h * column_length, w * imgs_per_row))
    for idx, image in enumerate(images):
        r_idx, c_idx = idx // imgs_per_row, idx % imgs_per_row
        top, left = r_idx * h, c_idx * w
        for c_ in range(image.shape[0]):
            img[c_, top: top + h, left: left + w] = image[c_, :, :]
    img = img.transpose(1, 2, 0)
    if c == 1:
        img = np.squeeze(img, -1)

    img = (img + 1) / 2  # un-transform (-1, 1) -> (0, 1)
    img = np.round(img * 255).astype(np.uint8)
    return img


def generate_animation(name, imgs, args):
    """Animation for generative models.

    Parameters
    ----------
    name : str
        Name for the .gif animation.
    imgs : int
        Collections of imgs.
    args : :class:`argparse.Namespace`
        Argument parser.

    """

    imageio.mimwrite('_'.join([args.model_name, name + '.gif']), imgs,
                     duration=0.5, loop=1)


def loss_monitor(per_batch_loss, name, args):
    """Draw loss vs. steps figure.

    per_batch_loss : :class:`collections.defaultdict`
        Record of loss per batch.
    name : str
        Figure name.
    args : :class:`argparse.Namespace`
        Argument parser.

    """

    x = range(len(per_batch_loss['D_loss']))

    y1 = per_batch_loss['D_loss']
    y2 = per_batch_loss['G_loss']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Steps')
    plt.ylabel('Loss')

    plt.legend(loc=2)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig('_'.join([args.model_name, name]))
    plt.close()


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


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
    pparser.add_argument('--c-dim', type=int, default=10,
                         help='Condition dim.')
    # optimization settings
    pparser.add_argument('--lr-d', type=float, default=2e-4,
                         help='Discriminator learning rate.')
    pparser.add_argument('--lr-g', type=float, default=2e-4,
                         help='Generator learning rate.')
    pparser.add_argument('--batch-size', type=int, default=64,
                         help='Minibatch size')
    pparser.add_argument('--epochs', type=int, default=50, help='Epochs')
    pparser.add_argument('--d-steps', type=int, default=1,
                         help='Train discriminator d steps every time.')
    pparser.add_argument('--g-steps', type=int, default=1,
                         help='Train generator g steps every time.')
    pparser.add_argument('--beta1', type=float, default=0.5,
                         help='Beta 1 of Adam.')
    pparser.add_argument('--beta2', type=float, default=0.999,
                         help='Beta 2 of Adam.')
    # dataset settings
    pparser.add_argument('--dataset', type=str, default='mnist',
                         choices=['mnist', 'fashion-mnist', 'svhn', 'celeba',
                                  'cifar10', 'stl10'], help='Dataset to use.')
    args = pparser.parse_args()
    args.device = 'cuda:{}'.format(args.cuda) \
        if torch.cuda.is_available() else 'cpu'
    print(args)
    return args
