# =============================================================================
# ================================== Import ===================================
# =============================================================================
import os
import torch
import imageio
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
    imgs_per_column = int(np.ceil(num_pics / imgs_per_row))
    # img = np.zeros((c, h * column_length, w * imgs_per_row), dtype=np.uint8)
    img = np.zeros((c, h * imgs_per_column, w * imgs_per_row))
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

    imageio.mimwrite(os.path.join(args.eval_dir,
                                  '_'.join([args.model_name, name + '.gif'])),
                     imgs, duration=0.5, loop=1)


def loss_monitor(record, name, args):
    """Draw record vs. steps figure.

    record : :class:`collections.defaultdict`
        Record per batch/epoch.
    name : str
        Figure name.
    args : :class:`argparse.Namespace`
        Argument parser.

    """

    for key, value in record.items():
        plt.plot(range(len(value)), value, label=key)

    if 'step' in name:
        plt.xlabel('Steps')
    elif 'epoch' in name:
        plt.xlabel('Epochs')
    if 'loss' in name:
        plt.ylabel('Loss')
    elif 'prob' in name:
        plt.ylabel('Probability')

    plt.legend(loc=2)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(args.eval_dir, '_'.join([args.model_name, name])))
    plt.close()


def weights_init(submodule):
    """Initialize model's parameters."""
    classname = submodule.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(submodule.weight, 0.0, 0.02)
        if submodule.bias is not None:
            nn.init.zeros_(submodule.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(submodule.weight, 1.0, 0.02)
        if submodule.bias is not None:
            nn.init.zeros_(submodule.bias)

#     for m in net.modules():
#         if isinstance(m, nn.Conv2d):
#             m.weight.data.normal_(0, 0.02)
#             m.bias.data.zero_()
#         elif isinstance(m, nn.ConvTranspose2d):
#             m.weight.data.normal_(0, 0.02)
#             m.bias.data.zero_()
#         elif isinstance(m, nn.Linear):
#             m.weight.data.normal_(0, 0.02)
#             m.bias.data.zero_()
