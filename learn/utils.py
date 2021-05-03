# =============================================================================
# ================================== Import ===================================
# =============================================================================
import imageio
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# ================================== Utils ====================================
# =============================================================================
def generate_animation(name, imgs):
    """Animation for generative models.

    Parameters
    ----------
    name : str
        Name for the .gif animation.

    imgs : int
        Collections of imgs.

    """

    imageio.mimwrite(name + '.gif', imgs, duration=0.5, loop=1)


def loss_monitor(per_batch_loss, name):
    """Draw loss vs. steps figure.

    per_batch_loss : :class:`collections.defaultdict`
        Record of loss per batch.

    name : str
        Figure name.

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

    plt.savefig(name)
    plt.close()


def concat_images(images, row_length):
    """Contact one batch of generated results in one image.

    Parameters
    ----------
    images :
        A batch of generated images, of shape (batch_size, c, h, w).
    row_length : int
        # of images per row.

    Returns
    -------
    img : :class:`numpy.array`
        An image tensor of shape (h * batch_size, w * batch_size, channels) or
        (h * batch_size, w * batch_size) if channles = 1.

    """

    num_pics, c, h, w = images.shape
    column_length = num_pics // row_length
    img = np.zeros((c, h * column_length, w * row_length), dtype=np.uint8)
    for idx, image in enumerate(images):
        row_idx, column_idx = idx // row_length, idx % row_length
        bottom, left = row_idx * h, column_idx * w
        for c_ in range(image.shape[0]):
            img[c_, bottom: bottom + h, left: left + w] = image[c_, :, :]
    img = img.transpose(1, 2, 0)
    if c == 1:
        img = np.squeeze(img, -1)
    return img
