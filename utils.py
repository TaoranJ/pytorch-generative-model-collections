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

    imageio.mimsave(name + '.gif', imgs, fps=2)


def loss_monitor(hist, name):
    """Draw loss vs. steps figure."""
    x = range(len(hist['D_loss']))

    y1 = hist['D_loss']
    y2 = hist['G_loss']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Steps')
    plt.ylabel('Loss')

    plt.legend(loc=4)
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

    """

    num_pics, c, h, w = images.shape
    column_length = num_pics // row_length
    if c == 1:
        img = np.zeros((h * column_length, w * row_length), dtype=np.uint8)
        for idx, image in enumerate(images):
            row_idx, column_idx = idx // row_length, idx % row_length
            bottom, left = row_idx * h, column_idx * w
            img[bottom: bottom + h, left: left + w] = image[0, :, :]
        return img
    elif c in [3, 4]:
        pass
    else:
        raise ValueError('in merge(images,size) images parameter ''must have '
                         'dimensions: HxW or HxWx3 or HxWx4')
