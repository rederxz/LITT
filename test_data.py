import torch
import numpy as np
import matplotlib.pyplot as plt

from data import get_LITT_dataset, load_data_v2
import compressed_sensing as cs
from utils import to_tensor_format


# using pytorch dataset
train_dataset = get_LITT_dataset(data_root='../LITT_data/', split='test', nt_network=6,
                                 single_echo=True, acc=6.0, sample_n=8,
                                 transform=None)
for key, img in train_dataset[0].items():
    print(key)
    img = img.numpy()
    print(img.shape)
    print(np.min(img))
    print(np.max(img))
    amplitude = np.sqrt(img[0] ** 2 + img[1] ** 2)
    print(np.min(amplitude))
    print(np.max(amplitude))
    print()
    fig, axs = plt.subplots(2, 3)
    for i, ax in enumerate(axs.ravel()):
        ax.imshow(amplitude[..., i])
    plt.show()


# using customized generator and pred_input
def iterate_minibatch(data, batch_size, shuffle=True):
    """
    yield batches in the given dataset
    :param data: data set with shape [n_samples, t, x, y]
    :param batch_size: ...
    :param shuffle: ...
    :return:
    """
    n = data.shape[0]
    if shuffle:
        data = np.random.permutation(data)
    for i in range(0, n, batch_size):
        yield data[i:i + batch_size]


def prep_input(im, **kwargs):
    """
    Under-sample the batch, then reformat them into what the network accepts.
    :param im: tensor of shape [n_samples, t, x, y]
    :param acc: accelerate rate
    :param sample_n:
    :return: i_und, k_und, mask, i_gt [n_samples, n_channels, x, y, t]
    """
    mask = cs.cartesian_mask(im.shape, **kwargs)
    im_und, k_und = cs.undersample(im, mask)

    # convert to float32 tensor (original format is float64)
    im_gnd_l = torch.from_numpy(to_tensor_format(im)).float()
    im_und_l = torch.from_numpy(to_tensor_format(im_und)).float()
    k_und_l = torch.from_numpy(to_tensor_format(k_und)).float()
    mask_l = torch.from_numpy(to_tensor_format(mask, mask=True)).float()

    return im_und_l, k_und_l, mask_l, im_gnd_l


data = load_data_v2('../LITT_data/', 'test', nt_network=6)
data = data.transpose((0, 3, 1, 2))
for im in iterate_minibatch(data=data, batch_size=1, shuffle=False):
    im_u, k_u, mask, gnd = prep_input(im, acc=6.0, sample_n=8)
    group = [im_u.numpy()[0], k_u.numpy()[0], mask.numpy()[0], gnd.numpy()[0]]
    for item in group:
        print(item.shape)
        amplitude = np.sqrt(item[0] ** 2 + item[1] ** 2)
        fig, axs = plt.subplots(2, 3)
        for i, ax in enumerate(axs.ravel()):
            ax.imshow(amplitude[..., i])
        plt.show()
        break
