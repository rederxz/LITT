import os
import pathlib

import numpy as np
import torch
from scipy.io import loadmat
from skimage import transform as T
from toolz import curry

import compressed_sensing as cs


def cut_data(input, block_size=(1, 1), cut_edge=None):
    """
    make sure input is divisible by the block size and cut edge area
    :param input: with shape [n_samples, (echo,)nt, x, y]
    :param block_size: [int, int], such as [256, 32]
    :param cut_edge: [0, 32]
    :return:
    """
    if cut_edge is not None:
        if cut_edge[0] != 0:
            input = input[..., cut_edge[0]:-cut_edge[0], :]
        if cut_edge[1] != 0:
            input = input[..., :, cut_edge[1]:-cut_edge[1]]

    # make sure x dim divisible
    start_x, end_x = 0, None
    assert input.shape[-2] >= block_size[0]
    remainder = input.shape[-2] % block_size[0]
    if remainder > 0:
        start_x += remainder // 2
        end_x = - (remainder - remainder // 2)

    # make sure y dim divisible
    start_y, end_y = 0, None
    assert input.shape[-1] >= block_size[1]
    remainder = input.shape[-1] % block_size[1]
    if remainder > 0:
        start_y += remainder // 2
        end_y = - (remainder - remainder // 2)

    input_p = input[..., start_x:end_x, start_y:end_y]

    return input_p


@ curry
def data_aug(img, mask=None, block_size=None, rotation_xy=False, flip_t=False):
    # FIXME in subsampling patterns (like cartesian), random crop of mask in both x and y direction may cause problem
    """
    apply data augmentation to a batch
    :param img: [n, (echo, )nt, x, y]
    :param mask: must be the same shape with img
    :param block_size: [int, int]
    :param rotation_xy: boolean
    :param flip_t: boolean
    :return:
    """
    if mask is not None:
        assert mask.shape == img.shape

    if rotation_xy and np.rand.random() > 0.5:
        k = np.random.randint(0, 4)
        img = np.rot90(img, k=k, axes=(-2, -1))
        if mask is not None:
            mask = np.rot90(mask, k=k, axes=(-2, -1))  # here we also rotate mask the same angle

    if flip_t and np.rand.random() > 0.5:
        img = img[..., ::-1, :, :]
        if mask is not None:
            mask = mask[..., ::-1, :, :]

    if block_size is not None:
        assert img.shape[-2] >= block_size[0] and img.shape[-1] >= block_size[1]

        # cut unimportant edge area
        img = cut_data(img, cut_edge=(0, 32))
        if mask is not None:
            mask = cut_data(mask, cut_edge=(0, 32))

        start_x = np.random.randint(0, img.shape[-2] - block_size[0] + 1)
        start_y = np.random.randint(0, img.shape[-1] - block_size[1] + 1)
        img = img[..., start_x: start_x + block_size[0], start_y: start_y + block_size[1]]
        if mask is not None:
            mask = mask[..., start_x: start_x + block_size[0], start_y: start_y + block_size[1]]

    return img if mask is None else (img, mask)


class LITT(torch.utils.data.dataset.Dataset):
    def __init__(self, mat_file_path, nt_network=1, single_echo=True, acc=6.0, sample_n=8,
                 mask_file_path=None, overlap=False, nt_wait=0, transform=None, img_resize=None):
        """
        build a LITT dataset
        :param mat_file_path: a list, the paths of mat files
        :param nt_network: the number of time frames required
        :param single_echo: if only use one echo
        :param acc: accelerating rate
        :param sample_n: preserve how many center lines (sample_n // 2 each side)
        :param mask_file_path: the path of mask(s) with shape [..., t, x, y]
        :param overlap: if samples overlap with each other along time
        :param nt_wait: if nt_wait > 0, we will use simulation mode where several extra frame groups
        will be insert in the beginning
        :param transform: transform applied to each sample, need to keep shape [...,time, x, y]
        :param img_resize : tuple, (x, y) to resize img
        :return: each sample has shape [(echo,) 2, x, y, time], 'time' size is set to nt_network if specified
        """
        super(LITT, self).__init__()
        self.single_echo = single_echo
        self.acc = acc
        self.sample_n = sample_n
        self.transform = transform
        if mask_file_path is not None:
            assert len(mat_file_path) == len(mask_file_path)

        self.data = list()
        self.mask = list()
        for idx, file_path in enumerate(mat_file_path):
            mat_data = loadmat(file_path)
            mFFE_img_imag = mat_data['mFFE_img_imag']
            mFFE_img_real = mat_data['mFFE_img_real']
            
            if img_resize is not None:
                mFFE_img_imag = T.resize(mFFE_img_imag, (*img_resize, *mFFE_img_imag.shape[-2:]))
                mFFE_img_real = T.resize(mFFE_img_real, (*img_resize, *mFFE_img_real.shape[-2:]))

            mFFE_img_complex = mFFE_img_real + 1j * mFFE_img_imag  # [x, y, time, echo]
            mFFE_img_complex = np.transpose(
                mFFE_img_complex, (3, 2, 0, 1))  # -> [echo, time, x, y]

            if single_echo:
                mFFE_img_complex = mFFE_img_complex[1]  # if single echo, use the second one, -> [time, x, y]

            if mask_file_path is not None:
                mask = loadmat(mask_file_path[idx])['mask'].reshape(mFFE_img_complex.shape)
            else:
                mask = cs.cartesian_mask(mFFE_img_complex.shape, acc=self.acc, sample_n=self.sample_n)

            if nt_wait > 0:  # the preparation stage
                assert nt_wait < nt_network, f'nt_wait({nt_wait}) must be smaller than nt_network({nt_network})'
                for i in range(nt_wait, nt_network):
                    self.data.append(mFFE_img_complex[..., :i, :, :])
                    self.mask.append(mask[..., :i, :, :])

            if not overlap:  # slice the data along time dim according to nt_network with no overlapping
                total_t = mFFE_img_complex.shape[-3]
                complete_slice = total_t // nt_network
                for i in range(complete_slice):
                    self.data.append(mFFE_img_complex[..., i * nt_network:(i + 1) * nt_network, :, :])
                    self.mask.append(mask[..., i * nt_network:(i + 1) * nt_network, :, :])
                if total_t % nt_network > 0:
                    self.data.append(mFFE_img_complex[..., -nt_network:, :, :])
                    self.mask.append(mask[..., -nt_network:, :, :])
            else:  # ... with overlapping
                for i in range(mFFE_img_complex.shape[-3] - (nt_network - 1)):
                    self.data.append(mFFE_img_complex[..., i:i + nt_network, :, :])
                    self.mask.append(mask[..., i:i + nt_network, :, :])

    def __getitem__(self, idx):
        img_gnd = self.data[idx]  # [(echo, )time, x, y]
        mask = self.mask[idx]  # the same shape with img_gnd

        if self.transform is not None:
            img_gnd, mask = self.transform(img_gnd, mask)

        img_u, k_u = cs.undersample(img_gnd, mask)

        # complex64 -> float32, [(echo, )time, x, y] -> [(echo, )time, x, y, 2] -> [(echo,) 2, x, y, time]
        perm = (3, 1, 2, 0) if self.single_echo else (0, 4, 2, 3, 1)
        img_gnd_tensor = torch.view_as_real(torch.from_numpy(img_gnd)).float().permute(perm)
        img_u_tensor = torch.view_as_real(torch.from_numpy(img_u)).float().permute(perm)
        k_u_tensor = torch.view_as_real(torch.from_numpy(k_u)).float().permute(perm)
        mask_tensor = torch.view_as_real(torch.from_numpy(mask*(1+1j))).float().permute(perm)

        return {'img_gnd': img_gnd_tensor,
                'img_u': img_u_tensor,
                'k_u': k_u_tensor,
                'mask': mask_tensor}

    def __len__(self):
        return len(self.data)


def get_LITT_dataset(data_root, split, **kwargs):
    data_root = pathlib.Path(data_root)
    base_folder = (pathlib.Path(__file__).parent.absolute()/pathlib.Path(f'data/{split}/')).resolve()
    mat_file_path = sorted([data_root/(x.name + '.mat') for x in base_folder.iterdir()])
    dataset = LITT(mat_file_path, **kwargs)
    return dataset
