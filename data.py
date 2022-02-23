import os

import numpy as np
import torch
from scipy.io import loadmat
from toolz import curry

import compressed_sensing as cs


def cut_data(input, block_size=(1, 1), cut_edge=None):
    """
    make sure input is divisible by the block size and cut edge area
    :param input: with shape [n_samples, echo, x, y, nt]
    :param block_size: [int, int], such as [256, 32]
    :param cut_edge: [0, 32]
    :return:
    """
    if cut_edge is not None:
        if cut_edge[0] != 0:
            input = input[..., cut_edge[0]:-cut_edge[0], :, :]
        if cut_edge[1] != 0:
            input = input[..., :, cut_edge[1]:-cut_edge[1], :]

    # make sure x dim divisible
    start_x, end_x = 0, None
    assert input.shape[-3] >= block_size[0]
    remainder = input.shape[-3] % block_size[0]
    if remainder > 0:
        start_x += remainder // 2
        end_x = - (remainder - remainder // 2)

    # make sure y dim divisible
    start_y, end_y = 0, None
    assert input.shape[-2] >= block_size[1]
    remainder = input.shape[-2] % block_size[1]
    if remainder > 0:
        start_y += remainder // 2
        end_y = - (remainder - remainder // 2)

    input_p = input[..., start_x:end_x, start_y:end_y, :]

    return input_p


@ curry
def data_aug(img, block_size=None, rotation_xy=False, flip_t=False):
    """
    apply data augmentation to a batch
    :param img: [n, x, y, nt] or [n, echo, x, y, nt]
    :param block_size: [int, int]
    :param rotation_xy: boolean
    :param flip_t: boolean
    :return:
    """
    if rotation_xy and np.rand.random() > 0.5:
        img = np.rot90(img, k=np.random.randint(0, 4), axes=(-3, -2))

    if flip_t and np.rand.random() > 0.5:
        img = img[..., ::-1]

    if block_size is not None:
        assert img.shape[-3] >= block_size[0] and img.shape[-2] >= block_size[1]

        # cut unimportant edge area
        img = cut_data(img, cut_edge=(0, 32))

        start_x = np.random.randint(0, img.shape[-3] - block_size[0] + 1)
        start_y = np.random.randint(0, img.shape[-2] - block_size[1] + 1)
        img = img[..., start_x: start_x + block_size[0], start_y: start_y + block_size[1], :]

    return img


def load_data(type, nt_network, load_one_data_ind=None):
    # jieying
    # load_one_data_ind is used to determine one data file for loading (only one)
    # , used in test process only
    # load_one_data_ind should be an int when nt_network=None
    # collect files
    data_path = '/home/jieying/dynamic_imging/LITT_data/Tiantan/'
    data_list = []
    for file in os.listdir(data_path):
        if os.path.splitext(file)[1] == '.mat':
            data_list.append(os.path.join(data_path, file))
    # create_fold_data
    nsp = len(data_list)
    ny_red = 10  # 10 fold
    sl = nsp // ny_red
    # train 8; val 1; test 1;
    if type == 'train':
        data_list = data_list[sl * 2:nsp]
    elif type == 'validation':
        data_list = data_list[sl * 1:sl * 2]
    elif type == 'test':
        data_list = data_list[sl * 0:sl * 1]  # data_list[sl * 0:sl * 1]
        if nt_network is None and load_one_data_ind is not None:
            data_list = data_list[load_one_data_ind]

    # Load Data
    img_all = []
    n_sample_list = []

    if nt_network is None:
        print(data_list)
        for i in range(len(data_list)):
            data = loadmat(data_list[i])
            mFFE_img_imag = data['mFFE_img_imag']
            mFFE_img_real = data['mFFE_img_real']
            mFFE_img_complex = mFFE_img_real + 1j * mFFE_img_imag  # [x,y,time,echo]
            mFFE_img_complex = mFFE_img_complex.transpose((3, 0, 1, 2))  # [echo,x,y,time]

            # normalization  TODO: no normalization?
            mFFE_img_complex_norm = mFFE_img_complex
            n_div = 1
            img_all.append(mFFE_img_complex_norm)
            n_sample_list.append(n_div)
        img_all = np.array(img_all)  # [n_div_sum,slice,echo,x,y,nt_network]

    else:
        for i in range(len(data_list)):
            folder = data_list[i]
            print(folder)
            data = loadmat(folder)
            mFFE_img_imag = data['mFFE_img_imag']
            mFFE_img_real = data['mFFE_img_real']
            mFFE_img_complex = mFFE_img_real + 1j * mFFE_img_imag  # [x,y,time,echo]
            mFFE_img_complex = mFFE_img_complex.transpose((3, 0, 1, 2))  # [echo,x,y,time]
            img_shape = mFFE_img_complex.shape

            # normalization
            # example_data is 0-1, will it cause a problem ??
            mFFE_img_complex_norm = mFFE_img_complex

            # divide into multiple samples
            nt = img_shape[3]
            n_div = nt // nt_network
            n_mod = nt % nt_network
            for n_sp in range(n_div):
                mFFE_img_complex_div = mFFE_img_complex_norm[:, :, :, n_sp * nt_network:(n_sp + 1) * nt_network]
                img_all.append(mFFE_img_complex_div)
            if n_mod != 0:
                n_div = n_div + 1
                mFFE_img_complex_div = mFFE_img_complex_norm[:, :, :, -nt_network:]
                img_all.append(mFFE_img_complex_div)
            n_sample_list.append(n_div)
        img_all = np.array(img_all)  # [n_div_sum,slice,echo,x,y,nt_network]

    return img_all, n_sample_list, data_list


def load_data_v2(data_path, split, nt_network=None, single_echo=True):
    """
    load data as np tensor from mat file
    :param data_path: the folder of mat files
    :param split: 'train', 'validation' or 'test'
    :param nt_network: the number of time frames required
    :param single_echo: if only use one echo
    :return: if single_echo, [n_samples, x, y, frames]; else [n_samples, echos, x, y, frames]
    frames is set to nt_network if specified
    """
    mat_file_path = os.listdir(data_path)
    mat_file_path = mat_file_path[:8] if split == 'train' \
        else (mat_file_path[8:9] if split == 'validation' else mat_file_path[9:10])

    data = list()
    for file in mat_file_path:
        mat_data = loadmat(os.path.join(data_path, file))
        mFFE_img_imag = mat_data['mFFE_img_imag']
        mFFE_img_real = mat_data['mFFE_img_real']
        mFFE_img_complex = mFFE_img_real + 1j * mFFE_img_imag  # [x,y,time,echo]
        mFFE_img_complex = mFFE_img_complex.transpose((3, 0, 1, 2))  # [echo,x,y,time]

        if single_echo:
            mFFE_img_complex = mFFE_img_complex[0]  # TODO: if single echo, use the 1st one

        if nt_network is None:
            data.append(mFFE_img_complex)
        else:  # slice the data along time dim according to nt_network
            total_t = mFFE_img_complex.shape[-1]
            complete_slice = total_t // nt_network
            for i in range(complete_slice):
                data.append(mFFE_img_complex[..., i * nt_network:(i + 1) * nt_network])
            if total_t % nt_network > 0:
                data.append(mFFE_img_complex[..., -nt_network:])

    data = np.array(data)

    return data


class LITT(torch.utils.data.dataset.Dataset):
    def __init__(self, mat_file_path, nt_network=None, single_echo=True, acc=6.0, sample_n=8, transform=None):
        """
        build a LITT dataset
        :param mat_file_path: a list, the paths of mat files
        :param nt_network: the number of time frames required
        :param single_echo: if only use one echo
        :param acc: accelerating rate
        :param sample_n: preserve how many center lines (sample_n // 2 each side)
        :param transform: transform applied to each sample
        :return: each sample has shape [(echo,) 2, x, y, time], 'time' size is set to nt_network if specified
        """
        super(LITT, self).__init__()
        self.single_echo = single_echo
        self.acc = acc
        self.sample_n = sample_n
        self.transform = transform

        self.data = list()
        for file_path in mat_file_path:
            mat_data = loadmat(file_path)
            mFFE_img_imag = mat_data['mFFE_img_imag']
            mFFE_img_real = mat_data['mFFE_img_real']
            mFFE_img_complex = mFFE_img_real + 1j * mFFE_img_imag  # [x, y, time, echo]
            mFFE_img_complex = np.transpose(
                mFFE_img_complex, (3, 2, 0, 1))  # -> [echo, time, x, y] to adapt to following precessing in __getitem__

            if single_echo:
                mFFE_img_complex = mFFE_img_complex[0]  # TODO: if single echo, use the 1st one

            if nt_network is None:
                self.data.append(mFFE_img_complex)
            else:  # slice the data along time dim according to nt_network
                total_t = mFFE_img_complex.shape[-3]
                complete_slice = total_t // nt_network
                for i in range(complete_slice):
                    self.data.append(mFFE_img_complex[..., i * nt_network:(i + 1) * nt_network, :, :])
                if total_t % nt_network > 0:
                    self.data.append(mFFE_img_complex[..., -nt_network:, :, :])

    def __getitem__(self, idx):
        img_gnd = self.data[idx]  # [(echo, )time, x, y]

        if self.transform is not None:
            img_gnd = self.transform(img_gnd)

        # note: mask generation and under-sampling require img with shape [..., x, y]
        mask = cs.cartesian_mask(img_gnd.shape, acc=self.acc, sample_n=self.sample_n)
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
    mat_file_path = os.listdir(data_root)
    mat_file_path = mat_file_path[:8] if split == 'train' \
        else (mat_file_path[8:9] if split == 'val' else mat_file_path[9:10])
    mat_file_path = [os.path.join(data_root, path) for path in mat_file_path]
    dataset = LITT(mat_file_path, **kwargs)
    return dataset

