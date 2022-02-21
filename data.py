import os

import torch
import numpy as np
from scipy.io import loadmat

import compressed_sensing as cs
from utils import to_tensor_format


def cut_data(input, block_size, cut_edge=None):
    """
    cut the images by specified block_size
    :param input: with shape [n_samples, echo, x, y, nt]
    :param block_size: [256, 32]
    :param cut_edge: [0, 32]
    :return:
    """
    if cut_edge is not None:
        if cut_edge[0] != 0:
            input = input[..., cut_edge[0]:-cut_edge[0], :, :]
        if cut_edge[1] != 0:
            input = input[..., :, cut_edge[1]:-cut_edge[1], :]
    ip_shape = input.shape  # [n_samples, echo, 256, 256 - 32 * 2 == 192, nt]

    dim = len(block_size)
    ncut = [None] * dim
    ncut_half = [None] * dim
    ncut_half_2 = [None] * dim

    for ndim in range(dim):  # 0
        ncut[ndim] = ip_shape[ndim + 2] % block_size[ndim]  # 64
        ncut_half[ndim] = round(ncut[ndim] / 2)  # 32
        ncut_half_2[ndim] = ncut[ndim] - ncut_half[ndim]  # 32
        if ncut[ndim] == 0:  # 正好是整数倍
            ncut_half[ndim] = 0
            ncut_half_2[ndim] = -ip_shape[ndim + 2]

    input_p = input[:, :, ncut_half[0]:-ncut_half_2[0], ncut_half[1]:-ncut_half_2[1], :]
    return input_p


def data_aug(img, block_size=None, rotation_xy=False, flip_t=False):
    """
    apply data augmentation to a batch
    :param img: [n, x, y, nt] or [n, echo, x, y, nt]
    :param block_size: [256, 32]
    :param rotation_xy: boolean
    :param flip_t: boolean
    :return:
    """
    if rotation_xy and np.rand.random() > 0.5:
        img = np.rot90(img, k=np.random.randint(0, 4), axes=(-3, -2))

    if flip_t and np.rand.random() > 0.5:
        img = img[..., ::-1]

    if block_size is not None:
        pass

    if block_size is not None:
        # img_pad = pad_data(img,block_size)
        img_pad = cut_data(img, block_size, cut_edge=(0, 32))
        img_shape = img_pad.shape  # [n_samples, echo, x, y, nt]
        n_x = int(img_shape[2] / block_size[0])
        n_y = int(img_shape[3] / block_size[1])
        img_block = []
        for i in range(n_x):
            for j in range(n_y):
                img_block.append(img_pad[:, :, i * block_size[0]:(i + 1) * block_size[0],
                                 j * block_size[1]:(j + 1) * block_size[1], :])
        img_block = np.array(img_block)
        sp = img_block.shape
        img = np.reshape(img_block, newshape=(sp[0] * sp[1], sp[2], sp[3], sp[4], sp[5]))

    print(img.shape)

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
        :return: if single_echo, each sample has shape [time, x, y]; else [echos, time, x, y]
        'time' size is set to nt_network if specified
        """
        super(LITT, self).__init__()
        self.acc = acc
        self.sample_n = sample_n
        self.transform = transform

        self.data = list()
        for file_path in mat_file_path:
            mat_data = loadmat(file_path)
            mFFE_img_imag = mat_data['mFFE_img_imag']
            mFFE_img_real = mat_data['mFFE_img_real']
            mFFE_img_complex = mFFE_img_real + 1j * mFFE_img_imag  # [x, y, time, echo]
            mFFE_img_complex = mFFE_img_complex.transpose((3, 2, 0, 1))  # [echo, time, x, y]

            if single_echo:  # [time, x, y]
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
        img_gnd = self.data[idx]

        if self.transform:
            img_gnd = self.transform(img_gnd)

        mask = cs.cartesian_mask(img_gnd.shape, acc=self.acc, sample_n=self.sample_n)
        img_u, k_u = cs.undersample(img_gnd, mask)

        # convert to float32 tensor (original format is float64)
        img_gnd_tensor = torch.from_numpy(to_tensor_format(img_gnd)).float()
        img_u_tensor = torch.from_numpy(to_tensor_format(img_u)).float()
        k_u_tensor = torch.from_numpy(to_tensor_format(k_u)).float()
        mask_tensor = torch.from_numpy(to_tensor_format(mask, mask=True)).float()

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

