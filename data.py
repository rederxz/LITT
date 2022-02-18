import numpy as np
from scipy.io import loadmat
import os
'''
supporting functions for LITT DL recon, which requires numpy only (dont require torch or tf)
jieying 20211018
'''

def load_data(type,nt_network,load_one_data_ind=None):
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
    # train 8; validate 1; test 1;
    if type=='train':
        data_list=data_list[sl * 2:nsp]
    elif type=='validation':
        data_list=data_list[sl * 1:sl * 2]
    elif type=='test':
        data_list=data_list[sl * 0:sl * 1] #data_list[sl * 0:sl * 1]
        if nt_network is None and load_one_data_ind is not None:
            data_list=data_list[load_one_data_ind]

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


def pad_data(input,block_size):
    ip_shape = input.shape  # [data,x,y,z,ch]
    # print(ip_shape)
    dim = len(block_size)
    npad = [None]*dim
    npad_half = [None]*dim
    for ndim in range(dim):
        npad[ndim] = block_size[ndim] - (ip_shape[ndim+1] % block_size[ndim])
        if npad[ndim]==block_size[ndim]:
            npad[ndim]=0
        npad_half[ndim] = round(npad[ndim] / 2)

    input_p = np.pad(input, ((0, 0),(npad_half[0], npad[0] - npad_half[0]), (npad_half[1], npad[1] - npad_half[1]),
                             (npad_half[2], npad[2] - npad_half[2]), (0, 0)), mode='constant')
    return input_p


def cut_data(input,block_size,cut_edge=None):
    # input: # [n_samples, echo,x,y,nt]
    if cut_edge is not None:
        if cut_edge[0] != 0:
            input = input[:,:,cut_edge[0]:-cut_edge[0],:,:]
        if cut_edge[1] != 0:
            input = input[:,:,:,cut_edge[1]:-cut_edge[1],:]
    ip_shape = input.shape
    # print(ip_shape)
    dim = len(block_size)
    ncut = [None]*dim
    ncut_half = [None]*dim
    ncut_half_2 = [None]*dim
    for ndim in range(dim):
        ncut[ndim] = ip_shape[ndim+2] % block_size[ndim]
        ncut_half[ndim] = round(ncut[ndim] / 2)
        ncut_half_2[ndim] = ncut[ndim]-ncut_half[ndim]
        if ncut[ndim]==0:
            ncut_half[ndim]=0
            ncut_half_2[ndim]=-ip_shape[ndim+2]

    input_p = input[:,:,ncut_half[0]:-ncut_half_2[0],ncut_half[1]:-ncut_half_2[1], :]
    return input_p

def data_aug(img,block_size,rotation_xy=False,flip_t=False):
    # img: # [n_samples, echo,x,y,nt]
    if rotation_xy == True:
        img_rot = []
        img_rot.append(img)
        for rotation_angle in range(1,4):
            img_rot.append(np.rot90(img, k=rotation_angle, axes=(2,3)))
        img_rot=np.array(img_rot)
        sp = img_rot.shape
        img_rot = np.reshape(img_rot,newshape=(sp[0]*sp[1],sp[2],sp[3],sp[4],sp[5]))
    else:
        img_rot = img
    print(img_rot.shape)

    if block_size is not None:
        # img_pad = pad_data(img,block_size)
        img_pad = cut_data(img_rot, block_size,cut_edge=(0,32))
        img_shape = img_pad.shape #[n_samples, echo,x,y,nt]
        n_x = int(img_shape[2]/block_size[0])
        n_y = int(img_shape[3] / block_size[1])
        # img_block = np.empty(shape=(img_shape[0]*n_x*n_y,block_size[0],block_size[1],block_size[2],img_shape[4]))
        img_block = []
        for i in range(n_x):
            for j in range(n_y):
                img_block.append(img_pad[:,:,i*block_size[0]:(i+1)*block_size[0],
                                            j*block_size[1]:(j+1)*block_size[1],:])
        img_block = np.array(img_block)
        sp = img_block.shape
        img_block = np.reshape(img_block,newshape=(sp[0]*sp[1],sp[2],sp[3],sp[4],sp[5]))
    else:
        img_block = img_rot
    print(img_block.shape)

    if flip_t == True:
        img_flip = []
        img_flip.append(img_rot)
        img_flip.append(img_rot[:, :, :, :, ::-1])
        img_flip = np.array(img_flip)
        sp = img_flip.shape
        img_flip = np.reshape(img_flip,newshape=(sp[0]*sp[1],sp[2],sp[3],sp[4],sp[5]))
    else:
        img_flip=img_block
    print(img_flip.shape)

    return img_flip