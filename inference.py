import argparse
import os
import time

import numpy as np
import torch
import scipy.io as sio
import matplotlib.pyplot as plt

from data import get_LITT_dataset
from model import CRNN
from utils import from_tensor_format


def step_test(dataloader, model, work_dir, fig_interval):
    img_gnd_all, img_u_all, mask_all, img_rec_all, t_rec_all = list(), list(), list(), list(), list()
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            img_u, k_u, mask, img_gnd = batch['img_u'], batch['k_u'], batch['mask'], batch['img_gnd']
            if torch.cuda.is_available():
                img_u, k_u, mask, img_gnd = img_u.cuda(), k_u.cuda(), mask.cuda(), img_gnd.cuda()
            tik = time.time()
            img_rec = model(img_u, k_u, mask)  # [batch_size, 2, width, height, n_seq]
            tok = time.time()

        # [batch_size, 2, width, height, n_seq] => [width, height] complex np array
        assert img_gnd.shape[0] == 1  # make sure batch_size == 1
        img_gnd_all.append(from_tensor_format(img_gnd[..., -1].cpu().numpy()).squeeze())
        img_u_all.append(from_tensor_format(img_u[..., -1].cpu().numpy()).squeeze())
        mask_all.append(from_tensor_format(mask[..., -1].cpu().numpy()).squeeze())
        img_rec_all.append(from_tensor_format(img_rec[..., -1].cpu().numpy()).squeeze())
        t_rec_all.append(tok - tik)

        if (i + 1) % fig_interval == 0:  # TODO 是否必要？
            # amp diff
            im1 = abs(img_gnd_all[i]) - abs(img_rec_all[i])
            im2 = abs(np.concatenate([img_u_all[i], img_rec_all[i], img_gnd_all[i]], 1))
            im = np.concatenate([im2, 2 * abs(im1)], 1)
            plt.imsave(os.path.join(work_dir, f'im{i + 1}_x.png'), im, cmap='gray')

            # complex phase_diff
            im1 = np.angle(img_gnd_all[i] * np.conj(img_rec_all[i]))
            im2 = np.angle(np.concatenate([img_u_all[i], img_rec_all[i], img_gnd_all[i]], 1))
            im = np.concatenate([im2, 2 * im1], 1)
            plt.imsave(os.path.join(work_dir, f'im{i + 1}_angle_x.png'), im, cmap='gray')

    # save result, [t, x, y] complex images
    sio.savemat(os.path.join(work_dir, 'test_result.mat'),
                {'im_grd': np.array(img_gnd_all), 'im_und': np.array(img_u_all), 'mask': np.array(mask_all),
                 'im_pred': np.array(img_rec_all), 'recon_time_all': np.array(t_rec_all)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./LITT_data/', help='the directory of data')
    parser.add_argument('--model_path', type=str, default='./crnn/model.pth', help='the path of model weights')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--acc', type=float, default=6.0, help='Acceleration factor for k-space sampling')
    parser.add_argument('--sampled_lines', type=int, default=8, help='Number of sampled lines at k-space center')
    parser.add_argument('--uni_direction', action='store_true', help='Bidirectional or unidirectional network')
    parser.add_argument('--mask_path', type=str, default=None, help='the path of the specified mask')
    parser.add_argument('--nt_network', type=int, default=6, help='Time frames involved in the network.')
    parser.add_argument('--fig_interval', type=int, default=10, help='Frame intervals to save figs.')
    parser.add_argument('--work_dir', type=str, default='crnn', help='work directory')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    args = parser.parse_args()

    # create work directory
    if not os.path.exists(args.work_dir):
        os.mkdir(args.work_dir)

    # data, each sample [n_samples(, echo), t, x, y]
    mask = sio.loadmat(args.mask_path) if args.mask_path is not None else None
    test_dataset = get_LITT_dataset(data_root=args.data_path, split='test', nt_network=args.nt_network,
                                    single_echo=True, acc=args.acc, sample_n=args.sampled_lines,
                                    mask=mask, overlap=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    # model
    rec_net = CRNN(uni_direction=args.uni_direction)
    rec_net.load_state_dict(torch.load(args.model_path))

    # device
    if torch.cuda.is_available():
        rec_net = rec_net.cuda()

    # test
    step_test(test_loader, rec_net, args.work_dir, args.fig_interval)
