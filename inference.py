import argparse
import os
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch

from data import get_LITT_dataset
from model import CRNN
from utils import from_tensor_format


def plot_diff(img_gnd, img_u, img_rec):
    # amp diff
    im1 = abs(img_gnd) - abs(img_rec)
    im2 = abs(np.concatenate([img_u, img_rec, img_gnd], 1))
    amp_diff = np.concatenate([im2, 2 * abs(im1)], 1)

    # complex phase_diff
    im1 = np.angle(img_gnd * np.conj(img_rec))
    im2 = np.angle(np.concatenate([img_u, img_rec, img_gnd], 1))
    phase_diff = np.concatenate([im2, 2 * im1], 1)

    return amp_diff, phase_diff


def step_inference(dataloader, model, work_dir, fig_interval, queue_mode, nt_wait):
    img_gnd_all, img_u_all, mask_all, img_rec_all, t_rec_all = list(), list(), list(), list(), list()

    model.eval()
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            img_u, k_u, mask, img_gnd = batch['img_u'], batch['k_u'], batch['mask'], batch['img_gnd']
            if torch.cuda.is_available():
                img_u, k_u, mask, img_gnd = img_u.cuda(), k_u.cuda(), mask.cuda(), img_gnd.cuda()

            wanted_start, time_record = (-1, 1) if img_gnd.shape[-3] != nt_wait else (0, nt_wait)

            tik = time.time()
            if queue_mode:
                img_rec, hidden = model.queue_forward(img_u, k_u, mask) if i == 0 \
                    else model.queue_forward(img_u, k_u, mask, hidden)  # [batch_size, 2, width, height, 1]
            else:
                img_rec = model(img_u, k_u, mask)  # [batch_size, 2, width, height, n_seq]
            tok = time.time()

        # [batch_size, 2, width, height, n_seq] => [t, width, height] complex np array
        assert img_gnd.shape[0] == 1  # make sure batch_size == 1
        img_gnd_all.append(from_tensor_format(img_gnd[..., wanted_start:].cpu().numpy()).squeeze(axis=0))
        img_u_all.append(from_tensor_format(img_u[..., wanted_start:].cpu().numpy()).squeeze(axis=0))
        mask_all.append(from_tensor_format(mask[..., wanted_start:].cpu().numpy()).squeeze(axis=0))
        img_rec_all.append(from_tensor_format(img_rec[..., wanted_start:].cpu().numpy()).squeeze(axis=0))
        t_rec_all.extend([tok - tik] * time_record)

        if (i + 1) % fig_interval == 0:
            amp_diff, phase_diff = plot_diff(img_gnd_all[-1][-1], img_u_all[-1][-1], img_rec_all[-1][-1])
            plt.imsave(os.path.join(work_dir, f'im{i + 1}_x.png'), amp_diff, cmap='gray')
            plt.imsave(os.path.join(work_dir, f'im{i + 1}_angle_x.png'), phase_diff, cmap='gray')

    # save result, [t, x, y] complex images
    sio.savemat(os.path.join(work_dir, 'test_result.mat'),
                {'im_grd': np.concatenate(img_gnd_all), 'im_und': np.concatenate(img_u_all),
                 'mask': np.concatenate(mask_all), 'im_pred': np.concatenate(img_rec_all),
                 'recon_time_all': np.array(t_rec_all), 'test_nt_wait': nt_wait})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # -- data --
    parser.add_argument('--data_path', type=str, default='./LITT_data/', help='the directory of data')
    parser.add_argument('--acc', type=float, default=6.0, help='Acceleration factor for k-space sampling')
    parser.add_argument('--sampled_lines', type=int, default=8, help='Number of sampled lines at k-space center')
    parser.add_argument('--mask_path', type=str, nargs='+', help='the path of the specified mask')
    # -- model --
    parser.add_argument('--model_path', type=str, default='./crnn/model.pth', help='the path of model weights')
    parser.add_argument('--uni_direction', action='store_true', help='Bidirectional or unidirectional network')
    parser.add_argument('--multi_hidden_t', type=int, default=1, help='Number of hidden_t involved in the model')
    # -- inference setting --
    parser.add_argument('--nt_network', type=int, default=1, help='Time frames involved in the network.')
    parser.add_argument('--queue_mode', action='store_true',
                        help='if inference one frame by one frame when testing unidirectional model')
    parser.add_argument('--nt_wait', type=int, default=0)
    # -- others --
    parser.add_argument('--work_dir', type=str, default='crnn', help='work directory')
    parser.add_argument('--fig_interval', type=int, default=10, help='Frame intervals to save figs.')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    args = parser.parse_args()

    # create work directory
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    # redirect output to file
    log_file = open(os.path.join(args.work_dir, 'log.log'), 'w')
    sys.stdout = log_file
    sys.stderr = log_file

    # print config
    print('Commit ID:')
    command = 'cd LITT && git rev-parse --short HEAD  && cd ..' if os.path.split(os.getcwd())[-1] != 'LITT' \
        else 'git rev-parse --short HEAD'
    print(subprocess.getoutput(command))
    print('Params:')
    print(vars(args))

    # data, each sample [n_samples(, echo), t, x, y]
    test_dataset = get_LITT_dataset(data_root=args.data_path, split='test', nt_network=args.nt_network,
                                    single_echo=True, acc=args.acc, sample_n=args.sampled_lines,
                                    mask_file_path=args.mask_path, overlap=True, nt_wait=args.nt_wait)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    # model
    rec_net = CRNN(uni_direction=args.uni_direction, multi_hidden_t=args.multi_hidden_t)
    rec_net.load_state_dict(torch.load(args.model_path))

    # device
    if torch.cuda.is_available():
        rec_net = rec_net.cuda()

    # test
    step_inference(test_loader, rec_net, args.work_dir, args.fig_interval, queue_mode=args.queue_mode,
                   nt_wait=args.nt_wait)
