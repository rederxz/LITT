import argparse
import os
import sys

sys.path.append('../')
import time
import subprocess

import torch
import numpy as np
import scipy.io as sio
from torch.utils.tensorboard import SummaryWriter
import h5py

from data import LITT_from_np_array
from model.model_crnn import CRNN
from utils import from_tensor_format


def step_inference(dataloader, model):
    img_gnd_all, img_u_all, mask_all, img_rec_all, t_rec_all = list(), list(), list(), list(), list()

    model.eval()
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            img_u, k_u, mask, img_gnd = batch['img_u'], batch['k_u'], batch['mask'], batch['img_gnd']
            if torch.cuda.is_available():
                img_u, k_u, mask, img_gnd = img_u.cuda(), k_u.cuda(), mask.cuda(), img_gnd.cuda()

            tik = time.time()
            img_rec = model(img_u, k_u, mask)  # [batch_size, 2, width, height, n_seq]
            tok = time.time()

        # [batch_size, 2, width, height, n_seq] => [t, width, height] complex np array
        assert img_gnd.shape[0] == 1  # make sure batch_size == 1
        img_gnd_all.append(from_tensor_format(img_gnd.cpu().numpy())[0, -1:])
        img_u_all.append(from_tensor_format(img_u.cpu().numpy())[0, -1:])
        mask_all.append(from_tensor_format(mask.cpu().numpy())[0, -1:])
        img_rec_all.append(from_tensor_format(img_rec.cpu().numpy())[0, -1:])
        t_rec_all.append(tok - tik)

    # [t, x, y] complex images
    return {'im_grd': np.concatenate(img_gnd_all), 'im_und': np.concatenate(img_u_all),
            'mask': np.concatenate(mask_all), 'im_pred': np.concatenate(img_rec_all),
            'recon_time_all': np.array(t_rec_all), 'test_nt_wait': 0}


parser = argparse.ArgumentParser()
parser.add_argument('--mat_path', type=str, default='/root/autodl-nas/test_data/test_data_21/res_001_single.mat')
parser.add_argument('--model_path', type=str, default='/root/log/256_cs_8x_c8_cos_CRNN_baseline_v3/model.pth')
parser.add_argument('--work_dir', type=str, default='/root/inference_log/256_cs_6x_c8_cos_CRNN6_baseline_v3')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

if not args.debug:
    # create work directory
    os.makedirs(args.work_dir, exist_ok=False)
    # redirect output to file
    log_file = open(os.path.join(args.work_dir, 'log.log'), 'w')
    sys.stdout = log_file
    sys.stderr = log_file

writer = SummaryWriter(args.work_dir)

# print config
print('Commit ID:')
command = 'git rev-parse --short HEAD'
print(subprocess.getoutput(command))
print('Params:')
print(vars(args))

# data
with h5py.File(args.mat_path, 'r') as mat_file:
    # [t, coil, z, y, x] -> [x, y, z, coil, t]
    img = np.array(mat_file['combined_img']).transpose()
    img = img['real'] + img['imag'] * 1j
    mask = np.array(mat_file['combined_mask']).transpose()
    mask = mask * (1 + 1j)
print(f'img shape: {img.shape}')

# model & device
rec_net = CRNN()
if torch.cuda.is_available():
    rec_net = rec_net.cuda()
    rec_net.load_state_dict(torch.load(args.model_path, map_location=torch.device('cuda')))
else:
    rec_net.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))

# inference
im_grd_holder = np.zeros_like(img)
im_und_holder = np.zeros_like(img)
mask_holder = np.zeros_like(img)
im_pred_holder = np.zeros_like(img)
recon_time_all_holder = np.zeros_like(img)

_, _, z_total, coil_total, _ = img.shape
for z in z_total:
    for coil in coil_total:
        # build dataset and dataloader for current z and coil
        test_dataset = LITT_from_np_array(img[:, :, z, coil],
                                          mask[:, :, z, coil],
                                          nt_network=6,
                                          overlap=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

        # run
        result = step_inference(dataloader=test_loader, model=rec_net)  # result for current z and coil

        # collect results
        im_grd_holder[:, :, z, coil] = result['im_grd']  # [t, x, y]
        im_und_holder[:, :, z, coil] = result['im_und']
        mask_holder[:, :, z, coil] = result['mask']
        im_pred_holder[:, :, z, coil] = result['im_pred']
        recon_time_all_holder[:, :, z, coil] = result['recon_time_all']

sio.savemat(os.path.join(args.work_dir, 'test_result.mat'),
            {'im_grd': im_grd_holder,
             'im_und': im_und_holder,
             'mask': mask_holder,
             'im_pred': im_pred_holder,
             'recon_time_all': recon_time_all_holder,
             'test_nt_wait': 0})
