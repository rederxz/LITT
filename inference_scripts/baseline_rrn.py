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

from data import LITT_v3
from model.model_zy import RRN
from utils import from_tensor_format


def step_inference(dataloader, model, work_dir, **kwargs):
    img_gnd_all, img_u_all, mask_all, img_rec_all, t_rec_all = list(), list(), list(), list(), list()

    model.eval()
    img_u_l, output_h, output_o, output_o_c = None, None, None, list()
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            img_u, k_u, mask, img_gnd = batch['img_u'], batch['k_u'], batch['mask'], batch['img_gnd']
            if torch.cuda.is_available():
                img_u, k_u, mask, img_gnd = img_u.cuda(), k_u.cuda(), mask.cuda(), img_gnd.cuda()

            tik = time.time()
            output_o, output_h = model(img_u[..., 0], k_u[..., 0], mask[..., 0],
                                       img_u_l, output_h, output_o)
            tok = time.time()
            img_u_l = img_u[..., 0]
            img_rec = output_o[..., None]  # [batch_size, 2, width, height] => [batch_size, 2, width, height, 1]

        # [batch_size, 2, width, height, 1] => [t, width, height] complex np array
        assert img_gnd.shape[0] == 1  # make sure batch_size == 1
        img_gnd_all.append(from_tensor_format(img_gnd.cpu().numpy())[0])
        img_u_all.append(from_tensor_format(img_u.cpu().numpy())[0])
        mask_all.append(from_tensor_format(mask.cpu().numpy())[0])
        img_rec_all.append(from_tensor_format(img_rec.cpu().numpy())[0])
        t_rec_all.append(tok - tik)

    # save result, [t, x, y] complex images
    sio.savemat(os.path.join(work_dir, 'test_result.mat'),
                {'im_grd': np.concatenate(img_gnd_all), 'im_und': np.concatenate(img_u_all),
                 'mask': np.concatenate(mask_all), 'im_pred': np.concatenate(img_rec_all),
                 'recon_time_all': np.array(t_rec_all), 'test_nt_wait': 0})


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/root/LITT_data')
parser.add_argument('--mask_path', type=str, default='/root/LITT_mask_8x_c8')
parser.add_argument('--model_path', type=str, default='/root/log/256_cs_8x_c8_cos_RRN_baseline_v3/model.pth')
parser.add_argument('--work_dir', type=str, default='/root/inference_log/256_cs_8x_c8_cos_RRN_baseline_v3')
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

# data, each sample [batch_size, 2, x, y, time]
test_dataset = LITT_v3(img_dir=args.data_path,
                       split_dir='sub_ds_split/test',
                       nt_network=1,
                       mask_dir=args.mask_path)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

# model & device
rec_net = RRN()
if torch.cuda.is_available():
    rec_net = rec_net.cuda()
    rec_net.load_state_dict(torch.load(args.model_path, map_location=torch.device('cuda')))
else:
    rec_net.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))

# training, validation & test
generic_settings = {
    'model': rec_net,
    'work_dir': args.work_dir,
}

step_inference(dataloader=test_loader, **generic_settings)
