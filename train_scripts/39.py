import argparse
import os
import sys
sys.path.append('../')

import utils

import time
import subprocess

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import structural_similarity as cal_ssim, mean_squared_error as cal_mse

from data import LITT_v3, data_aug
from metrics import complex_psnr
from model.model_zy import RRN_two_stage
from utils import from_tensor_format


def step_train(dataloader, model, criterion, optimizer, writer, epoch, **kwargs):
    train_loss = 0
    train_batches = 0
    model.train()
    for batch in dataloader:
        img_u, k_u, mask, img_gnd = batch['img_u'], batch['k_u'], batch['mask'], batch['img_gnd']
        if torch.cuda.is_available():
            img_u, k_u, mask, img_gnd = img_u.cuda(), k_u.cuda(), mask.cuda(), img_gnd.cuda()

        img_u_l = None
        output_h = None
        output_o = None
        output_o_c = list()
        for i in range(img_u.shape[-1]):  # iterate over t dimension
            output_o, output_h = model(img_u[..., i], k_u[..., i], mask[..., i],
                                       img_u_l, output_h, output_o)
            img_u_l = img_u[..., i]
            output_o_c.append(output_o)
        pred = torch.stack(output_o_c, dim=-1)

        loss = criterion(pred, img_gnd)

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.item()
        train_batches += 1

    train_loss /= train_batches

    print(time.strftime('%H:%M:%S') + ' ' + f'train'
          + ' - ' + f'loss: {train_loss:.6f}')
    writer.add_scalar('Loss/train', train_loss, epoch)

    scheduler.step()


def step_test(dataloader, model, criterion, work_dir, writer, epoch, **kwargs):
    test_loss = 0
    base_psnr, test_psnr = 0, 0
    base_mag_ssim, test_mag_ssim = 0, 0
    base_phase_rmse, test_phase_rmse = 0, 0
    test_batches = 0
    cnt = 0
    model.eval()
    for batch in dataloader:
        with torch.no_grad():
            img_u, k_u, mask, img_gnd = batch['img_u'], batch['k_u'], batch['mask'], batch['img_gnd']
            if torch.cuda.is_available():
                img_u, k_u, mask, img_gnd = img_u.cuda(), k_u.cuda(), mask.cuda(), img_gnd.cuda()

            img_u_l = None
            output_h = None
            output_o = None
            output_o_c = list()
            for i in range(img_u.shape[-1]):  # iterate over t dimension
                output_o, output_h = model(img_u[..., i], k_u[..., i], mask[..., i],
                                           img_u_l, output_h, output_o)
                img_u_l = img_u[..., i]
                output_o_c.append(output_o)
            pred = torch.stack(output_o_c, dim=-1)

            loss = criterion(pred, img_gnd)

        test_loss += loss.item()
        test_batches += 1

        for img_i, img_u_i, pred_i in zip(from_tensor_format(img_gnd.cpu().numpy()),
                                          from_tensor_format(img_u.cpu().numpy()),
                                          from_tensor_format(pred.cpu().numpy())):  # -> [t, x, y]
            base_psnr += complex_psnr(img_i, img_u_i, peak='max')
            test_psnr += complex_psnr(img_i, pred_i, peak='max')
            base_mag_ssim += np.mean([cal_ssim(abs(img_i[i, ...]), abs(img_u_i[i, ...]))
                                      for i in range(img_i.shape[0])])
            test_mag_ssim += np.mean([cal_ssim(abs(img_i[i, ...]), abs(pred_i[i, ...]))
                                      for i in range(img_i.shape[0])])
            base_phase_rmse += np.mean([np.sqrt(cal_mse(np.angle(img_i[i, ...]), np.angle(img_u_i[i, ...])))
                                        for i in range(img_i.shape[0])])
            test_phase_rmse += np.mean([np.sqrt(cal_mse(np.angle(img_i[i, ...]), np.angle(pred_i[i, ...])))
                                        for i in range(img_i.shape[0])])
            cnt += 1

    test_loss /= test_batches
    base_psnr /= cnt
    test_psnr /= cnt
    base_mag_ssim /= cnt
    test_mag_ssim /= cnt
    base_phase_rmse /= cnt
    test_phase_rmse /= cnt

    # save metrics
    print(time.strftime('%H:%M:%S') + ' ' + f'test'
          + ' - ' + f'loss: {test_loss:.6f}'
          + ', ' + f'PSNR: {base_psnr:.4f}->{test_psnr:.4f}'
          + ', ' + f'Magnitude SSIM: {base_mag_ssim:.4f}->{test_mag_ssim:.4f}'
          + ', ' + f'Phase RMSE: {base_phase_rmse:.4f}->{test_phase_rmse:.4f}')

    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('PSNR/base', base_psnr, epoch)
    writer.add_scalar('PSNR/test', test_psnr, epoch)
    writer.add_scalar('Magnitude SSIM/base', base_mag_ssim, epoch)
    writer.add_scalar('Magnitude SSIM/test', test_mag_ssim, epoch)
    writer.add_scalar('Phase RMSE/base', base_phase_rmse, epoch)
    writer.add_scalar('Phase RMSE/test', test_phase_rmse, epoch)

    # save model
    torch.save(rec_net.state_dict(), os.path.join(work_dir, 'model.pth'))

    # save images
    _img_gnd = from_tensor_format(img_gnd.cpu().numpy())[0, -1]
    _img_u = from_tensor_format(img_u.cpu().numpy())[0, -1]
    _pred = from_tensor_format(pred.cpu().numpy())[0, -1]
    mag_diagram = np.concatenate([abs(_img_gnd), abs(_img_u), abs(_pred)], axis=1)
    phase_diagram = np.concatenate([np.angle(_img_gnd), np.angle(_img_u), np.angle(_pred)], axis=1)
    writer.add_image('Viz/Magnitude', mag_diagram, epoch, dataformats='HW')
    writer.add_image('Viz/Phase', phase_diagram, epoch, dataformats='HW')


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/root/LITT_data')
parser.add_argument('--mask_path', type=str, default='/root/LITT_mask_8x_c8')
parser.add_argument('--num_epoch', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=1 * 4)
parser.add_argument('--lr', type=float, default=0.0001 * 4)
parser.add_argument('--acc', type=float, default=8.0)
parser.add_argument('--sampled_lines', type=int, default=8)
parser.add_argument('--nt_network', type=int, default=10)
parser.add_argument('--test_interval', type=int, default=20)
parser.add_argument('--work_dir', type=str, default='/root/log/256_cs_8x_c8_cos_RRN_two_stage_b3')
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
train_transform = data_aug(block_size=(256, 32), rotation_xy=False, flip_t=False)

train_dataset = LITT_v3(img_dir=args.data_path,
                        split_dir='sub_ds_split/train',
                        nt_network=args.nt_network,
                        mask_func=utils.cs_cartesian_mask(acc=args.acc, sample_n=args.sampled_lines),
                        transform=train_transform)
test_dataset = LITT_v3(img_dir=args.data_path,
                       split_dir='sub_ds_split/test',
                       nt_network=args.nt_network,
                       mask_dir=args.mask_path)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

# model, loss, optimizer
rec_net = RRN_two_stage(n_blocks=3)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(rec_net.parameters(), lr=args.lr, betas=(0.5, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch)

# device
if torch.cuda.is_available():
    rec_net = rec_net.cuda()
    criterion.cuda()

# training, validation & test
generic_settings = {
    'model': rec_net,
    'criterion': criterion,
    'optimizer': optimizer,
    'work_dir': args.work_dir,
    'writer': writer,
    'scheduler': scheduler
}
for epoch in range(1, args.num_epoch + 1):
    print(f'Epoch {epoch}/{args.num_epoch}')
    step_train(epoch=epoch, dataloader=train_loader, **generic_settings)
    if epoch % args.test_interval == 0:
        step_test(epoch=epoch, dataloader=test_loader, **generic_settings)
