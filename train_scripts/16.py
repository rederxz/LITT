import argparse
import os
import sys

import utils

sys.path.append('../')
import time
import subprocess

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import structural_similarity as cal_ssim, mean_squared_error as cal_mse

from data import get_LITT_dataset_v2, data_aug
from metrics import complex_psnr
from model.model_crnn import CRNN_DS
from utils import from_tensor_format


def step_train(dataloader, model, criterion, optimizer, writer, epoch, **kwargs):
    train_loss = 0
    train_batches = 0
    model.train()
    for batch in dataloader:
        img_u, k_u, mask, img_gnd = batch['img_u'], batch['k_u'], batch['mask'], batch['img_gnd']
        if torch.cuda.is_available():
            img_u, k_u, mask, img_gnd = img_u.cuda(), k_u.cuda(), mask.cuda(), img_gnd.cuda()

        rec = model(img_u, k_u, mask)
        loss = criterion(rec, img_gnd)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_batches += 1

    train_loss /= train_batches

    print(time.strftime('%H:%M:%S') + ' ' + f'train'
          + ' - ' + f'loss: {train_loss:.6f}')
    writer.add_scalar('Loss/train', train_loss, epoch)

    scheduler.step()


def step_val(dataloader, model, criterion, writer, epoch, **kwargs):
    val_loss = 0
    val_batches = 0
    model.eval()
    for batch in dataloader:
        with torch.no_grad():
            img_u, k_u, mask, img_gnd = batch['img_u'], batch['k_u'], batch['mask'], batch['img_gnd']
            if torch.cuda.is_available():
                img_u, k_u, mask, img_gnd = img_u.cuda(), k_u.cuda(), mask.cuda(), img_gnd.cuda()
            pred = model(img_u, k_u, mask)
            loss = criterion(pred, img_gnd)

        val_loss += loss.item()
        val_batches += 1

    val_loss /= val_batches

    print(time.strftime('%H:%M:%S') + ' ' + f'val'
          + ' - ' + f'loss: {val_loss:.6f}')
    writer.add_scalar('Loss/val', val_loss, epoch)


def step_test(dataloader, model, criterion, work_dir, writer, epoch, **kwargs):
    test_loss = 0
    base_psnr, test_psnr = 0, 0
    base_mag_ssim, test_mag_ssim = 0, 0
    base_phase_rmse, test_phase_rmse = 0, 0
    test_batches = 0
    model.eval()
    for batch in dataloader:
        with torch.no_grad():
            img_u, k_u, mask, img_gnd = batch['img_u'], batch['k_u'], batch['mask'], batch['img_gnd']
            if torch.cuda.is_available():
                img_u, k_u, mask, img_gnd = img_u.cuda(), k_u.cuda(), mask.cuda(), img_gnd.cuda()
            pred = model(img_u, k_u, mask)
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

    test_loss /= test_batches
    base_psnr /= test_batches * 1  # "1" for iteration within each mini-batch  (batch_size == 1)
    test_psnr /= test_batches * 1
    base_mag_ssim /= test_batches * 1
    test_mag_ssim /= test_batches * 1
    base_phase_rmse /= test_batches * 1
    test_phase_rmse /= test_batches * 1

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
    mag_diagram = np.concatenate([
        abs(from_tensor_format(img_gnd.cpu().numpy()).squeeze())[-1],
        abs(from_tensor_format(img_u.cpu().numpy()).squeeze())[-1],
        abs(from_tensor_format(pred.cpu().numpy()).squeeze())[-1],
    ], axis=1)
    phase_diagram = np.concatenate([
        np.angle(from_tensor_format(img_gnd.cpu().numpy()).squeeze())[-1],
        np.angle(from_tensor_format(img_u.cpu().numpy()).squeeze())[-1],
        np.angle(from_tensor_format(pred.cpu().numpy()).squeeze())[-1]
    ], axis=1)
    writer.add_image('Viz/Magnitude', mag_diagram, epoch, dataformats='HW')
    writer.add_image('Viz/Phase', phase_diagram, epoch, dataformats='HW')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/root/LITT_data', help='the directory of data')
    parser.add_argument('--num_epoch', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--acc', type=float, default=8.0, help='Acceleration factor for k-space sampling')
    parser.add_argument('--sampled_lines', type=int, default=8, help='Number of sampled lines at k-space center')
    parser.add_argument('--img_size', type=int, default=256, help='Input image size')
    parser.add_argument('--uni_direction', action='store_true', help='Bidirectional or unidirectional network')
    parser.add_argument('--nt_network', type=int, default=6, help='Time frames involved in the network.')
    parser.add_argument('--test_interval', type=int, default=20, help='Epoch intervals to test')
    parser.add_argument('--work_dir', type=str, default='/root/log/256_cs_8x_c8_cos_CRNN_ds', help='work directory')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    args = parser.parse_args()

    # create work directory
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    if not args.debug:
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

    # data, each sample [batch_size, (echo,) 2, x, y, time]
    train_transform = data_aug(block_size=(args.img_size, 32), rotation_xy=False, flip_t=False)

    train_dataset = get_LITT_dataset_v2(data_root=args.data_path, split='train',
                                        nt_network=args.nt_network,
                                        mask_func=utils.cs_cartesian_mask(acc=args.acc, sample_n=args.sampled_lines),
                                        img_resize=(args.img_size,) * 2,
                                        transform=train_transform)
    val_dataset = get_LITT_dataset_v2(data_root=args.data_path, split='val',
                                      nt_network=args.nt_network,
                                      mask_func=utils.cs_cartesian_mask(acc=args.acc, sample_n=args.sampled_lines),
                                      img_resize=(args.img_size,) * 2)
    test_dataset = get_LITT_dataset_v2(data_root=args.data_path, split='test',
                                       nt_network=args.nt_network,
                                       mask_func=utils.cs_cartesian_mask(acc=args.acc, sample_n=args.sampled_lines),
                                       img_resize=(args.img_size,) * 2)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    # model, loss, optimizer
    rec_net = CRNN_DS(sharing_size=5, uni_direction=args.uni_direction)
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
        step_val(epoch=epoch, dataloader=val_loader, **generic_settings)
        if epoch % args.test_interval == 0:
            step_test(epoch=epoch, dataloader=test_loader, **generic_settings)