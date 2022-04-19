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

from data import get_LITT_dataset, data_aug
from metric import complex_psnr
from model_vsr import CRNN
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
          + ' - ' + f'loss: {train_loss}')
    writer.add_scalar('Loss/train', train_loss, epoch)


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
          + ' - ' + f'loss: {val_loss}')
    writer.add_scalar('Loss/val', val_loss, epoch)


def step_test(dataloader, model, criterion, work_dir, writer, epoch, **kwargs):
    test_loss = 0
    base_psnr = 0
    test_psnr = 0
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
                                          from_tensor_format(pred.cpu().numpy())):
            base_psnr += complex_psnr(img_i, img_u_i, peak='max')
            test_psnr += complex_psnr(img_i, pred_i, peak='max')

    test_loss /= test_batches
    base_psnr /= test_batches * 1  # "1" for iteration within each mini-batch
    test_psnr /= test_batches * 1
    print(time.strftime('%H:%M:%S') + ' ' + f'test'
          + ' - ' + f'loss: {test_loss}'
          + ', ' + f'base PSNR: {base_psnr}'
          + ', ' + f'test PSNR: {test_psnr}')

    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('PSNR/base', base_psnr, epoch)
    writer.add_scalar('PSNR/test', test_psnr, epoch)

    # save model
    torch.save(rec_net.state_dict(), os.path.join(work_dir, 'model.pth'))

    # save image, [t, x, y] complex images
    # sio.savemat(os.path.join(work_dir, 'figure.mat'), {'img_gnd': from_tensor_format(img_gnd.cpu().numpy()).squeeze(),
    #                                                    'img_u': from_tensor_format(img_u.cpu().numpy()).squeeze(),
    #                                                    'img_rec': from_tensor_format(pred.cpu().numpy()).squeeze()})

    diagram = np.concatenate([
        abs(from_tensor_format(img_gnd.cpu().numpy()).squeeze())[-1],
        abs(from_tensor_format(img_u.cpu().numpy()).squeeze())[-1],
        abs(from_tensor_format(pred.cpu().numpy()).squeeze())[-1]
    ], axis=1)
    writer.add_image('Viz', diagram, epoch, dataformats='HW')
    # writer.add_image('Viz/img_gnd', abs(from_tensor_format(img_gnd.cpu().numpy()).squeeze())[-1], epoch, dataformats='HW')
    # writer.add_image('Viz/img_u', abs(from_tensor_format(img_u.cpu().numpy()).squeeze())[-1], epoch, dataformats='HW')
    # writer.add_image('Viz/img_rec', abs(from_tensor_format(pred.cpu().numpy()).squeeze())[-1], epoch, dataformats='HW')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/root/LITT_data', help='the directory of data')
    parser.add_argument('--num_epoch', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--acc', type=float, default=6.0, help='Acceleration factor for k-space sampling')
    parser.add_argument('--sampled_lines', type=int, default=8, help='Number of sampled lines at k-space center')
    parser.add_argument('--img_size', type=int, default=256, help='Input image size')
    parser.add_argument('--uni_direction', action='store_true', help='Bidirectional or unidirectional network')
    parser.add_argument('--nt_network', type=int, default=6, help='Time frames involved in the network.')
    parser.add_argument('--test_interval', type=int, default=20, help='Epoch intervals to test')
    parser.add_argument('--work_dir', type=str, default='/root/log/256x256_cs_6x_c8_exp_1', help='work directory')
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

    train_dataset = get_LITT_dataset(data_root=args.data_path, split='train', nt_network=args.nt_network,
                                     single_echo=True, acc=args.acc, sample_n=args.sampled_lines,
                                     img_resize=(args.img_size,) * 2, transform=train_transform)
    val_dataset = get_LITT_dataset(data_root=args.data_path, split='val', nt_network=args.nt_network,
                                   single_echo=True, acc=args.acc, sample_n=args.sampled_lines,
                                   img_resize=(args.img_size,) * 2)
    test_dataset = get_LITT_dataset(data_root=args.data_path, split='test', nt_network=args.nt_network,
                                    single_echo=True, acc=args.acc, sample_n=args.sampled_lines,
                                    img_resize=(args.img_size,) * 2)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    # model, loss, optimizer
    rec_net = CRNN(uni_direction=args.uni_direction)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(rec_net.parameters(), lr=args.lr, betas=(0.5, 0.999))

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
        'writer': writer
    }
    for epoch in range(1, args.num_epoch + 1):
        print(f'Epoch {epoch}/{args.num_epoch}')
        step_train(epoch=epoch, dataloader=train_loader, **generic_settings)
        step_val(epoch=epoch, dataloader=val_loader, **generic_settings)
        if (epoch + 1) % args.test_interval == 0:
            step_test(epoch=epoch, dataloader=test_loader, **generic_settings)
