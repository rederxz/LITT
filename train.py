import argparse
import os
import time

import torch
import numpy as np

from data import load_data_v2, data_aug, get_LITT_dataset
from model import CRNN_MRI_UniDir
import compressed_sensing as cs
from metric import complex_psnr
from utils import from_tensor_format, to_tensor_format


def prep_input(im, **kwargs):
    """
    Under-sample the batch, then reformat them into what the network accepts.
    :param im: tensor of shape [n_samples, t, x, y]
    :param acc: accelerate rate
    :param sample_n:
    :return: i_und, k_und, mask, i_gt [n_samples, n_channels, x, y, t]
    """
    mask = cs.cartesian_mask(im.shape, **kwargs)
    im_und, k_und = cs.undersample(im, mask)

    # convert to float32 tensor (original format is float64)
    im_gnd_l = torch.from_numpy(to_tensor_format(im)).float()
    im_und_l = torch.from_numpy(to_tensor_format(im_und)).float()
    k_und_l = torch.from_numpy(to_tensor_format(k_und)).float()
    mask_l = torch.from_numpy(to_tensor_format(mask, mask=True)).float()

    return im_und_l, k_und_l, mask_l, im_gnd_l


def step_train(dataloader, model, criterion, optimizer):
    train_loss = 0
    train_batches = 0
    model.train()
    for im in dataloader:
        # TODO: 在这里加入数据增强 (传入参数data_aug)?
        im_u, k_u, mask, gnd = prep_input(im, acc=args.acc, sample_n=args.sampled_lines)
        if torch.cuda.is_available():
            im_u, k_u, mask, gnd = im_u.cuda(), k_u.cuda(), mask.cuda(), gnd.cuda()

        rec = model(im_u, k_u, mask)
        loss = criterion(rec, gnd)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_batches += 1

    train_loss /= train_batches

    print(time.strftime('%H:%M:%S') + ' ' + f'train'
          + ' - ' + f'loss: {train_loss}')


def step_validate(dataloader, model, criterion):
    validate_loss = 0
    validate_batches = 0
    model.eval()
    for im in dataloader:
        with torch.no_grad():
            im_u, k_u, mask, gnd = prep_input(im, acc=args.acc, sample_n=args.sampled_lines)
            if torch.cuda.is_available():
                im_u, k_u, mask, gnd = im_u.cuda(), k_u.cuda(), mask.cuda(), gnd.cuda()
            pred = model(im_u, k_u, mask)
            loss = criterion(pred, gnd)

        validate_loss += loss.item()
        validate_batches += 1

    validate_loss /= validate_batches

    print(time.strftime('%H:%M:%S') + ' ' + f'valid'
          + ' - ' + f'loss: {validate_loss}')


def step_test(dataloader, model, criterion):
    test_loss = 0
    base_psnr = 0
    test_psnr = 0
    test_batches = 0
    for im in dataloader:
        with torch.no_grad():
            im_u, k_u, mask, gnd = prep_input(im, acc=args.acc, sample_n=args.sampled_lines)
            if torch.cuda.is_available():
                im_u, k_u, mask, gnd = im_u.cuda(), k_u.cuda(), mask.cuda(), gnd.cuda()
            pred = model(im_u, k_u, mask)
            loss = criterion(pred, gnd)

        test_loss += loss.item()
        test_batches += 1

        for im_i, und_i, pred_i in zip(im,
                                       from_tensor_format(im_u.cpu().numpy()),
                                       from_tensor_format(pred.cpu().numpy())):
            base_psnr += complex_psnr(im_i, und_i, peak='max')
            test_psnr += complex_psnr(im_i, pred_i, peak='max')

    test_loss /= test_batches
    base_psnr /= test_batches * 1  # "1" for iteration within each mini-batch
    test_psnr /= test_batches * 1
    print(time.strftime('%H:%M:%S') + ' ' + f'test'
          + ' - ' + f'loss: {test_loss}'
          + ', ' + f'base PSNR: {base_psnr}'
          + ', ' + f'test PSNR: {test_psnr}')

    # save model
    torch.save(rec_net.state_dict(), os.path.join(args.work_dir, f'model.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./LITT_data/', help='the directory of data')
    parser.add_argument('--num_epoch', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--acc', type=float, default=6.0, help='Acceleration factor for k-space sampling')
    parser.add_argument('--sampled_lines', type=int, default=15, help='Number of sampled lines at k-space center')
    parser.add_argument('--nt_network', type=int, default=5, help='Time frames involved in the network.')
    parser.add_argument('--test_interval', type=int, default=20, help='Epoch intervals to test')
    parser.add_argument('--work_dir', type=str, default='crnn', help='work directory')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    # parser.add_argument('--save_fig', action='store_true', help='Save output images and masks')  # TODO
    args = parser.parse_args()

    # aug = True
    # block_size = [256, 32]
    # rotation_xy = False
    # flip_t = False

    # create work directory
    if not os.path.exists(args.work_dir):
        os.mkdir(args.work_dir)

    # dataset, each sample [n_samples, t, x, y]
    train_transform = None
    train_dataset = get_LITT_dataset(data_root=args.data_path, split='train',
                                     nt_network=args.nt_network, transform=train_transform)  # TODO: 数据预处理与数据增强
    val_dataset = get_LITT_dataset(data_root=args.data_path, split='val', nt_network=args.nt_network)
    test_dataset = get_LITT_dataset(data_root=args.data_path, split='test', nt_network=args.nt_network)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    # model, loss, optimizer
    rec_net = CRNN_MRI_UniDir()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(rec_net.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # device
    if torch.cuda.is_available():
        rec_net = rec_net.cuda()
        criterion.cuda()

    # training, validation & test
    for epoch in range(args.num_epoch):
        print(f'Epoch {epoch + 1}/{args.num_epoch}')
        step_train(train_loader, rec_net, criterion, optimizer)
        step_validate(val_loader, rec_net, criterion)
        if (epoch + 1) % args.test_interval == 0:
            step_test(test_loader, rec_net, criterion)
