import argparse
import os
import time

import torch
import numpy as np

from data import load_data_v2, data_aug
from model import CRNN_MRI_UniDir
import compressed_sensing as cs
from metric import complex_psnr
from utils import from_tensor_format, to_tensor_format


def iterate_minibatch(data, batch_size, shuffle=True):
    """
    yield batches in the given dataset
    :param data: data set with shape [n_samples, t, x, y]
    :param batch_size: ...
    :param shuffle: ...
    :return:
    """
    n = data.shape[0]
    if shuffle:
        data = np.random.permutation(data)
    for i in range(0, n, batch_size):
        yield data[i:i + batch_size]


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


def step_train(data_generator, model, criterion, optimizer):
    train_loss = 0
    train_batches = 0
    model.train()
    for im in data_generator:
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


def step_validate(data_generator, model, criterion):
    validate_loss = 0
    validate_batches = 0
    model.eval()
    for im in data_generator:
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


def step_test(data_generator, model, criterion):
    test_loss = 0
    base_psnr = 0
    test_psnr = 0
    test_batches = 0
    for im in data_generator:
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

    # data
    train = load_data_v2(data_path=args.data_path,
                         split='train', nt_network=args.nt_network)  # each sample [n_samples, x, y, t]
    validate = load_data_v2(data_path=args.data_path, split='validation', nt_network=args.nt_network)  # ...
    test = load_data_v2(data_path=args.data_path, split='test', nt_network=args.nt_network)  # ...

    # TODO: 为什么不是在线的数据增强？
    # if aug:
    #     train = data_aug(train, block_size, rotation_xy, flip_t)
    #     validate = data_aug(validate, block_size, rotation_xy, flip_t)
    #     test = data_aug(test, block_size, rotation_xy, flip_t)  # TODO: test 为什么还需要数据增强?

    train = train.transpose((0, 3, 1, 2))  # each sample => [n_samples, t, x, y]
    validate = validate.transpose((0, 3, 1, 2))  # ...
    test = test.transpose((0, 3, 1, 2))  # ...

    # TODO: 改为使用dataloader
    train_generator = iterate_minibatch(data=train, batch_size=args.batch_size, shuffle=True)
    validate_generator = iterate_minibatch(data=validate, batch_size=1, shuffle=False)
    test_generator = iterate_minibatch(data=test, batch_size=1, shuffle=False)

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
        step_train(train_generator, rec_net, criterion, optimizer)
        step_validate(validate_generator, rec_net, criterion)
        if (epoch + 1) % args.test_interval == 0:
            step_test(test_generator, rec_net, criterion)
