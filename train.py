#!/home/jieying/anaconda3/bin/python
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


def iterate_minibatch(data, nbatch_per_epoch, batch_size, shuffle=True):
    n = nbatch_per_epoch
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

    im_gnd_l = torch.from_numpy(to_tensor_format(im))
    im_und_l = torch.from_numpy(to_tensor_format(im_und))
    k_und_l = torch.from_numpy(to_tensor_format(k_und))
    mask_l = torch.from_numpy(to_tensor_format(mask, mask=True))

    return im_und_l, k_und_l, mask_l, im_gnd_l


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', type=int, nargs=1, default=['200'], help='number of epochs')
    parser.add_argument('--batch_size', type=int, nargs=1, default=['1'], help='batch size')
    parser.add_argument('--num_batch', type=int, nargs=1, default=['10'], help='number of batches in each epoch')
    parser.add_argument('--lr', type=float, nargs=1, default=['0.0001'], help='initial learning rate')
    parser.add_argument('--acceleration_factor', type=float, nargs=1,
                        default=['6.0'], help='Acceleration factor for k-space sampling')
    parser.add_argument('--sampled_lines', type=int, nargs=1,
                        default=['15'], help='Number of sampled lines at k-space center')
    parser.add_argument('--work_dir', type=str, default='crnn', help='work directory')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--save_fig', action='store_true', help='Save output images and masks')
    args = parser.parse_args()

    num_epoch = args.num_epoch
    batch_size = args.batch_size
    num_batch = args.num_batch
    acc = args.acceleration_factor  # under-sampling rate
    sample_n = args.sampled_lines  # Number of sampled lines at k-space center

    save_fig = True
    nt_network = 5

    aug = True
    block_size = [256, 32]
    rotation_xy = False
    flip_t = False

    save_every = 20

    # dataset, sample shape [n_samples, x, y, t]
    train = load_data_v2(data_path='../LITT_data/',
                         split='train', nt_network=nt_network)
    validate = load_data_v2(data_path='../LITT_data/',
                         split='validation', nt_network=nt_network)
    test = load_data_v2(data_path='../LITT_data/',
                         split='test', nt_network=nt_network)

    # if aug:
    #     train = data_aug(train, block_size, rotation_xy, flip_t)
    #     validate = data_aug(validate, block_size, rotation_xy, flip_t)
    #     test = data_aug(test, block_size, rotation_xy, flip_t)  # TODO: test 为什么还需要数据增强?

    # => [n_samples, t, x, y]
    train = train.transpose((0, 3, 1, 2))
    validate = validate.transpose((0, 3, 1, 2))
    test = test.transpose((0, 3, 1, 2))

    # model, loss, optimizer
    rec_net = CRNN_MRI_UniDir()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(rec_net.parameters(), lr=args.lr, betas=(0.5, 0.999))

    if torch.cuda.is_available():
        rec_net = rec_net.cuda()
        criterion.cuda()

    # training & validation
    for epoch in range(num_epoch):
        print(f'Epoch {epoch + 1}/{num_epoch + 1}')

        # train step
        train_loss = 0
        train_batches = 0
        rec_net.train()
        for im in iterate_minibatch(data=train, nbatch_per_epoch=num_batch, batch_size=batch_size, shuffle=True):  # TODO: 为什么不使用trainloader？
            im_u, k_u, mask, gnd = prep_input(im, acc=acc, sample_n=sample_n)

            rec = rec_net(im_u, k_u, mask)
            loss = criterion(rec, gnd)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

            if args.debug and train_batches == 20:  # TODO: 为什么要限制batch数为20？
                break
        train_loss /= train_batches
        print(time.strftime('%H:%M:%S') + ' ' + f'train'
              + ' - ' + f'loss: {train_loss}')

        # validation step
        validate_loss = 0
        validate_batches = 0
        rec_net.eval()
        for im in iterate_minibatch(data=validate, nbatch_per_epoch=1, batch_size=batch_size, shuffle=False):
            with torch.no_grad():
                im_u, k_u, mask, gnd = prep_input(im, acc=acc, sample_n=sample_n)
                pred = rec_net(im_u, k_u, mask)
                loss = criterion(pred, gnd)

            validate_loss += loss.item()
            validate_batches += 1

            if args.debug and validate_batches == 20:
                break
        validate_loss /= validate_batches
        print(time.strftime('%H:%M:%S') + ' ' + f'valid'
              + ' - ' + f'loss: {validate_loss}')

        # test step
        if epoch % save_every != 0:
            continue

        # save model
        torch.save(rec_net.state_dict(), os.path.join(args.work_dir, f'model_epoch_{epoch + 1}.pth'))

        # save test results
        vis = []
        test_loss = 0
        base_psnr = 0
        test_psnr = 0
        test_batches = 0
        for im in iterate_minibatch(data=test, nbatch_per_epoch=1, batch_size=3, shuffle=False):  # TODO batch_size=3是什么操作？
            with torch.no_grad():
                im_u, k_u, mask, gnd = prep_input(im, acc=acc, sample_n=sample_n)
                pred = rec_net(im_u, k_u, mask)
                loss = criterion(pred, gnd)

            test_loss += loss.item()
            test_batches += 1

            for im_i, und_i, pred_i in zip(im,
                                           from_tensor_format(im_u.numpy()),
                                           from_tensor_format(pred.data.cpu().numpy())):
                base_psnr += complex_psnr(im_i, und_i, peak='max')
                test_psnr += complex_psnr(im_i, pred_i, peak='max')

            if args.debug and test_batches == 20:
                break

        test_loss /= test_batches
        base_psnr /= (test_batches * batch_size)
        test_psnr /= (test_batches * batch_size)
        print(time.strftime('%H:%M:%S') + ' ' + f'test'
              + ' - ' + f'loss: {test_loss}'
              + ', ' + f'base PSNR: {base_psnr}'
              + ', ' + f'test PSNR: {test_psnr}')
