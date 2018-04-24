# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2018/4/19 15:44.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR

from dataset.loader import get_test_loader, get_train_loader
from layers.bounding_box_layer import PredictBoundingBox
from layers.cores import nms
from layers.loss_layer import Loss
from model.detector_model import Model
from utils.config_util import config
from utils.gpu_util import set_gpu
from utils.logger_util import Logger

parser = argparse.ArgumentParser(description='PyTorch Deep3DBC Detector')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run (default: 100)')
parser.add_argument('--seed', default=21, type=int, metavar='N',
                    help='random seed to initialize environment (default: 21)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default: 0.01')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--lr-decay-step', '--lrds', default=5, type=int,
                    metavar='LRDS', help='learning rate decay step (default: 5)')
parser.add_argument('--lr-decay-rate', '--lrdr', default=0.9, type=float,
                    metavar='LRDR', help='learning rate decay rate (default: 0.9)')
parser.add_argument('--save-freq', default='1', type=int, metavar='S',
                    help='save frequency (default: 1)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save-dir', default='', type=str, metavar='SAVE',
                    help='directory to save checkpoint (default: none)')
parser.add_argument('--test', default=0, type=int, metavar='TEST',
                    help='1 do test evaluation, 0 not')
parser.add_argument('--split', default=8, type=int, metavar='SPLIT',
                    help='In the test phase, split the image to different parts (default: 8')
parser.add_argument('--gpu', default='all', type=str, metavar='N',
                    help='use gpu (default: all)')
parser.add_argument('--n_test', default=8, type=int, metavar='N',
                    help='number of gpu for test (default: 8)')


def main():
  global args
  args = parser.parse_args()

  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)

  net = Model()
  loss = Loss(config['num_hard'])
  get_pbb = PredictBoundingBox(config)

  start_epoch = args.start_epoch
  save_dir = args.save_dir

  if args.resume:
    checkpoint = torch.load(args.resume)
    if start_epoch == 0:
      start_epoch = 1
    else:
      start_epoch = checkpoint['epoch'] + 1

    if not save_dir:
      save_dir = checkpoint['save_dir']
    else:
      save_dir = os.path.join('results', save_dir)
    net.load_state_dict(checkpoint['state_dict'])
  else:
    if start_epoch == 0:
      start_epoch = 1
    if not save_dir:
      exp_id = time.strftime('%Y%m%d-%H%M%S', time.localtime())
      save_dir = os.path.join('results', 'res18' + '-' + exp_id)
    else:
      save_dir = os.path.join('results', save_dir)

  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  logfile = os.path.join(save_dir, 'log')
  if args.test != 1:
    sys.stdout = Logger(logfile)
  args.n_gpu = set_gpu(args.gpu)
  # load net to gpu devices
  # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  net = net.cuda()
  loss = loss.cuda()
  cudnn.benchmark = True
  net = DataParallel(net)

  if args.test == 1:
    test_loader = get_test_loader(args)
    test(test_loader, net, get_pbb, save_dir, config)
    return

  train_loader = get_train_loader(args)

  optimizer = torch.optim.SGD(
    net.parameters(),
    args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay)
  scheduler = StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)

  for epoch in range(start_epoch, args.epochs + 1):
    scheduler.step()
    for param_group in optimizer.param_groups:
      print("Current LR : {}".format(param_group['lr']))
    train(train_loader, net, loss, epoch, optimizer, save_dir)


def train(data_loader, net, loss, epoch, optimizer, save_dir):
  start_time = time.time()
  net.train()

  metrics = []
  for i, (data, target, coord) in enumerate(data_loader):
    data = Variable(data.cuda(async=True))
    target = Variable(target.cuda(async=True))
    coord = Variable(coord.cuda(async=True))

    output = net(data, coord)
    loss_output = loss(output, target)
    optimizer.zero_grad()
    loss_output[0].backward()
    optimizer.step()
    loss_output[0] = loss_output[0].data[0]
    metrics.append(loss_output)

    end_time = time.time()

    if (i % 100 == 0):
      print('[Train] tpr %3.2f, tnr %3.2f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * np.sum(loss_output[6]) / np.sum(loss_output[7]),
        100.0 * np.sum(loss_output[8]) / np.sum(loss_output[9]),
        np.sum(loss_output[7]),
        np.sum(loss_output[9]),
        end_time - start_time))

  if epoch % args.save_freq == 0:
    state_dict = net.module.state_dict()
    for key in state_dict.keys():
      state_dict[key] = state_dict[key].cpu()

    torch.save({
      'epoch': epoch,
      'save_dir': save_dir,
      'state_dict': state_dict,
      'args': args},
      os.path.join(save_dir, '%03d.ckpt' % epoch))

  end_time = time.time()

  metrics = np.asarray(metrics, np.float32)
  print('Epoch %03d (lr %.5f)' % (epoch, lr))
  print('Train:      tpr %3.2f, tnr %3.2f, total pos %d, total neg %d, time %3.2f' % (
    100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
    100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
    np.sum(metrics[:, 7]),
    np.sum(metrics[:, 9]),
    end_time - start_time))

  print('Loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
    np.mean(metrics[:, 0]),
    np.mean(metrics[:, 1]),
    np.mean(metrics[:, 2]),
    np.mean(metrics[:, 3]),
    np.mean(metrics[:, 4]),
    np.mean(metrics[:, 5])))


def test(data_loader, net, get_pbb, save_dir, config):
  start_time = time.time()
  save_dir = os.path.join(save_dir, 'bbox')
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  print(save_dir)
  net.eval()
  namelist = []
  split_comber = data_loader.dataset.split_comber
  for i_name, (data, target, coord, nzhw) in enumerate(data_loader):
    target = [np.asarray(t, np.float32) for t in target]
    lbb = target[0]
    nzhw = nzhw[0]
    name = data_loader.dataset.filenames[i_name].split('-')[0].split('/')[-1].split('_clean')[0]
    namelist.append(name)
    data = data[0][0]
    coord = coord[0][0]
    is_feat = False
    if 'output_feature' in config:
      if config['output_feature']:
        is_feat = True
    n_per_run = args.n_test
    print(data.size())
    split_list = range(0, len(data) + 1, n_per_run)
    if split_list[-1] != len(data):
      split_list.append(len(data))
    output_list = []
    feature_list = []

    for i in range(len(split_list) - 1):
      input = Variable(data[split_list[i]:split_list[i + 1]], volatile=True).cuda()
      input_coord = Variable(coord[split_list[i]:split_list[i + 1]], volatile=True).cuda()
      if is_feat:
        output, feature = net(input, input_coord)
        feature_list.append(feature.data.cpu().numpy())
      else:
        output = net(input, input_coord)
      output_list.append(output.data.cpu().numpy())
    output = np.concatenate(output_list, 0)
    output = split_comber.combine(output, nzhw=nzhw)
    if is_feat:
      feature = np.concatenate(feature_list, 0).transpose([0, 2, 3, 4, 1])[:, :, :, :, :, np.newaxis]
      feature = split_comber.combine(feature, side_len=144)[..., 0]

    thresh = -3
    pbb, mask = get_pbb(output, thresh, ismask=True)

    pbb = pbb[pbb[:, 0] > 0]
    nms_th = 0.05
    pbb = nms(pbb, nms_th)

    if is_feat:
      feature_selected = feature[mask[0], mask[1], mask[2]]
      np.save(os.path.join(save_dir, name + '_feature.npy'), feature_selected)
    e = time.time()
    np.save(os.path.join(save_dir, name + '_pbb.npy'), pbb)
    np.save(os.path.join(save_dir, name + '_lbb.npy'), lbb)
  np.save(os.path.join(save_dir, 'namelist.npy'), namelist)
  end_time = time.time()

  print('elapsed time is %3.2f seconds' % (end_time - start_time))


if __name__ == '__main__':
  main()
