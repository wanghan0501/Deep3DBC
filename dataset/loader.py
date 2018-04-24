# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2018/4/19 16:12.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""
from torch.utils.data import DataLoader

from dataset.data_detector import DetectorDataset, collate
from layers.split_combine_layer import SplitComb
from utils.config_util import config


def get_train_loader(args):
  dataset = DetectorDataset(
    config,
    phase='train')

  train_loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    #num_workers=args.workers,
    pin_memory=True)
  return train_loader


def get_test_loader(args):
  margin = 32
  sidelen = 144
  split_comber = SplitComb(sidelen, config['max_stride'], config['stride'], margin, config['pad_value'])
  dataset = DetectorDataset(
    config,
    phase='test',
    split_comber=split_comber)

  test_loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    #num_workers=args.workers,
    collate_fn=collate,
    pin_memory=False)

  return test_loader
