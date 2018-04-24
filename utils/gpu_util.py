# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2018/4/19 15:40.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""
import os

import pynvml


def get_freeId():
  pynvml.nvmlInit()

  def get_free_ratio(id):
    handle = pynvml.nvmlDeviceGetHandleByIndex(id)
    use = pynvml.nvmlDeviceGetUtilizationRates(handle)
    ratio = 0.5 * (float(use.gpu + float(use.memory)))
    return ratio

  deviceCount = pynvml.nvmlDeviceGetCount()
  available = []
  for i in range(deviceCount):
    if get_free_ratio(i) < 10:
      available.append(i)
  gpus = ''
  for g in available:
    gpus = gpus + str(g) + ','
  gpus = gpus[:-1]
  return gpus


def set_gpu(gpuinput):
  freeids = getFreeId()
  if gpuinput == 'all':
    gpus = freeids
  else:
    gpus = gpuinput
    if any([g not in freeids for g in gpus.split(',')]):
      raise ValueError('gpu ' + g + ' is being used')
  print('using gpu ' + gpus)
  os.environ['CUDA_VISIBLE_DEVICES'] = gpus
  return len(gpus.split(','))
