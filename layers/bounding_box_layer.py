# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2018/4/19 15:29.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""

import numpy as np


class PredictBoundingBox(object):
  def __init__(self, config):
    self.stride = config['detector_stride']
    self.anchors = np.asarray(config['detector_anchors'])

  def __call__(self, output, thresh=-3, is_mask=False):
    stride = self.stride
    anchors = self.anchors
    output = np.copy(output)
    offset = (float(stride) - 1) / 2
    output_size = output.shape
    oz = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride)
    oh = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)
    ow = np.arange(offset, offset + stride * (output_size[2] - 1) + 1, stride)

    output[:, :, :, :, 1] = oz.reshape((-1, 1, 1, 1)) + output[:, :, :, :, 1] * anchors.reshape((1, 1, 1, -1))
    output[:, :, :, :, 2] = oh.reshape((1, -1, 1, 1)) + output[:, :, :, :, 2] * anchors.reshape((1, 1, 1, -1))
    output[:, :, :, :, 3] = ow.reshape((1, 1, -1, 1)) + output[:, :, :, :, 3] * anchors.reshape((1, 1, 1, -1))
    output[:, :, :, :, 4] = np.exp(output[:, :, :, :, 4]) * anchors.reshape((1, 1, 1, -1))
    output[:, :, :, :, 5] = np.exp(output[:, :, :, :, 5]) * anchors.reshape((1, 1, 1, -1))
    output[:, :, :, :, 6] = np.exp(output[:, :, :, :, 6]) * anchors.reshape((1, 1, 1, -1))
    mask = output[..., 0] > thresh
    xx, yy, zz, aa = np.where(mask)

    output = output[xx, yy, zz, aa]

    if is_mask:
      return output, [xx, yy, zz, aa]
    else:
      return output
