# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2018/4/19 15:11.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""

import torch.nn.functional as F
from torch import nn

from layers.cores import hard_mining


class DetectorLoss(nn.Module):
  def __init__(self, num_hard=0):
    super(DetectorLoss, self).__init__()
    self.sigmoid = nn.Sigmoid()
    self.classify_loss = nn.BCELoss()
    self.regress_loss = nn.SmoothL1Loss()
    self.num_hard = num_hard

  def forward(self, output, labels, train=True):
    batch_size = labels.size(0)
    output = output.view(-1, 7)
    labels = labels.view(-1, 7)

    pos_idcs = labels[:, 0] > 0.5
    pos_idcs = pos_idcs.unsqueeze(1).expand(pos_idcs.size(0), 7)
    pos_output = output[pos_idcs].view(-1, 7)
    pos_labels = labels[pos_idcs].view(-1, 7)

    neg_idcs = labels[:, 0] < -0.5
    neg_output = output[:, 0][neg_idcs]
    neg_labels = labels[:, 0][neg_idcs]

    if self.num_hard > 0 and train:
      neg_output, neg_labels = hard_mining(neg_output, neg_labels, self.num_hard * batch_size)
    neg_prob = self.sigmoid(neg_output)

    if len(pos_output) > 0:
      pos_prob = self.sigmoid(pos_output[:, 0])
      pz, ph, pw, prz, prh, prw = pos_output[:, 1], pos_output[:, 2], pos_output[:, 3], \
                                  pos_output[:, 4], pos_output[:, 5], pos_output[:, 6]
      lz, lh, lw, lrz, lrh, lrw = pos_labels[:, 1], pos_labels[:, 2], pos_labels[:, 3], \
                                  pos_labels[:, 4], pos_labels[:, 5], pos_labels[:, 6]

      regress_losses = [
        self.regress_loss(pz, lz),
        self.regress_loss(ph, lh),
        self.regress_loss(pw, lw),
        self.regress_loss(prz, lrz),
        self.regress_loss(prh, lrh),
        self.regress_loss(prw, lrw),
      ]

      regress_losses_data = [l.data.item() for l in regress_losses]
      classify_loss = 0.5 * self.classify_loss(
        pos_prob, pos_labels[:, 0]) + 0.5 * self.classify_loss(
        neg_prob, neg_labels + 1)
      pos_correct = (pos_prob.data >= 0.5).sum().item()
      pos_total = len(pos_prob)

    else:
      regress_losses = [0, 0, 0, 0, 0, 0]
      classify_loss = 0.5 * self.classify_loss(
        neg_prob, neg_labels + 1)
      pos_correct = 0
      pos_total = 0
      regress_losses_data = [0, 0, 0, 0, 0, 0]

    classify_loss_data = classify_loss.data.item()

    loss = classify_loss
    for regress_loss in regress_losses:
      loss += regress_loss

    neg_correct = (neg_prob.data < 0.5).sum().item()
    neg_total = len(neg_prob)

    return [loss, classify_loss_data] + regress_losses_data + [pos_correct, pos_total, neg_correct, neg_total]


class ClassifierLoss(nn.Module):
  def __init__(self):
    super(ClassifierLoss, self).__init__()
    self.classify_loss = F.binary_cross_entropy()

  def forward(self, output, labels):
    pass
