# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2018/4/19 15:20.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""

import numpy as np
import torch


def hard_mining(neg_output, neg_labels, num_hard):
  _, idcs = torch.topk(neg_output, min(num_hard, len(neg_output)))
  neg_output = torch.index_select(neg_output, 0, idcs)
  neg_labels = torch.index_select(neg_labels, 0, idcs)
  return neg_output, neg_labels


def nms(output, nms_th):
  if len(output) == 0:
    return output

  output = output[np.argsort(-output[:, 0])]
  bboxes = [output[0]]

  for i in np.arange(1, len(output)):
    bbox = output[i]
    flag = 1
    for j in range(len(bboxes)):
      if iou(bbox[1:7], bboxes[j][1:7]) >= nms_th:
        flag = -1
        break
    if flag == 1:
      bboxes.append(bbox)

  bboxes = np.asarray(bboxes, np.float32)
  return bboxes


def iou(box0, box1):
  rzhw0 = box0[4:]
  s0 = box0[:3] - rzhw0
  e0 = box0[:3] + rzhw0

  rzhw1 = box1[4:]
  s1 = box1[:3] - rzhw1
  e1 = box1[:3] + rzhw1

  overlap = []
  for i in range(len(s0)):
    overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))

  intersection = overlap[0] * overlap[1] * overlap[2]

  dzhw0 = 2 * rzhw0
  dzhw1 = 2 * rzhw1
  union = dzhw0[0] * dzhw0[1] * dzhw0[2] + \
          dzhw1[0] * dzhw1[1] * dzhw1[2] - intersection

  return intersection / union


def acc(pbb, lbb, conf_th, nms_th, detect_th):
  pbb = pbb[pbb[:, 0] >= conf_th]
  pbb = nms(pbb, nms_th)

  tp = []
  fp = []
  fn = []
  l_flag = np.zeros((len(lbb),), np.int32)
  for p in pbb:
    flag = 0
    bestscore = 0
    for i, l in enumerate(lbb):
      score = iou(p[1:7], l)
      if score > bestscore:
        bestscore = score
        besti = i
    if bestscore > detect_th:
      flag = 1
      if l_flag[besti] == 0:
        l_flag[besti] = 1
        tp.append(np.concatenate([p, [bestscore]], 0))
      else:
        fp.append(np.concatenate([p, [bestscore]], 0))
    if flag == 0:
      fp.append(np.concatenate([p, [bestscore]], 0))
  for i, l in enumerate(lbb):
    if l_flag[i] == 0:
      score = []
      for p in pbb:
        score.append(iou(p[1:7], l))
      if len(score) != 0:
        bestscore = np.max(score)
      else:
        bestscore = 0

      if bestscore < detect_th:
        fn.append(np.concatenate([l, [bestscore]], 0))

  return tp, fp, fn, len(lbb)


def topkpbb(pbb, lbb, nms_th, detect_th, topk=30):
  conf_th = 0
  fp = []
  tp = []
  while len(tp) + len(fp) < topk:
    conf_th = conf_th - 0.2
    tp, fp, fn, _ = acc(pbb, lbb, conf_th, nms_th, detect_th)
    if conf_th < -3:
      break
  tp = np.array(tp).reshape([len(tp), 6])
  fp = np.array(fp).reshape([len(fp), 6])
  fn = np.array(fn).reshape([len(fn), 5])
  allp = np.concatenate([tp, fp], 0)
  sorting = np.argsort(allp[:, 0])[::-1]
  n_tp = len(tp)
  topk = np.min([topk, len(allp)])
  tp_in_topk = np.array([i for i in range(n_tp) if i in sorting[:topk]])
  fp_in_topk = np.array([i for i in range(topk) if sorting[i] not in range(n_tp)])
  #     print(fp_in_topk)
  fn_i = np.array([i for i in range(n_tp) if i not in sorting[:topk]])
  newallp = allp[:topk]
  if len(fn_i) > 0:
    fn = np.concatenate([fn, tp[fn_i, :5]])
  else:
    fn = fn
  if len(tp_in_topk) > 0:
    tp = tp[tp_in_topk]
  else:
    tp = []
  if len(fp_in_topk) > 0:
    fp = newallp[fp_in_topk]
  else:
    fp = []
  return tp, fp, fn
