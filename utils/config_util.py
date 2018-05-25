# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2018/4/19 15:34.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""

config = {}

# data
config['preprocess_result_path'] = '/root/workspace/preprocessing/py3_results/'
config['train_split_path'] = '/root/workspace/preprocessing/train.npy'
config['test_split_path'] = '/root/workspace/x.npy'
config['bbox_path'] = './results/res18/bbox/'
config['label_path'] = ''

# detector
config['detector_anchors'] = [30.0, 60.0, 100.0]
config['detector_chanel'] = 1
config['detector_crop_size'] = [192, 192, 192]
config['detector_stride'] = 4
config['detector_bound_size'] = 12
config['detector_max_stride'] = 16
config['detector_num_neg'] = 800
config['detector_th_neg'] = 0.02
config['detector_th_pos_train'] = 0.5
config['detector_th_pos_val'] = 1
config['detector_num_hard'] = 2
config['detector_reso'] = 1
config['sizelim'] = 0.
config['sizelim2'] = 20
config['sizelim3'] = 50
config['detector_aug_scale'] = True
config['detector_r_rand_crop'] = 0.3
config['detector_pad_value'] = 170
config['detector_augtype'] = {'flip': True, 'swap': False, 'scale': True, 'rotate': False}
config['detector_blacklist'] = []

# classifier
# config['classifier_crop_size'] = [96, 96, 96]
# config['topk'] = 5
# config['resample'] = None
# config['preload_train'] = True
# config['padmask'] = False
# config['crop_size'] = [96, 96, 96]
# config['scaleLim'] = [0.85, 1.15]
# config['radiusLim'] = [6, 100]
# config['jitter_range'] = 0.15
# config['is_scale'] = True
# config['random_sample'] = True
# config['T'] = 1
# config['topk'] = 5
# config['detect_th'] = 0.05
# config['conf_th'] = -1
# config['nms_th'] = 0.05
# config['filling_value'] = 160
# config['miss_ratio'] = 1
# config['miss_thresh'] = 0.03
