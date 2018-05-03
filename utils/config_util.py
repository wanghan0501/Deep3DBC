# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2018/4/19 15:34.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""

config = {}

# data
config['preprocess_result_path'] = '/root/workspace/preprocessing/results/'
config['train_split_path'] = '/root/workspace/preprocessing/train.npy'
config['test_split_path'] = '/root/workspace/test_new_4_9_g.npy'
config['bbox_path'] = './results/res18/bbox/'
config['label_path'] = ''

# detector
config['anchors'] = [20.0, 40.0, 60.0]
config['chanel'] = 1
config['crop_size'] = [128, 128, 128]
config['stride'] = 4
config['max_stride'] = 16
config['num_neg'] = 800
config['th_neg'] = 0.02
config['th_pos_train'] = 0.5
config['th_pos_val'] = 1
config['num_hard'] = 2
config['bound_size'] = 12
config['reso'] = 1
config['sizelim'] = 0.
config['sizelim2'] = 20
config['sizelim3'] = 50
config['aug_scale'] = True
config['r_rand_crop'] = 0.3
config['pad_value'] = 170
config['augtype'] = {'flip': True, 'swap': False, 'scale': True, 'rotate': False}
config['blacklist'] = []

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
# config['isScale'] = True
# config['random_sample'] = True
# config['T'] = 1
# config['topk'] = 5
# config['detect_th'] = 0.05
# config['conf_th'] = -1
# config['nms_th'] = 0.05
# config['filling_value'] = 160
# config['miss_ratio'] = 1
# config['miss_thresh'] = 0.03
