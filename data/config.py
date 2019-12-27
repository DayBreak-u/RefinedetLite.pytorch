# config.py
import os.path

# gets home dir cross platform
# HOME = os.path.expanduser("~")
HOME = '/mnt/data1/yanghuiyu/datas/coco/coco'

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (0.485 * 255, 0.456 * 255, 0.406 * 255)


# RefineDet CONFIGS
voc_refinedet =  {
        'num_classes': 21,
        'lr_steps': (80000, 100000, 120000),
        'max_iter': 120000,
        # 'feature_maps': [40, 20, 10, 5],
        'feature_maps': [20, 10, 5, 3],
        'min_dim': 320,
        # 'steps': [8, 16, 32, 64],
        'steps': [16, 32, 64, 107],
        'min_sizes': [32, 64, 128, 256],
        'max_sizes': [],
        "mbox":[3,3,3,3],
        'aspect_ratios': [[2], [2], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,
        'name': 'RefineDet_VOC_320',
    }

#
coco_refinedet = {
    'num_classes': 81,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [20, 10, 5, 3],
    'min_dim': 320,
    'steps': [16, 32, 64, 107],
    'min_sizes': [32, 64, 128, 256],
    'max_sizes': [],
    "mbox":[6,6,6,6],
    'aspect_ratios': [ [2,3], [2,3], [2,3], [2,3]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}
