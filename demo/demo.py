import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from utils.utils import color_list,vis_detections
import torch
from torch.autograd import Variable
import numpy as np
import cv2
from data import COCODetection, COCO_ROOT, COCOAnnotationTransform
from models.refinedetlite import build_refinedet
from data import COCO_CLASSES as labels

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
net = build_refinedet('test', 320, 81)    # initialize SSD
net.load_weights('../weights/RefineDetLiteCOCO/RefineDet320_COCO_138000.pth')
testset = COCODetection(COCO_ROOT,"val2017", None, COCOAnnotationTransform())
img_id = 121
image = testset.pull_image(img_id)

x = cv2.resize(image, (320, 320)).astype(np.float32)
x -= (104.0, 117.0, 123.0)
x = x.astype(np.float32)
x = x[:, :, ::-1].copy()

x = torch.from_numpy(x).permute(2, 0, 1)

xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable

if torch.cuda.is_available():
    xx = xx.cuda()

y = net(xx)

detections = y.data


detections = detections.cpu().numpy()
detections = detections[0]
h,w,_ = image.shape
for i in range(1,detections.shape[0]):

    det = detections[i]
    boxes = det[:,1:]
    scores = det[:,0]
    boxes[:, 0] *= w
    boxes[:, 2] *= w
    boxes[:, 1] *= h
    boxes[:, 3] *= h

    cls_dets = np.hstack((boxes,scores[:, np.newaxis])).astype(np.float32,
                                                         copy=False)


    color = color_list[i-1].tolist()
    label_name = labels[i-1]
    vis_detections(image, label_name, color, cls_dets, thresh=0.1)

cv2.imwrite("test.jpg",image)