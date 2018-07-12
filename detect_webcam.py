from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import cv2


parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', type=str, default='data/samples', help='path to dataset')
parser.add_argument('--config_path', type=str, default='config/yolov3.cfg', help='path to model config file')
parser.add_argument('--weights_path', type=str, default='weights/yolov3.weights', help='path to weights file')
parser.add_argument('--class_path', type=str, default='data/coco.names', help='path to class label file')
parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

os.makedirs('output', exist_ok=True)

# Set up model
model = Darknet(opt.config_path, img_size=opt.img_size)
model.load_weights(opt.weights_path)

if cuda:
    model.cuda()

model.eval() # Set in evaluation mode
classes = load_classes(opt.class_path) # Extracts class labels from file

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

print ('\nPerforming object detection:')
prev_time = time.time()

cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frameR = cv2.resize(frame, [opt.img_size, opt.img_size], interpolation=cv2.INTER_AREA)

     # Configure input
    input_imgs = Variable(frameR.type(Tensor))

    # Get detections
    with torch.no_grad():
        detections = model(input_imgs)
        detections = non_max_suppression(detections, 80, opt.conf_thres, opt.nms_thres)

    # Create plot
    #frameR을 numpy array로 변환!
    img = np.array(frameR)
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # The amount of padding that was added
    pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
    # Image height and width after padding is removed
    unpad_h = opt.img_size - pad_y
    unpad_w = opt.img_size - pad_x

    # Draw bounding boxes and labels of detections
    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            print('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))

            # Rescale coordinates to original dimensions
            box_h = ((y2 - y1) / unpad_h) * img.shape[0]
            box_w = ((x2 - x1) / unpad_w) * img.shape[1]
            y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
            x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]

            start = (x1,y1)
            end = (x1 + box_w, y1+box_h)
            cv2.rectangle(frameR, start, end, color=color, thickness=2)
            #label recg
            font = cv2.FONT_HERSHEY_SIMPLEX
            t_w, t_h = cv2.getTextSize(classes[int(cls_pred)], font, 0.5, 1)[0]

            cv2.rectangle(frameR, start, (start + t_w, start + t_h),
                          color=color, thickness=-1)
            cv2.putText(frameR, classes[int(cls_pred)], (start, start + t_h - 1),
                        font, 0.5, color='white', thickness=1)

    cv2.imshow('image', frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()