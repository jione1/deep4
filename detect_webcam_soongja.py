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

def detect_webcam():
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

    cuda = torch.cuda.is_available()

    # Set up model
    model = Darknet(opt.config_path, img_size=opt.img_size)
    model.load_weights(opt.weights_path)

    if cuda:
        model.cuda()

    model.eval() # Set in evaluation mode

    classes = load_classes(opt.class_path) # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    print ('\nPerforming object detection:')

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        # ImageFolder에서 수행하는 작업 실행
        h, w, _ = frame.shape
        dim_diff = np.abs(h - w)
        pad1, pad2 = dim_diff // 2, dim_diff // 2
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        input_img = np.pad(frame, pad, 'constant', constant_values=127.5) / 255
        input_img = resize(input_img, (opt.img_size, opt.img_size, 3), mode='reflect')
        input_img = np.transpose(input_img, (2, 0, 1))
        input_img = torch.from_numpy(input_img).float()

         # Configure input
        input_img = Variable(input_img.type(Tensor))
        input_img = torch.unsqueeze(input_img, 0)

        # Get detections
        with torch.no_grad():
            detections = model(input_img)
            detections = non_max_suppression(detections, 80, opt.conf_thres, opt.nms_thres)

        # The amount of padding that was added
        pad_x = max(frame.shape[0] - frame.shape[1], 0) * (opt.img_size / max(frame.shape))
        pad_y = max(frame.shape[1] - frame.shape[0], 0) * (opt.img_size / max(frame.shape))
        # Image height and width after padding is removed
        unpad_h = opt.img_size - pad_y
        unpad_w = opt.img_size - pad_x

        detections = detections[0]
        # Draw bounding boxes and labels of detections
        if detections is not None:
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)

            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))

                # Rescale coordinates to original dimensions
                box_h = ((y2 - y1) / unpad_h) * frame.shape[0]
                box_w = ((x2 - x1) / unpad_w) * frame.shape[1]
                y1 = ((y1 - pad_y // 2) / unpad_h) * frame.shape[0]
                x1 = ((x1 - pad_x // 2) / unpad_w) * frame.shape[1]

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]

                start = (x1,y1)
                end = (x1 + box_w, y1 + box_h)

                cv2.rectangle(frame, start, end, color=color, thickness=2)

                #label recg
                font = cv2.FONT_HERSHEY_SIMPLEX
                t_w, t_h = cv2.getTextSize(classes[int(cls_pred)], font, 0.5, 1)[0]

                # cv2.rectangle(frame, start, (x1 + t_w, y1 + t_h),
                #               color=color, thickness=-1)
                cv2.putText(frame, classes[int(cls_pred)], (x1, y1 - t_h),
                            font, 0.5, color=color, thickness=1)

        cv2.imshow('smile detector', frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_webcam()