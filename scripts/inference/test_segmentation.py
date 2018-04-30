#!/usr/bin/env python
import argparse
import os
import timeit
import sys; sys.path.insert(0, os.path.expanduser("~/free_space_estimation/baselines/segnet/caffe-segnet/python"))

import caffe
import cv2
import numpy as np

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument("--out_dir",
                    help="Output absolute path (images and ground truth)")
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--iter', type=int, required=True)
parser.add_argument('--time', action="store_true", help="Flag displays inference time")
args = parser.parse_args()

caffe.set_mode_gpu()

net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)
out_dir = args.out_dir
show_time = args.time

TrafficLightSign = [192,128,128]
LaneMarking = [255,69,0]
DontCare = [0,0,0]
Person = [64,64,0]
Car = [64,0,128]
TwoWheeler = [0,128,192]
Road = [128,64,128]
Ground = [128,128,128]
Barrier = [64,64,128]
Crosswalk = [128,0,0]
Curb = [192,192,128]
Sidewalk = [60,40,222]
LaneSeperator = [128,128,0]
Animal = [0, 69, 255]

class_colours = [TrafficLightSign, LaneMarking, DontCare, Person, Car,
                 TwoWheeler, Road, Ground, Barrier, Crosswalk, Curb, Sidewalk,
                 LaneSeperator, Animal]
elapsed_sum = 0
count = 0


for count in range(0, args.iter):
    # run net, get timing info if argument is given
    if show_time:
        start_time = timeit.default_timer()
        net.forward()
        elapsed = timeit.default_timer() - start_time
        elapsed_sum += elapsed
    else:
        net.forward()

    label = net.blobs['label'].data
    predicted = net.blobs['prob'].data
    output = np.squeeze(predicted[0, :, :, :])
    ind = np.argmax(output, axis=0)
    r = ind.copy()
    g = ind.copy()
    b = ind.copy()
    r_gt = label.copy()
    g_gt = label.copy()
    b_gt = label.copy()

    for l, class_ in enumerate(class_colours):
        r[ind == l] = class_[0]
        g[ind == l] = class_[1]
        b[ind == l] = class_[2]
        r_gt[label == l] = class_[0]
        g_gt[label == l] = class_[1]
        b_gt[label == l] = class_[2]

    rgb = np.zeros((ind.shape[0], ind.shape[1], 3)).astype(np.uint8)
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b

    rgb_gt = np.zeros((ind.shape[0], ind.shape[1], 3)).astype(np.uint8)
    rgb_gt[:, :, 0] = r_gt
    rgb_gt[:, :, 1] = g_gt
    rgb_gt[:, :, 2] = b_gt

    inf_path = os.path.join(out_dir, "Inference/", "{}.png".format(str(count).zfill(6)))
    gt_path = os.path.join(out_dir, "Ground_Truth/", "{}.png".format(str(count).zfill(6)))

    cv2.imwrite(inf_path,  cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(gt_path,  cv2.cvtColor(rgb_gt, cv2.COLOR_RGB2BGR))

    if show_time:
        print("{}: Time: {} {} {}".format(count, elapsed, inf_path, gt_path))
    else:
        print("{}: {} {}".format(count, inf_path, gt_path))


if show_time:
    print("Average Time: {}".format(elapsed_sum/args.iter))
else:
    print("Success!")