import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2

import caffe

# Use colours which match Cityscapes
# (180, 129, 69)  # 0 - Sky
# (69, 69, 69)    # 1 - Building
# (153, 153, 153) # 2 - Pole
# (255, 69, 0)    # 3 - Road Marking
# (128, 64, 128)  # 4 - Road
# (231, 35, 244)  # 5 - Sidewalk
# (35, 142, 106)  # 6 - Tree
# (29, 170, 250)  # 7 - SignSymbol
# (153, 153, 190) # 8 - Fence
# (142, 0 , 0)    # 9 - Car
# (60, 19, 219)   # 10 - Pedestrian
# (32, 10, 119)   # 11 - Cyclist

LABEL_COLOURS = [
    [180, 129, 69], [69, 69, 69], [153, 153, 153],
    [255, 69, 0], [128, 64, 128], [231, 35, 244],
    [35, 142, 106], [29, 170, 250], [153, 153, 190],
    [142, 0, 0], [60, 19, 219], [32, 10, 119],
]

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--cpu', action='store_true', default=False)
args = parser.parse_args()


if args.cpu:
    caffe.set_mode_cpu()
else:
    caffe.set_mode_gpu()

net = caffe.Net(args.model, args.weights, caffe.TEST)

input_shape = net.blobs['data'].data.shape
label_colours = cv2.imread(args.colours).astype(np.uint8)

with open(args.data) as f:
    for line in f:
        input_image_file, ground_truth_file = line.split()

    input_image_raw = caffe.io.load_image(input_image_file)
    ground_truth = cv2.imread(ground_truth_file, 0)

    input_image = caffe.io.resize_image(input_image_raw,
                                        (input_shape[2],
                                         input_shape[3]))
    input_image = input_image * 255
    input_image = input_image.transpose((2, 0, 1))
    input_image = input_image[(2, 1, 0), :, :]
    input_image = np.asarray([input_image])
    input_image = np.repeat(input_image, input_shape[0], axis=0)

    out = net.forward_all(data=input_image)

    predicted = net.blobs['prob'].data

    output = np.mean(predicted, axis=0)
    uncertainty = np.var(predicted, axis=0)
    ind = np.argmax(output, axis=0)

    segmentation_ind_3ch = np.resize(ind, (3, input_shape[2], input_shape[3]))
    segmentation_ind_3ch = \
        segmentation_ind_3ch.transpose(1, 2, 0).astype(np.uint8)
    segmentation_rgb = np.zeros(segmentation_ind_3ch.shape, dtype=np.uint8)

    gt_ind_3ch = np.resize(ground_truth, (3, input_shape[2], input_shape[3]))
    gt_ind_3ch = gt_ind_3ch.transpose(1, 2, 0).astype(np.uint8)
    gt_rgb = np.zeros(gt_ind_3ch.shape, dtype=np.uint8)

    cv2.LUT(segmentation_ind_3ch, label_colours, segmentation_rgb)
    cv2.LUT(gt_ind_3ch, label_colours, gt_rgb)

    uncertainty = np.transpose(uncertainty, (1, 2, 0))

    average_unc = np.mean(uncertainty,axis=2)
    min_average_unc = np.min(average_unc)
    max_average_unc = np.max(average_unc)
    max_unc = np.max(uncertainty)

    plt.imshow(input_image_raw, vmin=0, vmax=255)
    plt.figure()
    plt.imshow(segmentation_rgb, vmin=0, vmax=255)
    plt.figure()
    plt.imshow(gt_rgb, vmin=0, vmax=255)
    plt.set_cmap('bone_r')
    plt.figure()
    plt.imshow(average_unc, vmin=0, vmax=max_average_unc)
    plt.show()

    print 'Processed: ', input_image_file

print 'Success!'

