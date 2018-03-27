import numpy as np
import matplotlib.pyplot as plt
import os.path
import scipy
import argparse
import math
import cv2
import sys
import time

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

LABEL_COLOURS = np.array([
    [180, 129, 69], [69, 69, 69], [153, 153, 153],
    [255, 69, 0], [128, 64, 128], [231, 35, 244],
    [35, 142, 106], [29, 170, 250], [153, 153, 190],
    [142, 0, 0], [60, 19, 219], [32, 10, 119],
])


def overlay_segmentation_results(input_image, segmentation_rgb):
    """Overlays the segmentation results over the original image.

    Parameters
    ----------
    input_image:
        The original unsegmented image.
    segmentation_rgb:
        The segmented results.

    Returns
    -------
    segmented_image:
        The original image overlaid with the segmented results.
    """

    cv2.addWeighted(input_image, 0.5, segmentation_rgb,
                    0.5,
                    0,
                    segmentation_rgb)

    return segmentation_rgb

if __name__== "__main__":
    # Import arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument('--video_file', type=str)
    parser.add_argument('--webcam_num', type=int)
    args = parser.parse_args()

    # Check if input is video_file or webcam_num
    if args.video_file is not None and args.webcam_num is None:
        input_source = args.video_file
    elif args.video_file is None and args.webcam_num is not None:
        input_source = args.webcam_num
    elif args.video_file is None and args.webcam_num is None:
        raise IOError("One of video_file or webcam_num must be specified!")
    elif args.video_file is not None and args.webcam_num is not None:
        raise IOError("Cannot specify both video_file and webcam_num!")

    # Set up caffe net
    net = caffe.Net(args.model,
                    args.weights,
                    caffe.TEST)

    if args.cpu:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()

    input_shape = net.blobs['data'].data.shape
    out_data = net.blobs['argmax'].data
    output_shape = out_data.shape

    cap = cv2.VideoCapture(input_source)

    rval = True
    while rval:

        # Get image from VideoCapture
        rval, frame = cap.read()
        if rval is False:
            break

        # Crop and reshape input image
        cropped_frame = cv2.resize(frame, (input_shape[3], input_shape[2]))
        # Input shape is (y, x, 3), needs to be reshaped to (3, y, x)
        input_image = cropped_frame.transpose((2, 0, 1))

        # Inference using SegNet
        start = time.time()
        out = net.forward_all(data=input_image)
        end = time.time()
        print '%30s' % 'Executed SegNet in ', str((end - start) * 1000), 'ms'

        # Extract segmentation indices and
        segmentation_ind = np.squeeze(out_data).astype(np.uint8)
        segmentation_bgr = np.asarray(LABEL_COLOURS[segmentation_ind]).astype(np.uint8)

        # Overlay image with segmentation results and then display.
        segmented_image = overlay_segmentation_results(cropped_frame,
                                                       segmentation_bgr)

        # Display image. Add moveWindow to prevent it from opening off screen
        cv2.imshow("segmented_image", segmentation_bgr)

        key = cv2.waitKey(1)
        if key == 27:  # exit on ESC
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
