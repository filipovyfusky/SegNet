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


def crop_input(input_image, input_shape):
    """Crops an image to the desired size, rather than resizing.

    Parameters
    ----------
    input_image:
        The image to crop.
    Return
    ------
    output_image:
        The cropped image.
    """
    ht = input_shape[2]
    wt = input_shape[3]

    hs, ws, cs = input_image.shape

    if ht == hs and wt == ws:
        return input_image

    x = (ws - wt) / 2
    y = (hs - ht) / 2

    output_image = input_image[y:y + ht, x:x + wt]

    return output_image


def overlay_segmentation_results(input_image, segmented_image):
    """Overlays the segmentation results over the original image.

    Parameters
    ----------
    input_image:
        The original unsegmented image.
    segmented_image:
        The segmented results.

    Returns
    -------
    segmented_image:
        The original image overlaid with the segmented results.
    """
    cv2.addWeighted(input_image, 0.5, segmented_image, 0.5, 0, segmented_image)

    return segmented_image

if __name__== "__main__":
    # Import arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--colours', type=str, required=True)
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
    label_colours = cv2.imread(args.colours).astype(np.uint8)

    cap = cv2.VideoCapture(input_source)

    rval = True
    while rval:

        # Get image from VideoCapture
        rval, frame = cap.read()
        if rval is False:
            break

        # Crop and reshape input image
        cropped_frame = crop_input(frame, input_shape)
        # cropped_frame = cv2.resize(frame, (input_shape[3],input_shape[2]))
        # Input shape is (y, x, 3), needs to be reshaped to (3, y, x)
        input_image = cropped_frame.transpose((2, 0, 1))

        # Inference using SegNet
        start = time.time()
        out = net.forward_all(data=input_image)
        end = time.time()
        print '%30s' % 'Executed SegNet in ', str((end - start) * 1000), 'ms'

        segmentation_ind = np.squeeze(out_data)
        segmentation_ind_3ch = np.resize(segmentation_ind,
                                         (3, input_shape[2], input_shape[3]))
        segmentation_ind_3ch = \
            segmentation_ind_3ch.transpose(1, 2, 0).astype(np.uint8)

        segmentation_rgb = np.zeros(segmentation_ind_3ch.shape, dtype=np.uint8)

        # Lookup table transform to map the right colour for each class.
        cv2.LUT(segmentation_ind_3ch, label_colours, segmentation_rgb)

        # Overlay image with segmentation results and then display.
        segmented_image = overlay_segmentation_results(cropped_frame,
                                                       segmentation_rgb)
        cv2.imshow("segmented_image", segmented_image)

        key = cv2.waitKey(1)
        if key == 27:  # exit on ESC
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
