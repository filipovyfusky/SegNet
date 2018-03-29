import numpy as np
import argparse
import cv2
import time

import caffe

# Use colours which match Cityscapes
# (180, 129, 69)  # 0 - Sky
# (69, 69, 69)    # 1 - Building
# (153, 153, 153) # 2 - Column/Pole
# (0, 69, 255)    # 3 - Road Marking
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
    [0, 69, 255], [128, 64, 128], [231, 35, 244],
    [35, 142, 106], [29, 170, 250], [153, 153, 190],
    [142, 0, 0], [60, 19, 219], [32, 10, 119],
])


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


def display_results(segmented_image, confidence_map):
    seg_window = "segmented_image"
    conf_window = "confidence_map"

    cv2.namedWindow(seg_window)
    cv2.namedWindow(conf_window)

    cv2.moveWindow(seg_window, 450, 750)
    cv2.moveWindow(conf_window, 1000, 750)

    cv2.imshow(seg_window, segmented_image)
    cv2.imshow(conf_window, confidence_map)

if __name__== "__main__":
    # Import arguments
    parser = argparse.ArgumentParser(description="Semantically segment video/"
                                                 "image input using SegNet.")
    parser.add_argument('model',
                        type=str,
                        help="The model description to use for inference "
                             "(.prototxt file)")
    parser.add_argument('weights',
                        type=str,
                        help="The weights to use for inference"
                             " (.caffemodel file)")
    parser.add_argument('input_source',
                        type=str,
                        help="Input source for the network. May be either a "
                             "video file, or a path to a sequence of images. To"
                             "specify images, you must use the format required "
                             "by OpenCVs VideoCapture. Reference can be found "
                             "here: https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-videocapture.")
    parser.add_argument('--cpu',
                        action='store_true',
                        default=False,
                        help="Flag to indicate whether or not to use CPU for "
                             "computation. If not set, will use GPU.")
    args = parser.parse_args()

    # Set computation mode
    if args.cpu:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()

    # Load caffe network
    net = caffe.Net(args.model,
                    args.weights,
                    caffe.TEST)

    # Access blob data
    input_shape = net.blobs['data'].data.shape
    class_output = net.blobs['classes'].data
    confidence_output = net.blobs['confidence'].data

    cap = cv2.VideoCapture(args.input_source)

    rval = True
    while rval:
        # Get image from VideoCapture
        rval, frame = cap.read()
        if not rval:
            print("No image found!")
            break

        # Resize input image
        resized_image = cv2.resize(frame, (input_shape[3], input_shape[2]))

        # Input shape is (y, x, 3), needs to be reshaped to (3, y, x)
        input_image = resized_image.transpose((2, 0, 1))

        # Inference using SegNet
        start = time.time()
        out = net.forward_all(data=input_image)
        end = time.time()
        print '%30s' % 'Executed SegNet in ', str((end - start) * 1000), 'ms'

        # Prepare segmented image results
        classes = np.squeeze(class_output).astype(np.uint8)
        segmentation_bgr = np.asarray(LABEL_COLOURS[classes]).astype(np.uint8)
        segmented_image = overlay_segmentation_results(resized_image,
                                                       segmentation_bgr)

        # Prepare confidence results
        confidence = np.squeeze(confidence_output).astype(np.float64)

        display_results(segmentation_bgr, confidence)

        key = cv2.waitKey(1)
        if key == 27:  # exit on ESC
            break

    # Cleanup windows
    cap.release()
    cv2.destroyAllWindows()
