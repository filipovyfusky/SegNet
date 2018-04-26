import numpy as np
import argparse
import cv2
import time

import caffe

# Use colours which match Cityscapes
# (180, 129, 69)  # 0 - Sky
# (69, 69, 69)    # 1 - Building
# (153, 153, 153) # 2 - Column/Pole
# (128, 64, 128)  # 3 - Road
# (231, 35, 244)  # 4 - Sidewalk
# (35, 142, 106)  # 5 - Tree
# (29, 170, 250)  # 6 - SignSymbol
# (153, 153, 190) # 7 - Fence
# (142, 0 , 0)    # 8 - Car
# (60, 19, 219)   # 9 - Pedestrian
# (32, 10, 119)   # 10 - Cyclist
# TODO(jskhu): Fix up label names
LABEL_COLOURS = np.array([[32, 10, 119], [128, 64, 128], [231, 35, 244], [69, 69, 69]
                # -1= ground 0 = road, 1 = sidewalk, 2 = building
                ,[156, 102, 102], [153, 153, 190], [153, 153, 153]
                # 3 = wall, 4 = fence, 5 = pole
                ,[29, 170, 250], [0, 219, 219], [35, 142, 106]
                # 6 = traffic light, 7 = traffic sign, 8 = vegetation
                ,[152, 250, 152], [180, 129, 69], [60, 19, 219]
                # 9 = terrain, 10 = sky, 11 = person
                , [142, 0, 0], [69, 0, 0], [100, 60, 0]])
                # 12 = rider, 13 = car, 14 = truck


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


def display_results(segmented_image, confidence_map, variance_map):
    seg_window = "segmented_image"
    conf_window = "confidence_map"
    var_window = "variance_map"

    cv2.namedWindow(seg_window)
    cv2.namedWindow(conf_window)
    cv2.namedWindow(var_window)

    cv2.moveWindow(seg_window, 100, 500)
    cv2.moveWindow(conf_window, 600, 500)
    cv2.moveWindow(var_window, 1100, 500)

    cv2.imshow(seg_window, segmented_image)
    cv2.imshow(conf_window, confidence_map)
    cv2.imshow(var_window, variance_map)


def save_image(segmented_image, confidence, normalized_uncertainty, image_prefix):
    cv2.imwrite('{}_segmented.jpg'.format(image_prefix), segmented_image)
    cv2.imwrite('{}_confidence.jpg'.format(image_prefix), confidence)
    cv2.imwrite('{}_variance.jpg'.format(image_prefix), normalized_uncertainty)


def run_inference(net, image, input_shape, confidence_output):
    """
    Runs through SegNet inference
    :param net:
    :param image:
    :param input_shape:
    :param confidence_output:
    :return:
    """

    # Resize input image
    # Check if image and input image are the same shape, no need for resize
    resized_image = cv2.resize(image, (input_shape[3], input_shape[2]))

    # Input shape is (y, x, 3), needs to be reshaped to (3, y, x)
    input_image = resized_image.transpose((2, 0, 1))

    # Repeat image according to batch size for inference.
    input_image = np.repeat(input_image[np.newaxis, :, :, :],
                            input_shape[0],
                            axis=0)

    # Inference using Bayesian SegNet
    start = time.time()
    out = net.forward_all(data=input_image)
    end = time.time()
    print '%30s' % 'Executed Bayesian SegNet in ', \
        str((end - start) * 1000), 'ms'

    mean_confidence = np.mean(confidence_output, axis=0, dtype=np.float64)
    var_confidence = np.var(confidence_output, axis=0, dtype=np.float64)

    # Prepare segmented image results
    classes = np.argmax(mean_confidence, axis=0)
    segmentation_bgr = np.asarray(LABEL_COLOURS[classes]).astype(np.uint8)
    segmented_image = overlay_segmentation_results(resized_image,
                                                   segmentation_bgr)

    # Prepare confidence results
    confidence = np.amax(mean_confidence, axis=0)
    print('Mean Confidence', np.mean(confidence, axis=0))

    # Prepare uncertainty results
    uncertainty = np.mean(var_confidence,
                          axis=0,
                          dtype=np.float64)

    print('Mean Uncertainty', np.sqrt(np.mean(uncertainty)))

    # Normalize variance for display
    normalized_uncertainty = np.zeros((uncertainty.shape[0],
                                       uncertainty.shape[1]),
                                      np.float64)


    cv2.normalize(uncertainty,
                  normalized_uncertainty,
                  0,
                  1,
                  cv2.NORM_MINMAX,
                  cv2.CV_64FC1)

    print("Mean ")
    return segmented_image, confidence, normalized_uncertainty


def save_video_inference(input_source, model, weights, image_prefix, gpu):
    caffe.set_gpu(gpu)
    caffe.set_mode_gpu()

    # Load Caffe network
    net = caffe.Net(model, weights, caffe.TEST)

    # Access blob data
    input_shape = net.blobs['data'].data.shape
    confidence_output = net.blobs['prob'].data

    video_capture = cv2.VideoCapture(input_source)

    more_images = True
    count = 0
    while more_images:
        more_images, frame = cap.read()
        if more_images:
            segmented_image, confidence, normalized_uncertainty = run_inference(net, frame, input_shape, confidence_output)
            save_image(segmented_image, confidence, normalized_uncertainty, '{}_{}'.format(image_prefix, count))
            count += 1


def save_images_inferences(image_locs, model, weights, image_prefix, gpu):
    """
    Runs images through inference and calculates the segmentation on them
    TODO(jskhu): Currently runs through network for EVERY image. Can evaluate all images with one pass through
    :param image_locs:
    :param model:
    :param weights:
    :param image_prefix:
    :param gpu:
    :return:
    """
    caffe.set_device(gpu)
    caffe.set_mode_gpu()

    # Load Caffe network
    net = caffe.Net(model, weights, caffe.TEST)

    # Access blob data
    input_shape = net.blobs['data'].data.shape
    confidence_output = net.blobs['prob'].data

    for count, image_loc in enumerate(image_locs):
        image = cv2.imread(image_loc)
        segmented_image, confidence, normalized_uncertainty = run_inference(net, image, input_shape, confidence_output)
        save_image(segmented_image, confidence, normalized_uncertainty, '{}_{}'.format(image_prefix, count))


def make_parser():
    """Create ArgumentParser with description

    Returns
    -------
    parser
        The customized parser.
    """
    parser = argparse.ArgumentParser(description="Semantically segment video/"
                                                 "image input using Bayesian"
                                                 " SegNet or Bayesian SegNet"
                                                 " Basic.")
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

    return parser


if __name__ == "__main__":
    # Import arguments
    parser = make_parser()
    args = parser.parse_args()

    # Set computation mode
    if args.cpu:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()

    # Load Caffe network
    net = caffe.Net(args.model, args.weights, caffe.TEST)

    # Access blob data
    input_shape = net.blobs['data'].data.shape
    confidence_output = net.blobs['prob'].data

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

        # Repeat image according to batch size for inference.
        input_image = np.repeat(input_image[np.newaxis, :, :, :],
                                input_shape[0],
                                axis=0)

        # Inference using Bayesian SegNet
        start = time.time()
        out = net.forward_all(data=input_image)
        end = time.time()
        print '%30s' % 'Executed Bayesian SegNet in ',\
            str((end - start) * 1000), 'ms'

        mean_confidence = np.mean(confidence_output, axis=0, dtype=np.float64)
        var_confidence = np.var(confidence_output, axis=0, dtype=np.float64)

        # Prepare segmented image results
        classes = np.argmax(mean_confidence, axis=0)
        segmentation_bgr = np.asarray(LABEL_COLOURS[classes]).astype(np.uint8)
        segmented_image = overlay_segmentation_results(resized_image,
                                                       segmentation_bgr)

        # Prepare confidence results
        confidence = np.amax(mean_confidence, axis=0)

        # Prepare uncertainty results
        uncertainty = np.mean(var_confidence,
                              axis=0,
                              dtype=np.float64)

        print(np.sqrt(np.mean(uncertainty)))

        # Normalize variance for display
        normalized_uncertainty = np.zeros((uncertainty.shape[0],
                                          uncertainty.shape[1]),
                                          np.float64)


        cv2.normalize(uncertainty,
                      normalized_uncertainty,
                      0,
                      1,
                      cv2.NORM_MINMAX,
                      cv2.CV_64FC1)

        display_results(segmented_image, confidence, normalized_uncertainty)

        print("Mean ")

        key = cv2.waitKey(1)
        if key == 27:  # exit on ESC
            break

    # Cleanup windows
    cap.release()
    cv2.destroyAllWindows()
