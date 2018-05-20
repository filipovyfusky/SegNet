import numpy as np
import argparse
import cv2
import time

import matplotlib.pyplot as plt

import caffe

# Use colours which match Cityscapes
LABEL_COLOURS = np.zeros([256, 3])
LABEL_COLOURS[0] = [128, 64, 128]   # Road
LABEL_COLOURS[1] = [232, 35, 244]   # Sidewalk
LABEL_COLOURS[2] = [69, 69, 69]     # Building
LABEL_COLOURS[3] = [156, 102, 102]  # Wall/Fence
LABEL_COLOURS[4] = [153, 153, 153]  # Pole
LABEL_COLOURS[5] = [30, 170, 250]   # Traffic Light
LABEL_COLOURS[6] = [0, 220, 220]    # Traffic Sign
LABEL_COLOURS[7] = [35, 142, 107]   # Vegetation
LABEL_COLOURS[8] = [152, 251, 152]  # Terrain
LABEL_COLOURS[9] = [180, 130, 70]   # Sky
LABEL_COLOURS[10] = [60, 20, 220]   # Person/Rider
LABEL_COLOURS[11] = [142, 0, 0]     # Car
LABEL_COLOURS[12] = [70, 0, 0]      # Bus/Truck
LABEL_COLOURS[13] = [32, 11, 119]   # Motorcycle/Bicycle
LABEL_COLOURS[255] = [0, 0, 0]      # VOID


def make_parser():
    """Create ArgumentParser with description

    Returns
    -------
    parser
        The customized parser.
    """
    parser = argparse.ArgumentParser(description="Evaluate the distribution"
                                                 "of logits PRIOR to the "
                                                 "Softmax function for "
                                                 "Bayesian SegNet basic")
    parser.add_argument('model',
                        type=str,
                        help="The model description to use for inference "
                             "(.prototxt file)")
    parser.add_argument('weights',
                        type=str,
                        help="The weights to use for inference"
                             " (.caffemodel file)")
    parser.add_argument('input_image',
                       type=str,
                       help="Input image to evaluate the pixel distribution on."
                            " It will run this image through the network "
                            "num_iterations times, with the batch size defined"
                            " in the .prototxt file.")

    parser.add_argument('num_iterations',
                        type=int,
                        help="The number of times to pass the image through the"
                             " network. The larger the size, the better the"
                             " distribution will represent the population. Must"
                             " be greater than 0.")
    parser.add_argument('num_pixels',
                        type=int,
                        help="The number of pixels on which to calculate the "
                             "logit histograms.")
    parser.add_argument('--cpu',
                        action='store_true',
                        default=False,
                        help="Flag to indicate whether or not to use CPU for "
                             "computation. If not set, will use GPU.")

    return parser


def display_segmentation_results(segmented_image, confidence_map, variance_map):
    seg_window = "segmented_image"
    conf_window = "confidence_map"
    var_window = "variance_map"

    cv2.namedWindow(seg_window)
    cv2.namedWindow(conf_window)
    cv2.namedWindow(var_window)

    cv2.imshow(seg_window, segmented_image)
    cv2.imshow(conf_window, confidence_map)
    cv2.imshow(var_window, variance_map)

    key = cv2.waitKey(0)


def save_image(segmented_image, confidence, normalized_uncertainty, image_prefix):
    cv2.imwrite('{}_segmented.jpg'.format(image_prefix), segmented_image)
    cv2.imwrite('{}_confidence.jpg'.format(image_prefix), confidence)
    cv2.imwrite('{}_variance.jpg'.format(image_prefix), normalized_uncertainty)


def crop_input(input, shape):
    """ target size for placeholder """
    wt = shape[0]
    ht = shape[1]
    hs, ws, cs = input.shape
    if ht == hs and wt == ws:
        return input

    x = (ws - wt) / 2
    y = (hs - ht) / 2
    return input[y:y + ht, x:x + wt]


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


def prepare_segmentation_results(probs, output_shape, num_iterations):
    probs = np.reshape(probs,
                       (num_iterations * output_shape[0],
                        output_shape[1],
                        output_shape[2],
                        output_shape[3]))

    mean_probs = np.mean(probs, axis=0, dtype=np.float64)
    var_probs = np.var(probs, axis=0, dtype=np.float64)

    # Prepare segmented image results
    classes = np.argmax(mean_probs, axis=0)
    segmentation_bgr = np.asarray(LABEL_COLOURS[classes]).astype(np.uint8)
    segmented_image = overlay_segmentation_results(resized_image,
                                                   segmentation_bgr)

    # Prepare confidence results
    confidence = np.amax(mean_probs, axis=0)

    # Prepare uncertainty results. Index variance logits by class detection
    colgrid, rowgrid = np.ogrid[:output_shape[2], :output_shape[3]]
    uncertainty = var_probs[classes, colgrid, rowgrid]

    # Calculate the index of dispersion
    iod = uncertainty / confidence

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

    return segmented_image, classes, confidence, normalized_uncertainty, iod


def prepare_logit_histograms(logits, output_shape, num_iterations, pixels):
    # Combine all trials together.
    logits = np.reshape(logits,
                        (num_iterations * output_shape[0],
                         output_shape[1],
                         output_shape[2],
                         output_shape[3]))

    mean_logits = np.mean(logits, axis=0, dtype=np.float64)
    var_logits = np.var(logits, axis=0, dtype=np.float64)

    classes = np.argmax(mean_logits, axis=0)

    for count, p in enumerate(pixels):
        pix_logits = np.zeros(logits.shape[0])
        for n in xrange(0, logits.shape[0]):
            pix_logits[n] = logits[n, classes[tuple(p)], p[0], p[1]]

        # Create histogram
        fig = plt.figure()
        fig = plt.hist(pix_logits)
        mean = mean_logits[classes[tuple(p)], p[0], p[1]]
        var = var_logits[classes[tuple(p)], p[0], p[1]]
        plt.xlabel("Logit Value")
        plt.ylabel("Frequency")
        plt.title('Logit Histogram for Pixel [{0}, {1}],'
                  ' Class {2}, {3} Iterations'
                  .format(p[0], p[1],
                          classes[tuple(p)],
                          num_iterations*output_shape[0]))
        plt.savefig('logit_histogram_pixel_{0}.png'.format(count))

    plt.close('all')


def prepare_softmax_histograms(probs, output_shape, num_iterations, pixels):
    # Combine all trials together.
    probs = np.reshape(probs,
                        (num_iterations * output_shape[0],
                         output_shape[1],
                         output_shape[2],
                         output_shape[3]))

    mean_probs = np.mean(probs, axis=0, dtype=np.float64)
    var_probs = np.var(probs, axis=0, dtype=np.float64)

    classes = np.argmax(mean_probs, axis=0)

    for count, p in enumerate(pixels):
        pix_probs = np.zeros(probs.shape[0])
        for n in xrange(0, probs.shape[0]):
            pix_probs[n] = probs[n, classes[tuple(p)], p[0], p[1]]

        # Create histogram
        fig = plt.figure()
        fig = plt.hist(pix_probs)
        mean = mean_probs[classes[tuple(p)], p[0], p[1]]
        var = var_probs[classes[tuple(p)], p[0], p[1]]
        plt.xlabel("Class Probability")
        plt.ylabel("Frequency")
        plt.title('Probability Histogram for Pixel [{0}, {1}],'
                  ' Class {2}, {3} Iterations'
                  .format(p[0], p[1],
                          classes[tuple(p)],
                          num_iterations * output_shape[0]))
        plt.savefig('prob_histogram_pixel_{0}.png'.format(count))

    plt.close('all')

if __name__ == "__main__":
    # Import arguments
    parser = make_parser()
    args = parser.parse_args()

    # Set computation mode
    if args.cpu:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()

    # ensure the number of iterations is valid.
    if args.num_iterations is 0:
        raise ValueError("Num iterations must be greater than zero!")

    # Load Caffe network
    net = caffe.Net(args.model, args.weights, caffe.TEST)

    # Access blob data
    # TODO: Ensure bayesian segnet basic and bayesian segnet have same name for this layer.
    input_shape = net.blobs['data'].data.shape
    logit_blob = net.blobs['dense_softmax_inner_prod']
    prob_trial = net.blobs['prob'].data
    output_shape = prob_trial.shape

    # Read image
    im = cv2.imread(args.input_image, cv2.IMREAD_COLOR)

    # Create storage variables for all trials
    logits = np.zeros((args.num_iterations,) + output_shape)
    probs = np.zeros((args.num_iterations,) + output_shape)

    # Resize input image
    resized_image = crop_input(im, (input_shape[3], input_shape[2]))

    # Input shape is (y, x, 3), needs to be reshaped to (3, y, x)
    input_image = resized_image.transpose((2, 0, 1))

    # Repeat image according to batch size for inference.
    input_image = np.repeat(input_image[np.newaxis, :, :, :],
                            input_shape[0],
                            axis=0)

    for n in xrange(0, args.num_iterations):
        # Inference using Bayesian SegNet
        start = time.time()
        net.forward_all(data=input_image)
        end = time.time()
        print '%30s' % 'Executed Bayesian SegNet in ',\
            str((end - start) * 1000), 'ms'

        logits[n, :] = logit_blob.data
        probs[n, :] = prob_trial

    # Prepare and display segmentation results
    segmented_image, classes, confidence, normalized_uncertainty, iod = \
        prepare_segmentation_results(probs, output_shape, args.num_iterations)
    # display_segmentation_results(segmented_image, confidence, normalized_uncertainty)

    # Generate random pixels for viewing histograms.
    # Create an array with random pixel values from which to sample.
    pixels = np.zeros((args.num_pixels, 2), dtype=np.int)
    rows = np.random.randint(output_shape[2], size=args.num_pixels)
    cols = np.random.randint(output_shape[3], size=args.num_pixels)
    pixels[:, 0] = rows
    pixels[:, 1] = cols

    prepare_logit_histograms(logits,
                             output_shape,
                             args.num_iterations,
                             pixels)
    prepare_softmax_histograms(probs,
                               output_shape,
                               args.num_iterations,
                               pixels)

    # Cleanup windows
    cv2.destroyAllWindows()

