#!/usr/bin/env python
from glob import glob
import argparse
import cv2
import numpy as np
import os

# modifiable constants
""""
---------- MODIFIABLE CONSTANTS ----------
IMG_PATH[0]: Training images (Inputs to the Neural Network)
IMG_PATH[1]: Ground truth of training images
OUT_PATH: Directory where processed images will be made
TXT_PATH: 


"""
IMG_PATH = [
    os.path.expanduser("~/datasets/cityscapes/leftImg8bit/train/*/*.png"),
    os.path.expanduser("~/datasets/cityscapes/gtFine/train/*/*gtFine_labelTrainIds.png")]
OUT_PATH = os.path.expanduser("~/wave/SegNet/datasets/dataset_1/")
DATA_TYPE = "train"
TXT_PATH = ""
RESIZE_IMGS = True
WIDTH, HEIGHT = 1024, 352
INTERPOLATION = [cv2.INTER_CUBIC, cv2.INTER_NEAREST]
CROP_TO_ASPECT_RATIO = True
CROP_HEIGHT_POSITION = 'bottom'
CROP_WIDTH_POSITION = 'middle'

def get_args(arg_vals=None):
    """
    parse any relevant runtime arguments
    :param arg_vals:    args to check (None defaults to command-line args)
    :return:            argparse.Namespace object
    """

    global WIDTH, HEIGHT, IMG_PATH, OUT_PATH, DATA_TYPE, TXT_PATH, RESIZE_IMGS
    parser = argparse.ArgumentParser(
        description="analyze SegNet output vs ground truth using \
            global accuracy, mean class accuracy, mIoU")
    parser.add_argument("--width", default=WIDTH,
                        help="desired width for images")
    parser.add_argument("--height", default=HEIGHT,
                        help="desired height for images")
    parser.add_argument("-iim", "--in_img_dir", default=IMG_PATH[0],
                        help="input path for camera images (default is {})"
                        .format(IMG_PATH[0]))
    parser.add_argument("-igt", "--in_gt_dir", default=IMG_PATH[1],
                        help="input path for ground truth (default is {})"
                        .format(IMG_PATH[1]))
    parser.add_argument("-o", "--out_dir", default=OUT_PATH,
                        help="output path (default is {})"
                        .format(OUT_PATH))
    parser.add_argument("-d", "--data_type", default=DATA_TYPE,
                        help="type of data being handled (test/val/train)")
    parser.add_argument("-nr", "--noresize", action="store_false",
                        help="flag set skips images resizing")
    parser.add_argument("-ctar", "--crop_to_aspect_ratio", default=CROP_TO_ASPECT_RATIO,
                        help="flag set crops input image to the aspect ratio of specified height x width and then resizes")
    parser.add_argument("-ch", "--crop_height_position", default=CROP_HEIGHT_POSITION, choices=['top', 'middle', 'bottom'],
                        help="position to crop on height from if crops_to_aspect_ratio is set")
    parser.add_argument("-cw", "--crop_width_position", default=CROP_WIDTH_POSITION, choices=['left', 'middle', 'right'],
                        help="position to crop on width from if crops_to_aspect_ratio is set")

    ret_args = parser.parse_args(arg_vals)

    WIDTH = int(ret_args.width) - int(ret_args.width) % 32
    HEIGHT = int(ret_args.height) - int(ret_args.height) % 32
    IMG_PATH[0] = os.path.expanduser(ret_args.in_img_dir)
    IMG_PATH[1] = os.path.expanduser(ret_args.in_gt_dir)
    OUT_PATH = ret_args.out_dir
    DATA_TYPE = ret_args.data_type
    RESIZE_IMGS = ret_args.noresize
    if not TXT_PATH:
        TXT_PATH = "{}{}.txt".format(OUT_PATH, DATA_TYPE)

    print ("width set to: {}; height set to: {}".format(WIDTH, HEIGHT))
    return ret_args


def crop_image(img, img_dim, position):
    """
    Crops the image based on the desired dimension and position to start cropping
    :param img: openCV image
    :param img_dim: (desired_height, desired_width)
    :param position: (crop_height_position, crop_width_position)
    :return:
    """
    if position[0] == "top":
        h_t = img.shape[0] - img_dim[0]
        h_b = None
    elif position[0] == "middle":
        h_t = int((img.shape[0] - img_dim[0]) / 2)
        h_b = h_t + img_dim[0]
    elif position[0] == "bottom":
        h_t = None
        h_b = img_dim[0]
    else:
        raise ValueError("Invalid position, expecting (top, middle, bottom)")

    if position[1] == "left":
        w_l = img.shape[1] - img_dim[1]
        w_r = None
    elif position[1] == "middle":
        w_l = int((img.shape[1] - img_dim[1]) / 2)
        w_r = w_l + img_dim[1]
    elif position[1] == "right":
        w_l = None
        w_r = img_dim[1]
    else:
        raise ValueError("Invalid position, expecting (left, middle, right)")

    return img[h_t:h_b, w_l:w_r]


def main():
    get_args()

    # get lists of files
    images = [sorted(glob(IMG_PATH[0])), sorted(glob(IMG_PATH[1]))]

    # check for folders' integrity
    assert len(images[0]) > 0, "no images in: %s" % (IMG_PATH[0])
    assert len(images[1]) > 0, "no images in: %s" % (IMG_PATH[1])
    assert len(images[0]) == len(images[1]), "number of images mismatch"

    # determine image size
    src_shape = cv2.imread(images[0][0]).shape
    """
    for image in np.array(images).flatten():
        assert cv2.imread(image).shape == src_shape, \
            "size mismatch: %s" % image
    """
    assert preprocessor(src_shape, zip(images[0], images[1])), \
        "images not resized"


def preprocessor(src_shape, img_pairs):
    # verify output directories
    out_path = [os.path.join(OUT_PATH, "test/"),
                os.path.join(OUT_PATH, "testannot/")]
    if DATA_TYPE == "train":
        out_path = [os.path.join(OUT_PATH, "train/"),
                    os.path.join(OUT_PATH, "trainannot/")]
    elif DATA_TYPE == "val":
        out_path = [os.path.join(OUT_PATH, "val/"),
                    os.path.join(OUT_PATH, "valannot/")]
    for i in range(2):
        assert os.path.isdir(out_path[i]), \
            "non-existent output directory {}".format(out_path[i])

    dst_shape = (HEIGHT, WIDTH, 3)
    resize_imgs = RESIZE_IMGS and src_shape != dst_shape
    if CROP_TO_ASPECT_RATIO:
        height_multiple = int(src_shape[0] / dst_shape[0])
        width_multiple = int(src_shape[1] / dst_shape[1])
        min_multiple = min(height_multiple, width_multiple)
        height_crop_dim = dst_shape[0] * min_multiple
        width_crop_dim = dst_shape[1] * min_multiple

    count, freqs, class_counts = 0, np.zeros(256), np.zeros(256, np.int)

    with open(TXT_PATH, "w") as test_file:
        print TXT_PATH
        for image0, image1 in img_pairs:
            path = ["", ""]

            for i, image in enumerate([image0, image1]):
                if i == 0:
                    img = cv2.imread(image)
                else:
                    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

                if CROP_TO_ASPECT_RATIO:
                    img = crop_image(img, (height_crop_dim, width_crop_dim), (CROP_HEIGHT_POSITION, CROP_WIDTH_POSITION))

                if resize_imgs:
                    img = cv2.resize(img, (WIDTH, HEIGHT),
                                            interpolation=INTERPOLATION[i])

                if i == 1 and DATA_TYPE == "train":
                    freq = np.reshape(cv2.calcHist(
                        [img], [0], None, [256], [0, 256]), 256)
                    img_classes = np.nonzero(freq)
                    freqs[img_classes] += freq[img_classes]
                    class_counts[img_classes] += 1

                path[i] = out_path[i] + os.path.basename(image)
                cv2.imwrite(path[i], img)
            # write test.txt for SegNet
            test_file.write("%s %s\n" % (path[0], path[1]))
            count += 1
            print("%d: %s %s" % (count, path[0], path[1]))

    if DATA_TYPE == "train":
        n_classes = np.nonzero(class_counts)
        freqs = freqs[n_classes] / class_counts[n_classes]
        freqs = np.median(freqs) / freqs
        print("softmax class weights:\n{}".format(freqs))

    return resize_imgs


if __name__ == "__main__":
    try:
        main()
    except AssertionError as err:
        print(err.args[0])
