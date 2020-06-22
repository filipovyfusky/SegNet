import json
import numpy
from PIL import Image
from shutil import copyfile
import urllib.request
from math import *
from create_prototxt import make_prototxt
import cv2
import time

caffe_root = '/path/to/caffe-segnet/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

# Use colours to visualize classes
LABEL_COLOURS = numpy.zeros([256, 3])
LABEL_COLOURS[0] = [128, 69, 69]
LABEL_COLOURS[1] = [152, 251, 152]
# Enter total number of classes
NUM_CLASSES = 2

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

def iou(model, weights, input_source, input_source_label):
    iou_list = []
    timer = []
    for i in range(NUM_CLASSES):
        iou_list.append([])
    caffe.set_mode_gpu()
    # Load Caffe network
    net = caffe.Net(model, weights, caffe.TEST)

    # Access blob data
    input_shape = net.blobs['data'].data.shape
    confidence_output = net.blobs['prob'].data
    cap = cv2.VideoCapture(input_source)
    cap_label = cv2.VideoCapture(input_source_label)

    rval = True
    while rval:
        # Get image from VideoCapture
        rval, frame = cap.read()
        rval_lab, frame_lab = cap_label.read()
        if not rval:
            print("No image found!")
            break

        # Resize input image
        resized_image = crop_input(frame, (input_shape[3], input_shape[2]))
        cropped = numpy.int32(resized_image)
        # Subtract per-channel mean
        B_mean = 129
        G_mean = 126
        R_mean = 126
        cropped[:, :, 0] -= R_mean
        cropped[:, :, 1] -= G_mean
        cropped[:, :, 2] -= B_mean
        # Input shape is (y, x, 3), needs to be reshaped to (3, y, x)
        input_image = cropped.transpose((2, 0, 1))
        # Repeat image according to batch size for inference.
        MCDO_samples = input_shape[0]
        input_image = numpy.repeat(input_image[numpy.newaxis, :, :, :],
                                MCDO_samples,
                                axis=0)
        # Inference using Bayesian SegNet
        start = time.time()
        out = net.forward_all(data=input_image)
        end = time.time()
        timer.append(end-start)

        # By Alex Kendall
        mean_confidence = numpy.mean(confidence_output, axis=0, dtype=numpy.float64)
        var_confidence = numpy.var(confidence_output, axis=0, dtype=numpy.float64)
        # Prepare segmented image results
        classes = numpy.argmax(mean_confidence, axis=0)

        # Calculae IOU CLASS 1
        frame_lab = frame_lab[:,:,0]

        for i in range(NUM_CLASSES):
            boolean_frame = numpy.int32((frame_lab == i))
            boolean_classes = numpy.int32((classes == i))
            union = boolean_frame | boolean_classes
            intersection = boolean_frame & boolean_classes
            iou = numpy.sum(intersection.flatten()) / numpy.sum(union.flatten())
            print(round(iou,3))
            iou_list[i].append(iou)

    miou = []
    for i in range(NUM_CLASSES):
        miou.append(numpy.round(numpy.mean(iou_list[i]),3))

    cap.release()
    cv2.destroyAllWindows()
    print("NAZDAR")
    mtimer = numpy.mean(timer)
    return miou, mtimer

def download(file, base_path):
    """
    Downloads images and labels from Labelbox using the exported JSON file.
    """
    with open(file) as json_file:
        datas = json.load(json_file)
        pom_save = 0
        pom_read = 0
        i = 0
        # -------Find corrupted items in JSON---------
        try:
            while True:
                if not "objects" in datas[i]["Label"]:
                    del datas[i]
                else:
                    i += 1
        except IndexError:
            print('Konec seznamu')
            print(i)
        # -------Go through URLs and perform file transfer---------
        for dat in datas:
            success = False
            while not success:
                try:
                    print(dat)
                    urllib.request.urlretrieve(dat["Labeled Data"], base_path + r"/imgs/img_" + str(pom_save) + ".jpg")
                    urllib.request.urlretrieve(dat["Label"]["objects"][0]["instanceURI"],
                                               base_path + r"/labs/lab_" + str(pom_save) + ".png")
                except IndexError as e:
                    if not dat["Label"]["objects"]:
                        print("Index {} is empty".format(pom_read))
                    success = True
                except KeyError as e:
                    print("Index {} is corrupted".format(pom_read))
                    success = True
                except urllib.error.HTTPError as e:
                    print('NASTAL ERROR HTTP')
                else:
                    print(str(pom_save))
                    print(str(pom_read))
                    pom_save += 1
                    success = True
            pom_read += 1


def convert(base_path):
    """
    Convert labels to the right format for SegNet (labels from labelbox have range 0-255,
    SegNet requires values 0,1,2....NUM_CLASSES-1)
    """
    # PREDELAT POM A POM GLOB < NEPREHLEDNE
    pom_read = 0
    pom_save = 0
    ban = []

    path = base_path + r"/labs"
    path2 = base_path + r"/labs_single"
    # ====Go through labels and skip corrupted ones====
    print("Converting...")
    while (True):
        try:
            img = Image.open(path + r"/lab_" + str(pom_read) + ".png").convert("L")
            imgarr = numpy.array(img)
            pix = numpy.unique(imgarr)
            # Labels can contain only 0 to NUM_CLASSES-1 pixels!
            if (numpy.amax(imgarr) == 0) or (pix.size != NUM_CLASSES):
                print("Chyba v " + str(pom_read) + " size_pix = " + str(pix.size))
                ban.append(pom_read)
                pom_read += 1
                continue
            #imgarr = imgarr * ((NUM_CLASSES-1)/numpy.amax(imgarr))
            # Replace pixel values
            for i in range(NUM_CLASSES):
                imgarr[imgarr==pix[i]] = i
            im = Image.fromarray(numpy.uint8(imgarr))
            im.save(path2 + r"/lab_" + str(pom_save) + ".png", "PNG")
            pom_save += 1
            pom_read += 1
        except FileNotFoundError:
            print("Ran out of images, currently: {}".format(str(pom_read)))
            break
        except OSError:
            print("Cannot identify image file: {}".format(str(pom_read)))
            break
        else:
            #print(pom_save)
            pass
    # Show skipped images and pass to another function
    print(ban)
    make_dirs(ban, base_path)

def make_dirs(ban, base_path):
    """
    Take images and put them into train, val and test folders,
    then create file with image paths (for SegNet, CamVid format)
    """
    print("Making dirs")
    # Specify number of training and validation samples
    NUM_TRAIN = 0
    NUM_VAL = 0
    NUM_TEST = 179
    # Aux variables
    pom_img = 0
    pom_lab = 0

    # ------Create TRAINING images and labels--------
    if NUM_TRAIN != 0:
        pom_save = 0
        while True:
            if pom_img in ban:
                print(pom_img)
                pom_img += 1
                continue
            copyfile(base_path + r"/imgs/img_" + str(pom_img) + ".jpg", base_path + r"/train/img_" + str(pom_save) + ".jpg")
            copyfile(base_path + r"/labs_single/lab_" + str(pom_lab) + ".png",
                     base_path + r"/trainannot/lab_" + str(pom_save) + ".png")
            pom_img += 1
            pom_lab += 1
            pom_save += 1
            if pom_save == NUM_TRAIN:
                break
    # ------Create VALIDATION images and labels--------
    if NUM_VAL != 0:
        pom_save = 0
        while True:
            if pom_img in ban:
                print(pom_img)
                pom_img += 1
                continue
            copyfile(base_path + r"/imgs/img_" + str(pom_img) + ".jpg", base_path + r"/val/img_" + str(pom_save) + ".jpg")
            copyfile(base_path + r"/labs_single/lab_" + str(pom_lab) + ".png",
                     base_path + r"/valannot/lab_" + str(pom_save) + ".png")
            pom_img += 1
            pom_lab += 1
            pom_save += 1
            if pom_save == NUM_VAL:
                break
    # ------Create TEST images and labels--------
    if NUM_TEST != 0:
        pom_save = 0
        while True:
            if pom_img in ban:
                print(pom_img)
                pom_img += 1
                continue
            copyfile(base_path + r"/imgs/img_" + str(pom_img) + ".jpg", base_path + r"/test/img_" + str(pom_save) + ".jpg")
            copyfile(base_path + r"/labs_single/lab_" + str(pom_lab) + ".png",
                     base_path + r"/testannot/lab_" + str(pom_save) + ".png")
            pom_img += 1
            pom_lab += 1
            pom_save += 1
            if pom_save == NUM_TEST:
                break

    # ------Write TRAIN file paths (image+label pairs)--------
    if NUM_TRAIN != 0:
        with open("train_linux.txt", "w") as train_file:
            pom_save = 0
            while(True):
                source = base_path + r"/train/img_" + str(pom_save) + ".jpg"
                label = base_path + r"/trainannot/lab_" + str(pom_save) + ".png"
                print(source + " " + label, file=train_file)
                pom_save += 1
                if pom_save == NUM_TRAIN:
                    break
    # ------Write VALIDATION file paths (image+label pairs)--------
    if NUM_VAL != 0:
        with open("val_linux.txt", "w") as val_file:
            pom_save = 0
            while(True):
                if NUM_VAL==0:
                    break
                source = base_path + r"/val/img_" + str(pom_save) + ".jpg"
                label = base_path + r"/valannot/lab_" + str(pom_save) + ".png"
                print(source + " " + label, file=val_file)
                pom_save += 1
                if pom_save == NUM_VAL:
                    break
    # ------Write TEST file paths (image+label pairs)--------
    if NUM_TEST != 0:
        with open("test_linux.txt", "w") as test_file:
            pom_save = 0
            while (True):
                source = base_path + r"/test/img_" + str(pom_save) + ".jpg"
                label = base_path + r"/testannot/lab_" + str(pom_save) + ".png"
                print(source + " " + label, file=test_file)
                pom_save += 1
                if pom_save == NUM_TEST:
                    break

def per_channel_mean():
    """
    Compute per channel mean
    """
    B_mean = []
    G_mean = []
    R_mean = []
    pom_read = 0
    path_base = r"/media/phil/SegNet/data/custom/train"
    # ====Go through labels and skip corrupted ones (labelbox sometimes exports bad labels...)====
    print("Pocitam per channel prumer")
    while (True):
        try:
            img = Image.open(path_base + r"/img_" + str(pom_read) + ".jpg")
            # Flip horizontally
            imgarr = numpy.array(img)
            B_mean.append(numpy.uint8(numpy.mean(imgarr[:, :, 2])))
            G_mean.append(numpy.uint8(numpy.mean(imgarr[:, :, 1])))
            R_mean.append(numpy.uint8(numpy.mean(imgarr[:, :, 0])))
            pom_read += 1
        except FileNotFoundError as e:
            print("Ran out of images, currently: {}".format(str(pom_read)))
            break
        except OSError:
            print("Cannot identify image file: {}".format(str(pom_read)))
            break
        else:
            #print(pom_save)
            pass
    B = numpy.uint8(numpy.mean(B_mean))
    G = numpy.uint8(numpy.mean(G_mean))
    R = numpy.uint8(numpy.mean(R_mean))
    print(B,G,R)