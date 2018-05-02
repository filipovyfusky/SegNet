import os
import random

import caffe
import numpy as np
import cv2


class SegNetSegDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from a general dataset
    one-at-a-time while reshaping the net to preserve dimensions.
    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:
        - data_dirs: path to dataset directories
        - data_proportions: proportions of datasets
        - batch_size: number of images per batch
        - split: train / val / test
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)
        for semantic segmentation.
        example: params = dict(data_dir="/path/to/Dataset", split="val")
        """
        # config
        params = eval(self.param_str)
        self.data_dirs = [os.path.expanduser(data_dir) for data_dir in params['data_dirs'].split(",")]
        self.data_proportions = params['data_proportions']
        self.batch_size = params['batch_size']
        self.split = params['split']
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.dataset_idx = 0
        self.datasets = []

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # load image and label paths
        for data_dir, proportion in zip(self.data_dirs, self.data_proportions):
            with open("{}/{}.txt".format(data_dir, self.split), "r") as paths:
                lines = paths.read().splitlines()
                num_lines = int(round(proportion * len(lines)))

                # randomization: seed and randomly sample, otherwise sample in order
                if self.random:
                    random.seed(self.seed)
                    lines = random.sample(lines, num_lines)
                else:
                    lines = lines[:num_lines]
                self.datasets.append([[data_path for data_path in line.split()] for line in lines])

        self.idxs = [0] * len(self.datasets)

    def reshape(self, bottom, top):
        # load image + label image pair
        idx = self.idxs[self.dataset_idx]
        self.data = self.load_image(self.datasets[self.dataset_idx][idx][0])
        self.label = self.load_label(self.datasets[self.dataset_idx][idx][1])

        # reshape tops to fit
        top[0].reshape(self.batch_size, *self.data.shape)
        top[1].reshape(self.batch_size, *self.label.shape)

    def forward(self, bottom, top):
        for itt in range(self.batch_size):
            # assign output
            idx = self.idxs[self.dataset_idx]
            top[0].data[itt, ...] = self.load_image(self.datasets[self.dataset_idx][idx][0])
            top[1].data[itt, ...] = self.load_label(self.datasets[self.dataset_idx][idx][1])

            # pick next input
            self.idxs[self.dataset_idx] += 1
            if self.idxs[self.dataset_idx] == len(self.datasets[self.dataset_idx]):
                self.idxs[self.dataset_idx] = 0
            self.dataset_idx += 1
            if self.dataset_idx == len(self.datasets):
                self.dataset_idx = 0

    def backward(self, top, propagate_down, bottom):
        pass

    def load_image(self, path):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - transpose to channel x height x width order
        """
        im = cv2.imread(path)
        in_ = np.array(im, dtype=np.float32)
        in_ = in_.transpose((2,0,1))
        return in_

    def load_label(self, path):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """

        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        label = np.array(im, dtype=np.float32).reshape((1, im.shape[0], im.shape[1]))
        return label
