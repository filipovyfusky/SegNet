#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
After SegNet has been trained, run compute_bn_statistics.py script and then batch_norm_absorber.py.

For inference batch normalization layer can be merged into convolutional kernels, to
speed up the network. Both layers applies a linear transformation. For that reason
the batch normalization layer can be absorbed in the previous convolutional layer
by modifying its weights and biases. That is exactly what the script does.
"""
import os
import numpy as np
from argparse import ArgumentParser
from os.path import join
import argparse
import sys
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format


__author__ = 'Timo Sämann'
__university__ = 'Aschaffenburg University of Applied Sciences'
__email__ = 'Timo.Saemann@gmx.de'
__data__ = '6th May, 2017'


def copy_double(data):
    return np.array(data, copy=True, dtype=np.double)


def bn_absorber_weights(model, weights):

    # load the prototxt file as a protobuf message
    with open(model) as f:
        str2 = f.read()
    msg = caffe_pb2.NetParameter()
    text_format.Merge(str2, msg)

    # load net
    net = caffe.Net(model, weights, caffe.TEST)

    # iterate over all layers of the network
    for i, layer in enumerate(msg.layer):

        # check if conv layer exist right before bn layer, otherwise merging
        #  is not possible and skip
        if not layer.type == 'BN':
            continue
        if not msg.layer[i-1].type == 'Convolution':
            continue

        # get the name of the bn and conv layer
        bn_layer = msg.layer[i].name
        conv_layer = msg.layer[i-1].name

        # get some necessary sizes
        kernel_size = 1
        shape_of_kernel_blob = net.params[conv_layer][0].data.shape
        number_of_feature_maps = list(shape_of_kernel_blob[0:1])
        shape_of_kernel_blob = list(shape_of_kernel_blob[1:4])
        for x in shape_of_kernel_blob:
            kernel_size *= x

        weight = copy_double(net.params[conv_layer][0].data)
        bias = copy_double(net.params[conv_layer][1].data)

        # receive new_gamma and new_beta which was already calculated by the
        # compute_bn_statistics.py script
        new_gamma = net.params[bn_layer][0].data[...]
        new_beta = net.params[bn_layer][1].data[...]

        # manipulate the weights and biases over all feature maps:
        # weight_new = weight * gamma_new
        # bias_new = bias * gamma_new + beta_new
        # for more information see https://github.com/alexgkendall/caffe-segnet/issues/109
        for j in xrange(number_of_feature_maps[0]):

            net.params[conv_layer][0].data[j] = weight[j] * np.repeat(new_gamma.item(j), kernel_size).reshape(
                net.params[conv_layer][0].data[j].shape)
            net.params[conv_layer][1].data[j] = bias[j] * new_gamma.item(j) + new_beta.item(j)

        # set the no longer needed bn params to zero
        net.params[bn_layer][0].data[:] = 0
        net.params[bn_layer][1].data[:] = 0

    return net


def bn_absorber_prototxt(model):

    # load the prototxt file as a protobuf message
    with open(model) as k:
        str1 = k.read()
    msg1 = caffe_pb2.NetParameter()
    text_format.Merge(str1, msg1)

    # search for bn layer and remove them
    for l in msg1.layer:
        if l.type == "BN":
            msg1.layer.remove(l)

    return msg1


def make_parser():
    parser = argparse.ArgumentParser("Script to merge together batch"
                                     "normalization layers with preceding "
                                     "convolution layers in order to "
                                     "speed up inference.")
    parser.add_argument('model',
                        type=str,
                        help="The model description to use for inference "
                             "(.prototxt file)")
    parser.add_argument('weights',
                        type=str,
                        help="The weights (.caffemodel file) in which the "
                             "batch normalization and convolution layers are"
                             "to be merged.")
    parser.add_argument('out_dir',
                        type=str,
                        help="Output directory to store the merged .prototxt"
                             "and .caffemodel files.")
    parser.add_argument('--cpu',
                        action='store_true',
                        default=False,
                        help="Flag to indicate whether or not to use CPU for "
                             "computation. If not set, will use GPU.")
    return parser

if __name__ == '__main__':
    parser1 = make_parser()
    args = parser1.parse_args()
    
    if args.cpu:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()

    # Check if output directory exists, create if not
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # Merge bn layer into conv kernel
    network = bn_absorber_weights(args.model, args.weights)

    # Remove bn layer from prototxt file
    msg_proto = bn_absorber_prototxt(args.model)

    # Save prototxt for inference
    print "Saving inference prototxt file..."
    path = os.path.join(args.out_dir, "bn_conv_merged_model.prototxt")
    with open(path, 'w') as m:
        m.write(text_format.MessageToString(msg_proto))

    # Save weights
    print "Saving new weights..."
    network.save(os.path.join(args.out_dir,
                              "bn_conv_merged_weights.caffemodel"))
    print "Done!"
