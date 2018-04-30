#!/usr/bin/env python
import os
import sys
import argparse
import configparser
import sys
sys.path.insert(0, os.path.expanduser("~/wave/caffe-segnet-cudnn7/python")) # Might not need this if you add to $PATH from ~/.bashrc
sys.path.insert(0, os.path.expanduser("~/wave/SegNet/custom_layers"))

import caffe
import numpy as np
from compute_bn_statistics import compute_bn_statistics
from caffe.proto import caffe_pb2
from google.protobuf import text_format

def get_args():
    """
    parse any relevant runtime arguments
    param arg_vals:    args to check (None defaults to command-line args)
    return:            gpu id, solvers list, and weights list
    """

    parser = argparse.ArgumentParser(
        description="Initiate training for SegNet")
    parser.add_argument("--gpu", type=int, default = 0,
        help="GPU ID for training (default is 0)")
    parser.add_argument("--config", default = "train_config.ini" ,
        help="file path to train_config.ini file that lists solvers and weights to use")

    args= parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    # load configuration parameters into lists
    solvers = [os.path.expanduser(solver) for solver in config["Solvers"].values()]
    init_weights = [os.path.expanduser(weight) for weight in config["Init_Weights"].values()]
    inf_weights = [os.path.expanduser(weight) for weight in config['Inference_Weights'].values()]
    solverstates = [os.path.expanduser(solverstate) for solverstate in config["Solverstates"].values()]

     # verify parameters are correct
    for solver, init_weight, solverstate in zip(solvers, init_weights, solverstates):
        assert os.path.exists(solver), "Cannot find {}".format(solver)

        # verify solver parameters are correct
        solver_config = caffe_pb2.SolverParameter()
        with open(solver) as proto:
            text_format.Merge(str(proto.read()), solver_config)
        assert os.path.exists(solver_config.net), "Invalid path: {}".format(solver_config.net)
        assert os.path.exists(os.path.dirname(solver_config.snapshot_prefix)), "Invalid path: {}".format(os.path.dirname(solver_config.snapshot_prefix))
        assert os.path.exists(solverstate) or not solverstate, "Invalid path: {}".format(solverstate)

        # verify train net and val net prototxt parameters are correct
        verify_model_params(solver_config.net, "train")
        assert os.path.exists(init_weight) or init_weight in inf_weights or not init_weight, "Invalid path: {}".format(init_weight)

    assert len(solvers) == len(init_weights), "number of solver and initialization weight files mismatch"
    assert len(solvers) == len(inf_weights), "number of solver and inference weight paths mismatch"
    assert len(solvers) == len(solverstates), "number of solver and solverstates mismatch"

    return (args.gpu, zip(solvers, init_weights, inf_weights, solverstates))

def verify_model_params(model, split):
    net = caffe_pb2.NetParameter()
    with open(model) as proto:
        text_format.Merge(str(proto.read()), net)

    params = eval(net.layer[0].python_param.param_str)
    data_dirs = [os.path.expanduser(data_dir) for data_dir in params['data_dirs'].split(",")]
    data_proportions = params['data_proportions']

    for data_dir, data_proportion in zip(data_dirs, data_proportions):
        assert os.path.exists("{}/{}.txt".format(data_dir, split)), "Invalid path:{}/{}.txt".format(data_dir, split)
        assert data_proportion >= 0 and data_proportion <= 1, "Invalid data proportion: {} (Accepted range: 0-1)"

    assert len(data_proportions) == len(data_dirs), "Invalid number of data proportions"
    assert params['batch_size'], "Batch size not specified"
    assert params['split'], "Data type (split) not specfied"

def train(gpu, train_paths):

    # init
    caffe.set_device(gpu)
    caffe.set_mode_gpu()

    for proto, init_weight_path, inf_weight_path, solverstate in train_paths:
        # Get solver parameters
        solver_config = caffe_pb2.SolverParameter()
        with open(proto) as solver:
            text_format.Merge(str(solver.read()), solver_config)
        train_weight_path = "{}_iter_{}.caffemodel".format(solver_config.snapshot_prefix, solver_config.max_iter)

        solver = caffe.SGDSolver(str(proto))

        # initialize weights if given
        if solverstate:
            solver.restore(solverstate)
        elif init_weight_path:
            solver.net.copy_from(init_weight_path)

        # Train until completion, remove solver to free up GPU, and compute mean and variance for the batch norm layers
        solver.solve()
        del solver
        compute_bn_statistics(solver_config.net, train_weight_path, inf_weight_path)


if __name__ == "__main__":
    gpu, train_paths = get_args()
    train(gpu, train_paths)
