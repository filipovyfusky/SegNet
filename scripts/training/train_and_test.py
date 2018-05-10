import os
import sys
import random
import argparse
import configparser
from multiprocessing import Process
import pyinotify
sys.path.insert(0, os.path.expanduser("/PATH/TO/SegNet/caffe-segnet-cudnn7/python")) # Might not need this if you add to $PATH from ~/.bashrc
sys.path.insert(0, os.path.expanduser("/PATH/TO/SegNet/custom_layers"))
sys.path.insert(0, os.path.expanduser("/PATH/TO/SegNet/scripts"))

import caffe
import numpy as np
from compute_bn_statistics import compute_bn_statistics
from caffe.proto import caffe_pb2
from google.protobuf import text_format

from inference import bayesian_segnet_inference


class SnapshotEventHandler(pyinotify.ProcessEvent):
    """
    Handles snapshots from Bayesian SegNet. Once a new .caffemodel is created, The batch norm for the
    snapshot is calculated. Once completed, a random subsample of validation images are run through the
    network and the results are saved to the specified log_dir
    """
    def __init__(self, training_model, test_model, test_image_file, log_dir, gpu, test_shape):
        self.training_model = training_model
        self.test_model = test_model
        self.test_image_locs = []
        with open(test_image_file, "r") as text_file:
            for line in text_file:
                # Text file from cityscapes script has two columns in each row, one for the image and one for
                # the annotated one. line.split(" ")[0] takes the non-annotaed image
                self.test_image_locs.append(line.split(" ")[0])
        self.log_dir = log_dir
        self.gpu = gpu
        self.test_shape = test_shape
        # TODO(jskhu): Remove hardcoded amount of images to test on
        # TODO(jskhu): Add assertion that num_test_images is less than num of images in text file
        self.num_test_images = 10
        caffe.set_device(self.gpu)
        caffe.set_mode_gpu()

        super(SnapshotEventHandler, self).__init__()

    def process_IN_CREATE(self, event):
        # Only triggers on .caffemodel
        # Checking _inf.caffemodel stops infinite recursion on created caffemodels
        if not event.pathname.endswith('.caffemodel') or event.pathname.endswith('_inf.caffemodel'):
            return

        print("Snapshot created!")

        train_weight_path = event.pathname
        inf_weight_path = os.path.splitext(train_weight_path)[0] + '_inf.caffemodel'

        compute_bn_statistics(self.training_model, train_weight_path, inf_weight_path, self.test_shape)
        weights_name = os.path.splitext(os.path.basename(event.pathname))[0]

        log_file = file('{}/{}.log'.format(self.log_dir, weights_name), "w+")
        image_prefix = '{}/{}_image'.format(self.log_dir, weights_name)
        temp_stdout = sys.stdout
        sys.stdout = log_file
        print 'Testing: {}'.format(event.pathname)
        # TODO(jskhu): Change this to be generic. Currently only works for Bayesian SegNet
        random.shuffle(self.test_image_locs)
        image_locs = random.sample(self.test_image_locs, self.num_test_images)
        bayesian_segnet_inference.save_images_inferences(image_locs, self.test_model, inf_weight_path, image_prefix, self.gpu)

        sys.stdout = temp_stdout


def wait_and_test_snapshots(snapshot_dir, training_model, test_model, test_image_file, log_dir, gpu, test_shape):
    wm = pyinotify.WatchManager()
    handler = SnapshotEventHandler(training_model, test_model, test_image_file, log_dir, gpu, test_shape)
    notifier = pyinotify.Notifier(wm, handler)
    mask = pyinotify.IN_CREATE
    # rec=False stops recursive check to nested folder. ONLY checks changes in snapshot_dir
    wdd = wm.add_watch(snapshot_dir, mask, rec=False)
    notifier.loop()


def get_args():
    """
    parse any relevant runtime arguments
    param arg_vals:    args to check (None defaults to command-line args)
    return:            gpu id, solvers list, and weights list
    """

    parser = argparse.ArgumentParser(
        description="Initiate training for SegNet")
    parser.add_argument("--train_gpu", type=int, default = 0,
                        help="GPU ID for training (default is 0)")
    parser.add_argument("--test_gpu", type=int, default = 0,
                        help="GPU ID for testing (default is 0)")
    parser.add_argument("--run_inference", default=True,
                        help="Turn on/off whether to run inferences on intermediate caffe models")
    parser.add_argument("--config", default = "train_config.ini" ,
                        help="file path to train_config.ini file that lists solvers and weights to use")

    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    # load configuration parameters into lists
    solvers = [os.path.expanduser(solver) for solver in config["Solvers"].values()]
    init_weights = [os.path.expanduser(weight) for weight in config["Init_Weights"].values()]
    inf_weights = [os.path.expanduser(weight) for weight in config['Inference_Weights'].values()]
    solverstates = [os.path.expanduser(solverstate) for solverstate in config["Solverstates"].values()]
    test_models = [os.path.expanduser(test_model) for test_model in config["Test_Models"].values()]
    test_images = [os.path.expanduser(test_image) for test_image in config["Test_Images"].values()]
    log_dirs = [os.path.expanduser(log_dir) for log_dir in config["Log_Dirs"].values()]
    test_shape = [os.path.expanduser(test_shape) for test_shape in config["Test_Shape"].values()]
    
    # Convert test_shape to a list of integers
    test_shape = [int(x) for x in test_shape[0].split(",")]

    # verify parameters are correct
    for solver, init_weight, solverstate, test_model, test_image, log_dir in zip(solvers, init_weights, solverstates, test_models, test_images, log_dirs):
        assert os.path.exists(solver), "Cannot find {}".format(solver)

        # verify solver parameters are correct
        solver_config = caffe_pb2.SolverParameter()
        with open(solver) as proto:
            text_format.Merge(str(proto.read()), solver_config)
        assert os.path.exists(solver_config.net), "Invalid path: {}".format(solver_config.net)
        assert os.path.exists(os.path.dirname(solver_config.snapshot_prefix)), "Invalid path: {}".format(os.path.dirname(solver_config.snapshot_prefix))
        assert os.path.exists(solverstate) or not solverstate, "Invalid path: {}".format(solverstate)
        assert os.path.exists(test_model) or not test_model, "Invalid path: {}".format(test_model)
        assert os.path.exists(test_image) or not test_image, "Invalid path: {}".format(test_image)
        assert os.path.exists(log_dir) or not log_dir, "Invalid path: {}".format(log_dir)

        # verify train net and val net prototxt parameters are correct
        assert os.path.exists(init_weight) or init_weight in inf_weights or not init_weight, "Invalid path: {}".format(init_weight)

    assert len(solvers) == len(init_weights), "number of solver and initialization weight files mismatch"
    assert len(solvers) == len(inf_weights), "number of solver and inference weight paths mismatch"
    assert len(solvers) == len(solverstates), "number of solver and solverstates mismatch"
    assert len(solvers) == len(test_models), "number of solver and test_models mismatch"
    assert len(solvers) == len(test_images), "number of solver and test_images mismatch"
    assert len(solvers) == len(log_dirs), "number of solver and log_dirs mismatch"
    assert len(test_shape) == 4, "test_shape must have a shape of 4"

    return (args.run_inference, args.train_gpu, args.test_gpu, zip(solvers, init_weights, inf_weights, solverstates, test_models, test_images, log_dirs), test_shape)


def train_network(gpu, train_path, test_shape):
    caffe.set_device(gpu)
    caffe.set_mode_gpu()
    proto, init_weight_path, inf_weight_path, solverstate, test_model, test_image, log_dir = train_path
    # Get solver parameters
    solver_config = caffe_pb2.SolverParameter()

    # TODO(jskhu): Set log file
    with open(proto) as solver:
        text_format.Merge(str(solver.read()), solver_config)
    train_weight_path = "{}_iter_{}.caffemodel".format(solver_config.snapshot_prefix, solver_config.max_iter)

    # If the testing is run in parallel with training, a seperate process will be made which will run everytime
    # a new snapshot is created

    solver = caffe.SGDSolver(str(proto))

    # initialize weights if given
    if solverstate:
        solver.restore(str(solverstate))
    elif init_weight_path:
        solver.net.copy_from(str(init_weight_path))

    # Train until completion, remove solver to free up GPU, and compute mean and variance for the batch norm layers
    solver.solve()

    print "Training complete"

    compute_bn_statistics(solver_config.net, train_weight_path, inf_weight_path, test_shape)
    del solver
    # TODO(jskhu): Run entire validation set on trained network


if __name__ == "__main__":
    run_inference, train_gpu, test_gpu, train_paths, test_shape = get_args()
    for train_path in train_paths:
        print train_path
        proto, init_weight_path, inf_weight_path, solverstate, test_model, test_images, log_dir = train_path
        print(test_shape)

        train_process = Process(target=train_network, args=(train_gpu,
                                                            train_path,
                                                            test_shape))
        if run_inference:
            solver_config = caffe_pb2.SolverParameter()
            with open(proto) as solver:
                text_format.Merge(str(solver.read()), solver_config)
            snapshot_dir = '/'.join(solver_config.snapshot_prefix.split('/')[0:-1])
            test_snapshot_process = Process(target=wait_and_test_snapshots,
                                            args=(snapshot_dir,
                                                  solver_config.net,
                                                  test_model,
                                                  test_images,
                                                  log_dir,
                                                  test_gpu,
                                                  test_shape))
            test_snapshot_process.start()

        train_process.start()
        train_process.join()
        # TODO(jskhu): Currently testing process dies as soon as training is completed. Find a way to either wait
        # for the testing to complete, or gracefully shut it down
        if run_inference:
            test_snapshot_process.terminate()

