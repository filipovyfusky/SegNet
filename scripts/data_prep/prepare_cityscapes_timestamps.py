#!/usr/bin/env python

import argparse
import numpy as np
import time
import os
import glob


def get_args():
    parser = argparse.ArgumentParser(description="Consolidates the Cityscapes "
                                                 "timestamps for each image "
                                                 "into the format required for "
                                                 "ORBSLAM.")
    parser.add_argument("path",
                        type=str,
                        help="Path to folder containing Cityscapes timestamps.")
    parser.add_argument("out_path",
                        type=str,
                        help="Output path for timestamps file")

    args = parser.parse_args()

    return args.path, args.out_path


def extract_timestamps(path):
    if path.endswith("/"):
        timestamps_path = path + "*.txt"
    else:
        timestamps_path = path + "/*.txt"

    files = sorted(glob.glob(timestamps_path))

    if files is None:
        raise RuntimeError("No files found in {}".format(path))

    # Define timestamps
    timestamps = np.empty(len(files), dtype=float)

    for idx, file in enumerate(files):
        with open(file) as f:
            # Read line, convert string to number.
            time = float(f.readline())
            timestamps[idx] = time

    # Set first timestamp as 0 and normalize
    timestamps = timestamps - timestamps[0]

    # Convert from nano seconds to seconds
    timestamps = timestamps * 10E-9


    return timestamps


def write_timestamps_to_file(timestamps, out_path):
    if out_path.endswith("/"):
        timestamps_file = out_path + "times.txt"
    else:
        timestamps_file = out_path + "/times.txt"

    np.savetxt(timestamps_file, timestamps, fmt='%.06e')


def main():
    # Get path to timestamps folder from argument parser.
    path, out_path = get_args()

    # Extract timestamps from files.
    timestamps = extract_timestamps(path)

    # Generate times.txt.
    write_timestamps_to_file(timestamps, out_path)


if __name__ == "__main__":
    try:
        main()
        print("Timestamps file created successfully!")
    except AssertionError as err:
        print(err.args[0])
