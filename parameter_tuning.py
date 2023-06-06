# Script for auto tuning of parameters (Optimising mAP)

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import cv2 as cv
import argparse

# Custom File Imports:
import segmentation as seg
import stereo
import cloud
import lidar
from m2dp import createDescriptor, createColorDescriptor
from sem2dp import createSemDescriptor, des_compress_new, createSemDescriptorHisto, des_decompress_new, hamming_compare

# Argument Parser Options:
parser = argparse.ArgumentParser(description="Perform Parameter Tuning on SeM2DP Models")
# Required Parameters
parser.add_argument('-l', '--circles', required=True, type=int, help="Specify the maximum number of circles to divide Semantic Descriptor into")
parser.add_argument('-t', '--bins', required=True, type=int, help="Specify the maximum number of radial bins to divide each circle into")
parser.add_argument('-n', '--sequence', required=True, type=int, help="Specify the KITTI sequence to use for tuning")
# Other Parameters
parser.add_argument('-m', '--model', default='normal', type=str, help="Specify the model type, either 'normal', 'super', or 'hamming'")

# Parser arguments and check for inconsistencies
args = parser.parse_args()

max_circles = args.circles
max_bins = args.bins
seq = args.sequence
mod_type = args.model

if mod_type not in ['normal', 'super', 'hamming']:
    print("Undefined Model Type Specified, please choose either: \n\n\tnormal \n\tsuper \n\thamming")
    exit()

if seq not in range(11):
    print("Chosen sequence has no ground truth available, please choose from Kitti Sequences 00 thru 10")
    exit()

if max_circles < 1:
    print("Please select a number of circles equal to or greater than 1")
    exit()

if max_bins < 1:
    print("Please select a number of bins greater than or equal to 1")
    exit()

# Create all of the model required systems:
segmentation_engine = seg.SegmentationEngine(model_id=6, use_gpu=False)
stereo_extractor = stereo.StereoExtractor(segmentation_engine, detector='SIFT', matcher='BF', camera_id=0, seq=seq)
cloud_engine = cloud.CloudProcessor(stereo_extractor)
velo_proc = lidar.VelodyneProcessor(camera_id=0, seq=seq)
seq_leng = stereo_extractor.seq_len
seq_name = f'{seq:02}'

print("\n\nPARAMETER SUMMARY:\n\tMODEL TYPE:\t"+mod_type+"\n\tTUNING SEQ:\t"+seq_name+"\n\tCIRCLE RANGE:\t1-"+str(max_circles)+"\n\tBIN RANGE:\t1-"+str(max_bins))

# Check the number of parameter combinations:
circles = np.arange(1, max_circles+1)
bins = np.arange(1, max_bins+1)

num_eval = len(circles) * len(bins)

if num_eval > 9:
    print("Using Dynamic Search")
    print(num_eval)