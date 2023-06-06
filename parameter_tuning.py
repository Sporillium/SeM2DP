# Script for auto tuning of parameters (Optimising mAP)

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import cv2 as cv
import argparse
from tqdm import trange

# Custom File Imports:
import segmentation as seg
import stereo
import cloud
import lidar
from sem2dp import createSemDescriptorHisto_Tune

# Function Definitions
def generate_decriptor_array(circs, bins, cloud_engine, seq_leng):
    print("EVAL @ L="+str(circs)+", T="+str(bins))
    des_size = ((4*16)+((16*8)+(150*circs*bins)))

    descriptors = np.zeros((seq_leng,des_size))
    for im in trange(seq_leng):
        point_cloud, labels = cloud_engine.processFrame(im)
        descriptors[im,:] = createSemDescriptorHisto_Tune(point_cloud, labels, C=circs, B=bins)
    return descriptors, des_size

def average_precision(prec, rec):
    av_prec = 0
    for i in range(len(prec)-1):
        av_prec += (rec[i+1]-rec[i])*prec[i+1]
    return av_prec

def evaluate_matches(input_descriptors, cloud_ids, distances, des_size):
    matcher = cv.BFMatcher_create(cv.NORM_L2)
    match_boundary = 50 # Number of frames before/after matching can occur
    gt_threshold = 10.0
    
    thresholds = np.arange(0.0, 1.0, 0.005)
    precision = []
    recall = []

    for thresh in thresholds:
        # Define values:
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for id in cloud_ids:
            base_cloud = input_descriptors[id,:].reshape((1,des_size))    
            # Create valid matching set:
            if id-match_boundary <= 0:
                continue
            else:
                valid_descriptors = input_descriptors[:id-match_boundary, :]
                valid_ids = cloud_ids[:id-match_boundary]
            
            # Run matcher
            match_result = matcher.match(base_cloud, valid_descriptors)
            best_match = match_result[0]

            matched_cloud_id = valid_ids[best_match.trainIdx]
            match_distance = best_match.distance

            # Perform Evaluations:
            dist_vector = distances[id, :id-match_boundary]
            if match_distance <= thresh:
                if dist_vector[matched_cloud_id] <= gt_threshold:
                    tp += 1
                else:
                    fp += 1
            else:
                if dist_vector[matched_cloud_id] <= gt_threshold: 
                # dist_vector[np.logical_and(dist_vector > 0.0, dist_vector < gt_threshold)].shape[0] > 0:
                    fn += 1
                else:
                    tn += 1
        try:
            precision.append(tp/(tp+fp))
        except:
            precision.append(1)
        
        try:
            recall.append(tp/(tp+fn))
        except:
            recall.append(0)
    
    mAP = average_precision(precision, recall)
    
    return mAP

if __name__ == '__main__':
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
    seq_leng = stereo_extractor.seq_len
    seq_name = f'{seq:02}'

    print("\n\nPARAMETER SUMMARY:\n\tMODEL TYPE:\t"+mod_type+"\n\tTUNING SEQ:\t"+seq_name+"\n\tCIRCLE RANGE:\t1-"+str(max_circles)+"\n\tBIN RANGE:\t1-"+str(max_bins))

    # Load GT data
    poses = np.loadtxt('/home/march/devel/datasets/Kitti/odometry-2012/poses/'+seq_name+'.txt')
    locations = np.zeros((poses.shape[0], 3))
    for i in range(poses.shape[0]):
        base_pose = np.reshape(poses[0, :], (3,4))
        pose = np.reshape(poses[i, :], (3,4))
        locations[i, :] = pose[:,3]
    print("\n\nGT POSES LOADED")
    distances = distance_matrix(locations, locations)

    # Check the number of parameter combinations:
    circles = np.arange(1, max_circles+1)
    bins = np.arange(1, max_bins+1)

    num_eval = len(circles) * len(bins)

    if num_eval > 9:
        print("Using Dynamic Search")
        print(num_eval)
    
    print("Quick Test")

    descriptors, des_len = generate_decriptor_array(1, 1, cloud_engine, seq_leng)
    print(des_len)
    cloud_ids = np.arange(seq_leng)
    mAP = evaluate_matches(descriptors, cloud_ids, distances, des_len)
    print(mAP)