# Script for auto tuning of parameters (Optimising mAP)

import numpy as np
from scipy.spatial import distance_matrix
import cv2 as cv
import argparse
from tqdm import trange, tqdm
import itertools as iter
import matplotlib.pyplot as plt
import time

# Custom File Imports:
import segmentation as seg
import stereo
import cloud
from sem2dp import createSemDescriptorHisto_Tune

# Global Settings
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

SMALL_WIDTH = 3
MEDIUM_WIDTH = 4
BIGGER_WIDTH = 5

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title\
plt.rc('lines', linewidth=SMALL_WIDTH)  # line width

# Function Definitions
def generate_decriptor_array(circs, bins, cloud_engine, seq_leng, model):
    label = "DESC @ L="+str(circs)+", T="+str(bins)
    if model == 'normal':
        classes = 150
    elif model == 'super':
        classes = 10
    des_size = ((4*16)+((16*8)+(classes*circs*bins)))
    descriptors = np.zeros((seq_leng,des_size))
    for im in tqdm(range(seq_leng), desc=label, leave=False):
        if model == 'normal':
            point_cloud = cloud_engine.processFrameFast(im)
            classes = 150
        elif model == 'super':
            point_cloud = cloud_engine.processFrameSuperSemantics(im)
            classes = 10
        labels = point_cloud[:, 3]
        points = point_cloud[:, :3]
        descriptors[im,:] = createSemDescriptorHisto_Tune(points, labels, C=circs, B=bins, S=classes)
    descriptors = descriptors.astype(np.float32)
    return descriptors, des_size

def average_precision(prec, rec):
    av_prec = 0
    for i in range(len(prec)-1):
        av_prec += (rec[i+1]-rec[i])*prec[i+1]
    return av_prec

def evaluate_matches(input_descriptors, cloud_ids, distances, des_size, label, return_data=False):
    matcher = cv.BFMatcher_create(cv.NORM_L2)
    match_boundary = 50 # Number of frames before/after matching can occur
    gt_threshold = 10.0
    thresholds = np.arange(0.0, 1.0, 0.005)
    precision = []
    recall = []
    for thresh in tqdm(thresholds, desc='MATCH '+label, leave=False):
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

    if not return_data:
        return mAP
    else:
        return precision, recall, mAP

def eval_Params(params, model):
    (circs, bins) = params
    label = "@ L="+str(circs)+", T="+str(bins)
    descriptors, des_len = generate_decriptor_array(circs, bins, cloud_engine, seq_leng, model)
    cloud_ids = np.arange(seq_leng)
    mAP = evaluate_matches(descriptors, cloud_ids, distances, des_len, label)
    print("mAP: {}".format(mAP))
    print("Descriptor Length: {}".format(des_len))
    print("-------------------------------------")
    return mAP

def define_curves(params, model, axes):
    st = time.time()
    (circs, bins) = params
    label = "@ L="+str(circs)+", T="+str(bins)
    descriptors, des_len = generate_decriptor_array(circs, bins, cloud_engine, seq_leng, model)
    cloud_ids = np.arange(seq_leng)
    precision, recall, mAP = evaluate_matches(descriptors, cloud_ids, distances, des_len, label, return_data=True)
    plot_key, = axes.plot(recall, precision)
    plot_key.set_label(model+" mAP="+f'{mAP:.3}'+" "+label)
    et = time.time()
    elapsed_time = (et-st)/60
    print(model+" mAP="+f'{mAP:.3}'+" "+label+" total elapsed time: {:d}:{:d}".format(elapsed_time/60, elapsed_time%60))
    #print("-------------------------------------")

def bilinear(circs, bins, vals):
    mid_circ = (circs[1] + circs[0])/2
    mid_bin = (bins[1] + bins[0])/2
    weight = 1/((circs[1] - circs[0])*(bins[1] - bins[0]))
    circ_lerp = np.array([circs[1]-mid_circ, mid_circ-circs[0]])
    bin_lerp = np.array([[bins[1]-mid_bin],[mid_bin-bins[0]]])
    val_lerp = np.array([[vals[0], vals[1]],[vals[2], vals[3]]])
    return weight*(circ_lerp@val_lerp@bin_lerp)

def recursive_search(c_min, c_max, b_min, b_max, search_space, model):
    c_mid = int(np.floor((c_min + c_max)/2))
    b_mid = int(np.floor((b_min + b_max)/2))
    # Check if new division is too small
    if c_mid == c_min or b_mid == b_min:
        return None
    # Define Search Parameters
    c_search = [c_min, c_mid, c_max]
    b_search = [b_min, b_mid, b_max]
    search_iteration = iter.product(c_search, b_search)
    # Perform Function Evals
    for pair in search_iteration:
        if search_space[pair] == 0:
            search_space[pair] = eval_Params(pair, model)
        else:
            continue
    
    print(search_space)
    return None
    
def read_file(file_name):
    #print("Reading inputs from "+file_name)
    pairs = []
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if '#' in line:
                continue
            else:
                params = np.fromstring(line, sep=',',dtype=np.int16)
                param_tuple = (params[0], params[1])
                pairs.append(param_tuple)
    return pairs

if __name__ == '__main__':
    # Argument Parser Options:
    parser = argparse.ArgumentParser(description="Perform Parameter Tuning on SeM2DP Models")
    # Required Parameters
    parser.add_argument('-e', '--ex_type', required=True, type=str, help="Specify the execution type, either 'search', 'single', or 'display'")
    parser.add_argument('-n', '--sequence', required=True, type=int, help="Specify the KITTI sequence to use for tuning")
    # Other Parameters
    parser.add_argument('-m', '--model', default='normal', type=str, help="Specify the model type, either 'normal', 'super', or 'hamming'. 'compare' can be use in display mode")
    parser.add_argument('-f', '--file', default=None, type=str, help="Specify file path to parameter pairs for execution. Not available in 'single' mode")
    parser.add_argument('-l', '--circles', default=None, type=int, help="Specify the maximum number of circles to divide Semantic Descriptor into")
    parser.add_argument('-t', '--bins', default=None, type=int, help="Specify the maximum number of radial bins to divide each circle into")
    parser.add_argument('-s', '--save', default=None, type=str, help="Specify file name for saving figures")
    # Parser arguments and check for inconsistencies
    args = parser.parse_args()

    max_circles = args.circles
    max_bins = args.bins
    seq = args.sequence
    mod_type = args.model
    ex_type = args.ex_type
    file_name = args.file
    save_name = args.save

    if mod_type not in ['normal', 'super', 'hamming', 'compare']:
        print("Undefined Model Type Specified, please choose either: \n\n\tnormal \n\tsuper \n\thamming")
        exit()
    if ex_type not in ['search', 'single', 'display']:
        print("Undefined Execution type specified, please choose either 'search' or 'single'")
        exit()
    if mod_type == 'compare' and ex_type != 'display':
        print("'compare mode only available in display mode. Use -e display or --ex_type display to enter this mode")
        exit()
    if seq not in range(11):
        print("Chosen sequence has no ground truth available, please choose from Kitti Sequences 00 thru 10")
        exit()
    if max_circles != None and max_bins != None and file_name == None:
        if max_circles < 1 or max_bins < 1:
            print("Please select valid parameter ranges if an input file is not specified [L >= 1, T >= 1]")
            exit()
    if max_circles == None and max_bins == None and file_name == None:
        print("Please include either a range of Circles and Bins using the -l and -t switches, or a file of parameters using the -f switch")
        exit()

    # Create all of the model required systems:
    segmentation_engine = seg.SegmentationEngine(model_id=6, use_gpu=True)
    stereo_extractor = stereo.StereoExtractor(segmentation_engine, detector='SIFT', matcher='BF', camera_id=0, seq=seq)
    cloud_engine = cloud.CloudProcessor(stereo_extractor)
    seq_leng = stereo_extractor.seq_len
    seq_name = f'{seq:02}'

    # Load GT data
    poses = np.loadtxt('/home/march/devel/datasets/Kitti/odometry-2012/poses/'+seq_name+'.txt')
    locations = np.zeros((poses.shape[0], 3))
    for i in range(poses.shape[0]):
        base_pose = np.reshape(poses[0, :], (3,4))
        pose = np.reshape(poses[i, :], (3,4))
        locations[i, :] = pose[:,3]
    print("GT poses loaded")
    distances = distance_matrix(locations, locations)

    # Check the number of parameter combinations:
    if ex_type == 'search':
        if file_name == None:
            print("\n\nPARAMETER SUMMARY:\n\tEVAL TYPE:\t"+ex_type+"-[range]\n\tMODEL TYPE:\t"+mod_type+"\n\tTUNING SEQ:\t"+seq_name+"\n\tCIRCLE RANGE:\t1-"+str(max_circles)+"\n\tBIN RANGE:\t1-"+str(max_bins)+"\n")
            circles = np.arange(1, max_circles+1)
            bins = np.arange(1, max_bins+1)
            num_eval = len(circles) * len(bins)
            search_space = np.zeros((len(circles)+1, len(bins)+1))
            recursive_search(circles[0], circles[-1], bins[0], bins[-1], search_space, mod_type)
        else:
            param_list = read_file(file_name)
            print("\n\nPARAMETER SUMMARY:\n\tEVAL TYPE:\t"+ex_type+"-[file]\n\tMODEL TYPE:\t"+mod_type+"\n\tTUNING SEQ:\t"+seq_name+"\n\tFILE NAME:\t"+file_name+"\n\tFILE LENGTH:\t"+str(len(param_list))+"\n")
            for pair in param_list:
                eval_Params(pair, mod_type)
    elif ex_type == 'single':
        print("\n\nPARAMETER SUMMARY:\n\tEVAL TYPE:\t"+ex_type+"\n\tMODEL TYPE:\t"+mod_type+"\n\tTUNING SEQ:\t"+seq_name+"\n\tCIRCLE VAL:\t"+str(max_circles)+"\n\tBIN VAL:\t"+str(max_bins)+"\n")
        eval_Params((max_circles,max_bins), mod_type)
    elif ex_type == 'display':
        # Load Parameters to Evaluate
        if file_name == None:
            print("NOT CURRENTLY IMPLEMENTED! PLEASE SPECIFY A FILE WITH DISPLAY MODE")
        else:
            if mod_type != 'compare':
                param_list = read_file(file_name)
                print("\n\nPARAMETER SUMMARY:\n\tEVAL TYPE:\t"+ex_type+"-[file]\n\tMODEL TYPE:\t"+mod_type+"\n\tTUNING SEQ:\t"+seq_name+"\n\tFILE NAME:\t"+file_name+"\n\tFILE LENGTH:\t"+str(len(param_list))+"\n")
                num_plots = len(param_list)
                
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.set_aspect('equal')
                ax.set_title("Precision - Recall Curve: Sequence "+seq_name)
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")

                ax.set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, num_plots))))
            
                for pair in param_list:
                    define_curves(pair, mod_type, ax)

                ax.grid()
                ax.legend()
                ax.set_xlim([-0.01, 1.01])
                ax.set_ylim([-0.01, 1.01])

                manager = plt.get_current_fig_manager()
                manager.window.showMaximized()
                
                fig.set_size_inches((8.5, 11), forward=False)
                plt.savefig(save_name+'.png', dpi=300, bbox_inches='tight')
            else:
                param_list = read_file(file_name)
                print("\n\nPARAMETER SUMMARY:\n\tEVAL TYPE:\t"+ex_type+"-[file]\n\tMODEL TYPE:\t"+mod_type+"\n\tTUNING SEQ:\t"+seq_name+"\n\tFILE NAME:\t"+file_name+"\n\tFILE LENGTH:\t"+str(len(param_list))+"\n")
                num_plots = 2*len(param_list)
                
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.set_aspect('equal')
                ax.set_title("Precision - Recall Curve: Sequence "+seq_name)
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, num_plots))))

                for pair in param_list:
                    define_curves(pair, 'super', ax)
                    define_curves(pair, 'normal', ax)

                ax.grid()
                ax.legend()
                ax.set_xlim([-0.01, 1.01])
                ax.set_ylim([-0.01, 1.01])

                manager = plt.get_current_fig_manager()
                manager.window.showMaximized()
                
                fig.set_size_inches((8.5, 11), forward=False)
                plt.savefig(save_name+'.png', dpi=300, bbox_inches='tight')
                #plt.show()
                