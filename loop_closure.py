# Loop closure testing script
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import cv2 as cv
import argparse
from sem2dp import des_decompress_new, hamming_compare

#np.set_printoptions(threshold=10000)

# Argument Parser:
parser = argparse.ArgumentParser(description="Detect Loop Closures using M2DP Descriptor")
parser.add_argument('-n', '--number', required=True, type=int, help="Specify sequence number to be processes") # Sequence Number
parser.add_argument('-r', '--use_regular', action='store_true', help="Set flag to use visual-based descriptors") # Visual Descriptors
parser.add_argument('-s', '--use_sem', action='store_true', help="Set flag to use visual-semantic based descriptors") #Semantic-Visual Descriptors
parser.add_argument('-v', '--use_velodyne', action='store_true', help="Set flag to use Velodyne-based descriptors") #Use Velo True/False
parser.add_argument('-x', '--use_velo_sem', action='store_true', help="Set flag to use Velodyne-Semantic based descriptors") #Use VeloSem true_false
parser.add_argument('-m', '--use_mod_velo', action='store_true', help="Set flag to use Modified Velodyne based descriptors") # Use Mod Velo true_false
parser.add_argument('-c', '--use_color_velo', action='store_true', help="Set flag to use c-M2DP descriptors") # Use Color true_false
parser.add_argument('-d', '--use_mod_sem', action='store_true', help="Set flag to use new SeM2DP descriptors") # Use mod_sem true_false
parser.add_argument('-f', '--use_mod_sem_vis', action='store_true', help="Set flag to use new Visual SeM2DP descriptors") # use_mod_sem_vis true_false
parser.add_argument('-g', '--use_color_vis', action='store_true', help="Set flag to use new Visual Color descriptors") # use_col_vis true_false

args = parser.parse_args()

# Define Execution flags:
SEQ_NUM = args.number

USE_REG = args.use_regular
USE_SEM = args.use_sem
USE_VELO = args.use_velodyne
USE_MOD_VELO = args.use_mod_velo
USE_VELO_SEM = args.use_velo_sem
USE_COLOR_VELO = args.use_color_velo
USE_MOD_SEM = args.use_mod_sem
USE_MOD_SEM_VIS = args.use_mod_sem_vis
USE_COLOR_VIS = args.use_color_vis

seq_name = f'{SEQ_NUM:02}'

if SEQ_NUM > 10:
    print("NO GROUND TRUTH AVAILABLE FOR SEQUENCE " +seq_name+". PLEASE USE DIFFERENT SCRIPT")
    exit()

if (USE_REG or USE_SEM or USE_VELO or USE_MOD_VELO or USE_VELO_SEM) is False:
    print("AT LEAST ONE DESCRIPTOR TYPE MUST BE USED. PLEASE SEE HELP FOR DETAILS")
    exit()

match_boundary = 50 # Number of frames before/after matching can occur
gt_threshold = 10.0
SEARCH_RANGE = 1.0
SEARCH_INTERVAL = 0.005

des_size = 2048
des_size_velo = 8192
des_size_col = 576
des_size_mod_sem = 1392

matcher = cv.BFMatcher_create(cv.NORM_L2)
matcher_sem = cv.BFMatcher_create(cv.NORM_HAMMING)

def evaluate_match(input_descriptor, cloud_ids, distances, des_size):
    # Define values:
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for id in cloud_ids:
        base_cloud = input_descriptor[id,:].reshape((1,des_size))    
        # Create valid matching set:
        if id-match_boundary <= 0:
            continue
        else:
            valid_descriptors = input_descriptor[:id-match_boundary, :]
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
        prec = tp/(tp+fp)
    except:
        prec = 1
    
    try:
        rec = tp/(tp+fn)
    except:
        rec = 0

    return prec, rec

def evaluate_match_sem(input_descriptors_normal, input_descriptors_sem, cloud_ids, distances, des_size):
    # Define values:
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for id in cloud_ids:
        base_cloud = input_descriptors_normal[id,:].reshape((1, 192))
        base_sem = input_descriptors_sem[id,:].reshape((1, des_size))

        # Create our search space:
        if id-match_boundary <= 0:
            continue
        else:
            valid_descriptors_normal = input_descriptors_normal[:id-match_boundary,:]
            valid_descriptors_sem = input_descriptors_sem[:id-match_boundary,:]
            valid_ids = cloud_ids[:id-match_boundary]

        # Calculate the Hamming distances between the base and all the valid    
        hamming_distances = hamming_compare(base_sem, valid_descriptors_sem)
        #match_result_sem = matcher_sem.match(base_sem, valid_descriptors_sem)
        #print(len(match_result_sem))
        #hamming_distances = np.asarray([mat.distance for mat in match_result_sem])
        #print(hamming_distances)
        hamming_weights = 1/(1-hamming_distances)

        # Modify our descriptor vectors based on the hamming distances
        weighted_descriptors_normal = np.multiply(valid_descriptors_normal, hamming_weights[:, np.newaxis]).astype(np.float32)
        #print(weighted_descriptors_normal, valid_descriptors_normal)
        # Run matcher
        match_result = matcher.match(base_cloud, weighted_descriptors_normal)
        best_match = match_result[0]

        matched_cloud_id = valid_ids[best_match.trainIdx]
        match_distance = best_match.distance
        #print(match_distance)

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
        prec = tp/(tp+fp)
    except:
        prec = 1

    try:
        rec = tp/(tp+fn)
    except:
        rec = 0

    return prec, rec

def find_max_rec(prec_list, rec_list):
    prec = np.asarray(prec_list)
    rec = np.asarray(rec_list)

    return np.max(rec[prec >= 1.0])
    
def average_precision(prec, rec):
    av_prec = 0
    for i in range(len(prec)-1):
        av_prec += (rec[i+1]-rec[i])*prec[i+1]
    return av_prec

# Load Poses for GT Loop closures
poses = np.loadtxt('/home/march/devel/datasets/Kitti/odometry-2012/poses/'+seq_name+'.txt')
locations = np.zeros((poses.shape[0], 3))
for i in range(poses.shape[0]):
    base_pose = np.reshape(poses[0, :], (3,4))
    pose = np.reshape(poses[i, :], (3,4))
    locations[i, :] = pose[:,3]
print("GT POSES LOADED")
distances = distance_matrix(locations, locations)
seq_len = len(locations)

# Load Pre-computed Descriptors for Evaluation based on Flags:
if USE_REG:  
    descriptors_reg = np.zeros((seq_len, 192))
    with open('descriptor_texts/basic_descriptors_kitti_'+seq_name+'.txt', 'r') as file:
        lines = file.readlines()
    for i, line in zip(range(len(lines)), lines):
        try:
            des = np.fromstring(line.strip('[]\n'), sep=';')
            descriptors_reg[i, :] = des
        except:
            print("Error opening Regular Descriptor: ",i)
            #print(line)
    print("VISUAL DESCRIPTORS LOADED")
    descriptors_reg = descriptors_reg.astype(np.float32)
    precision_reg = []
    recall_reg = []

if USE_SEM:  
    descriptors_sem = np.zeros((seq_len, 192))
    sem_descriptors_sem = np.zeros((seq_len, des_size))
    with open('descriptor_texts/sem_descriptors_kitti_'+seq_name+'.txt', 'r') as file:
        lines = file.readlines()
        #print(len(lines))
        des_lines = lines[::2]
        sem_lines = lines[1::2]
    for i in range(len(des_lines)):
        try:
            des = np.fromstring(des_lines[i].strip('[]\n'), sep=';')
            #print(des)
        except:
            print("Error opening Vis-Sem Vis discriptor: ",i)
            #print(des_lines[i])

        try:
            sem_des = np.fromstring(sem_lines[i].strip('[]\n'), sep=';').astype(np.uint8)
        except:
            print("Error opening Vis-Sem Sem discriptor: ",i)
            #print(sem_lines[i])
        sem_des_decomp = des_decompress_new(sem_des, des_size)
        descriptors_sem[i, :] = des
        sem_descriptors_sem[i, :] = sem_des_decomp
    print("VISUAL-SEMANTIC DESCRIPTORS LOADED")
    descriptors_sem = descriptors_sem.astype(np.float32)
    sem_descriptors_sem = sem_descriptors_sem.astype(np.uint8)
    precision_sem = []
    recall_sem = []

if USE_VELO:  
    descriptors_velo = np.zeros((seq_len, 192))
    with open('descriptor_texts/velo_descriptors_kitti_'+seq_name+'.txt', 'r') as file:
        lines = file.readlines()
    for i, line in zip(range(len(lines)), lines):
        try:
            des = np.fromstring(line.strip('[]\n'), sep=';')
            descriptors_velo[i, :] = des
        except:
            print("Error opening Velo discriptor: ",i)
            #print(line)
    print("VELODYNE DESCRIPTORS LOADED")
    descriptors_velo = descriptors_velo.astype(np.float32)
    precision_velo = []
    recall_velo = []

if USE_MOD_VELO:  
    descriptors_mod_velo = np.zeros((seq_len, 192))
    with open('descriptor_texts/mod_velo_descriptors_kitti_'+seq_name+'.txt', 'r') as file:
        lines = file.readlines()
    for i, line in zip(range(len(lines)), lines):
        try:
            des = np.fromstring(line.strip('[]\n'), sep=';')
            descriptors_mod_velo[i, :] = des
        except:
            print("Error opening Mod-Velo discriptor: ",i)
            #print(line)
    print("MODIFIED VELODYNE DESCRIPTORS LOADED")
    descriptors_mod_velo = descriptors_mod_velo.astype(np.float32)
    precision_mod_velo = []
    recall_mod_velo = []

if USE_VELO_SEM:  
    descriptors_velo_sem = np.zeros((seq_len, 192))
    sem_descriptors_velo_sem = np.zeros((seq_len, des_size_velo))
    with open('descriptor_texts/sem_velo_descriptors_kitti_'+seq_name+'.txt', 'r') as file:
        lines = file.readlines()
        #print(len(lines))
        des_lines = lines[::2]
        sem_lines = lines[1::2]
    for i in range(len(des_lines)):
        try:
            des = np.fromstring(des_lines[i].strip('[]\n'), sep=';')
            #print(des)
        except:
            print("Error opening Velo-Sem Vis discriptor: ",i)
            #print(des_lines[i])

        try:
            sem_des = np.fromstring(sem_lines[i].strip('[]\n'), sep=';').astype(np.uint8)
        except:
            print("Error opening Velo-Sem Sem discriptor: ",i)
            #print(sem_lines[i])
        velo_sem_des_decomp = des_decompress_new(sem_des, des_size_velo)
        descriptors_velo_sem[i, :] = des
        sem_descriptors_velo_sem[i, :] = velo_sem_des_decomp
    print("VELO-SEMANTIC DESCRIPTORS LOADED")
    descriptors_velo_sem = descriptors_velo_sem.astype(np.float32)
    sem_descriptors_velo_sem = sem_descriptors_velo_sem.astype(np.uint8)
    precision_velo_sem = []
    recall_velo_sem = []

if USE_COLOR_VELO:
    descriptors_color_velo = np.zeros((seq_len, 576))
    with open('descriptor_texts/color_velo_descriptors_kitti_'+seq_name+'.txt', 'r') as file:
        lines = file.readlines()
    for i, line in zip(range(len(lines)), lines):
        try:
            des = np.fromstring(line.strip('[]\n'), sep=';')
            descriptors_color_velo[i, :] = des
        except:
            print("Error opening color discriptor: ",i)
            #print(line)
    print("COLOR-VELO DESCRIPTORS LOADED")
    descriptors_color_velo = descriptors_color_velo.astype(np.float32)
    precision_color_velo = []
    recall_color_velo = []

if USE_MOD_SEM:
    descriptors_mod_sem = np.zeros((seq_len, des_size_mod_sem))
    with open('descriptor_texts/sem_histo_velo_descriptors_kitti_'+seq_name+'.txt', 'r') as file:
        lines = file.readlines()
    for i, line in zip(range(len(lines)), lines):
        try:
            des = np.fromstring(line.strip('[]\n'), sep=';')
            descriptors_mod_sem[i, :] = des
        except:
            print("Error opening new sem discriptor: ",i)
            #print(line)
    print("SEM-HISTO DESCRIPTORS LOADED")
    descriptors_mod_sem= descriptors_mod_sem.astype(np.float32)
    precision_mod_sem = []
    recall_mod_sem = []

if USE_MOD_SEM_VIS:
    descriptors_mod_sem_vis = np.zeros((seq_len, des_size_mod_sem))
    with open('descriptor_texts/sem_histo_vis_descriptors_kitti_'+seq_name+'.txt', 'r') as file:
        lines = file.readlines()
    for i, line in zip(range(len(lines)), lines):
        try:
            des = np.fromstring(line.strip('[]\n'), sep=';')
            descriptors_mod_sem_vis[i, :] = des
        except:
            print("Error opening new sem discriptor: ",i)
            #print(line)
    print("VIS-SEM-HISTO DESCRIPTORS LOADED")
    descriptors_mod_sem_vis = descriptors_mod_sem_vis.astype(np.float32)
    precision_mod_sem_vis = []
    recall_mod_sem_vis = []

if USE_COLOR_VIS:
    descriptors_color_vis = np.zeros((seq_len, 576))
    with open('descriptor_texts/color_vis_descriptors_kitti_'+seq_name+'.txt', 'r') as file:
        lines = file.readlines()
    for i, line in zip(range(len(lines)), lines):
        try:
            des = np.fromstring(line.strip('[]\n'), sep=';')
            descriptors_color_vis[i, :] = des
        except:
            print("Error opening color discriptor: ",i)
            #print(line)
    print("COLOR-VIS DESCRIPTORS LOADED")
    descriptors_color_vis = descriptors_color_vis.astype(np.float32)
    precision_color_vis = []
    recall_color_vis = []

cloud_ids = np.arange(seq_len)

# Loop through distance Thresholds:
thresholds = np.arange(0.0, SEARCH_RANGE, SEARCH_INTERVAL)
for thresh in tqdm(thresholds):
    if USE_REG:
        prec_reg, rec_reg = evaluate_match(descriptors_reg, cloud_ids, distances, 192)
        precision_reg.append(prec_reg)
        recall_reg.append(rec_reg)
    if USE_SEM:
        prec_sem, rec_sem = evaluate_match_sem(descriptors_sem, sem_descriptors_sem, cloud_ids, distances, des_size)
        precision_sem.append(prec_sem)
        recall_sem.append(rec_sem)
    if USE_VELO:
        prec_velo, rec_velo = evaluate_match(descriptors_velo, cloud_ids, distances, 192)
        precision_velo.append(prec_velo)
        recall_velo.append(rec_velo)
    if USE_MOD_VELO:
        prec_mod_velo, rec_mod_velo = evaluate_match(descriptors_mod_velo, cloud_ids, distances, 192)
        precision_mod_velo.append(prec_mod_velo)
        recall_mod_velo.append(rec_mod_velo)
    if USE_VELO_SEM:
        prec_velo_sem, rec_velo_sem = evaluate_match_sem(descriptors_velo_sem, sem_descriptors_velo_sem, cloud_ids, distances, des_size_velo)
        precision_velo_sem.append(prec_velo_sem)
        recall_velo_sem.append(rec_velo_sem)
    if USE_COLOR_VELO:
        prec_color_velo, rec_color_velo = evaluate_match(descriptors_color_velo, cloud_ids, distances, des_size_col)
        precision_color_velo.append(prec_color_velo)
        recall_color_velo.append(rec_color_velo)
    if USE_MOD_SEM:
        prec_mod_sem, rec_mod_sem = evaluate_match(descriptors_mod_sem, cloud_ids, distances, des_size_mod_sem)
        precision_mod_sem.append(prec_mod_sem)
        recall_mod_sem.append(rec_mod_sem)
    if USE_MOD_SEM_VIS:
        prec_mod_sem_vis, rec_mod_sem_vis = evaluate_match(descriptors_mod_sem_vis, cloud_ids, distances, des_size_mod_sem)
        precision_mod_sem_vis.append(prec_mod_sem_vis)
        recall_mod_sem_vis.append(rec_mod_sem_vis)
    if USE_COLOR_VIS:
        prec_color_vis, rec_color_vis = evaluate_match(descriptors_color_vis, cloud_ids, distances, des_size_col)
        precision_color_vis.append(prec_color_vis)
        recall_color_vis.append(rec_color_vis)
# Plot the Curve
print("\n\n") 

# Figure Display Settings:
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


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')
ax.set_title("Precision - Recall Curve: Sequence "+seq_name)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
if USE_REG:
    reg_key, = ax.plot(recall_reg, precision_reg,  'b-')
    reg_key.set_label("Visual Baseline(AP="+f'{average_precision(precision_reg, recall_reg):.3}'+")")
    print("Visual Recall @ 100% Precision: "+f'{find_max_rec(precision_reg, recall_reg):.3}')
if USE_SEM:
    sem_key, = ax.plot(recall_sem, precision_sem,  'r-')
    sem_key.set_label("Visual-Semantic(ArgMax)(AP="+f'{average_precision(precision_sem, recall_sem):.3}'+")")
    print("Visual-Semantic Recall @ 100% Precision: "+f'{find_max_rec(precision_sem, recall_sem):.3}')
if USE_MOD_SEM_VIS:
    mod_sem_vis_key, = ax.plot(recall_mod_sem_vis, precision_mod_sem_vis,  'g-')
    mod_sem_vis_key.set_label("Visual-Semantic(Histogram)(AP="+f'{average_precision(precision_mod_sem_vis, recall_mod_sem_vis):.3}'+")")
    print("Mod-Sem_Vis Recall @ 100% Precision: "+f'{find_max_rec(precision_mod_sem_vis, recall_mod_sem_vis):.3}')
if USE_COLOR_VIS:
    color_vis_key, = ax.plot(recall_color_vis, precision_color_vis,  'm-')
    color_vis_key.set_label("Visual-Colour(AP="+f'{average_precision(precision_color_vis, recall_color_vis):.3}'+")")
    print("Color-Vis Recall @ 100% Precision: "+f'{find_max_rec(precision_color_vis, recall_color_vis):.3}')

if USE_VELO:
    velo_key, = ax.plot(recall_velo, precision_velo,  'y-')
    velo_key.set_label("Pure Velodyne(AP="+f'{average_precision(precision_velo, recall_velo):.3}'+")")
    print("Velodyne Recall @ 100% Precision: "+f'{find_max_rec(precision_velo, recall_velo):.3}')

if USE_MOD_VELO:
    mod_velo_key, = ax.plot(recall_mod_velo, precision_mod_velo,  'b--')
    mod_velo_key.set_label("Constrained Velodyne(AP="+f'{average_precision(precision_mod_velo, recall_mod_velo):.3}'+")")
    print("Modified Velodyne Recall @ 100% Precision: "+f'{find_max_rec(precision_mod_velo, recall_mod_velo):.3}')
if USE_VELO_SEM:
    velo_sem_key, = ax.plot(recall_velo_sem, precision_velo_sem,  'r--')
    velo_sem_key.set_label("Velodyne-Semantic(ArgMax)(AP="+f'{average_precision(precision_velo_sem, recall_velo_sem):.3}'+")")
    print("Velodyne-Semantic Recall @ 100% Precision: "+f'{find_max_rec(precision_velo_sem, recall_velo_sem):.3}')
if USE_MOD_SEM:
    mod_sem_key, = ax.plot(recall_mod_sem, precision_mod_sem,  'g--')
    mod_sem_key.set_label("Velodyne-Semantic(Histogram)(AP="+f'{average_precision(precision_mod_sem, recall_mod_sem):.3}'+")")
    print("Mod-Sem Recall @ 100% Precision: "+f'{find_max_rec(precision_mod_sem, recall_mod_sem):.3}')
if USE_COLOR_VELO:
    color_velo_key, = ax.plot(recall_color_velo, precision_color_velo,  'm--')
    color_velo_key.set_label("Velodyne-Colour(AP="+f'{average_precision(precision_color_velo, recall_color_velo):.3}'+")")
    print("Color-Velo Recall @ 100% Precision: "+f'{find_max_rec(precision_color_velo, recall_color_velo):.3}')


ax.grid()
ax.legend()
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.01])
plt.show()

    