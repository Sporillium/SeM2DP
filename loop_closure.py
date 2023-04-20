# Loop closure testing script
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import cv2 as cv
import argparse

# Argument Parser:
parser = argparse.ArgumentParser(description="Detect Loop Closures using M2DP Descriptor")
parser.add_argument('-n', '--number', required=True, type=int, help="Specify sequence number to be processes") # Sequence Number
parser.add_argument('-r', '--use_regular', action='store_true', help="Set flag to use visual-based descriptors") # Visual Descriptors
parser.add_argument('-s', '--use_sem', action='store_true', help="Set flag to use visual-semantic based descriptors") #Semantic-Visual Descriptors
parser.add_argument('-v', '--use_velodyne', action='store_true', help="Set flag to use Velodyne-based descriptors") #Use Velo True/False
parser.add_argument('-c', '--use_velo_sem', action='store_true', help="Set flag to use Velodyne-Semantic based descriptors") #Use VeloSem true_false

args = parser.parse_args()

# Define Execution flags:
SEQ_NUM = args.number

USE_REG = args.use_regular
USE_SEM = args.use_sem
USE_VELO = args.use_velodyne
USE_VELO_SEM = args.use_velo_sem

seq_name = f'{SEQ_NUM:02}'

if SEQ_NUM > 10:
    print("NO GROUND TRUTH AVAILABLE FOR SEQUENCE " +seq_name+". PLEASE USE DIFFERENT SCRIPT")
    exit()

if (USE_REG or USE_SEM or USE_VELO or USE_VELO_SEM) is False:
    print("AT LEAST ONE DESCRIPTOR TYPE MUST BE USED. PLEASE SEE HELP FOR DETAILS")
    exit()

match_boundary = 50 # Number of frames before/after matching can occur
gt_threshold = 10.0
SEARCH_RANGE = 2.0
SEARCH_INTERVAL = 0.002

matcher = cv.BFMatcher_create(cv.NORM_L2)

def evaluate_match(input_descriptor, cloud_ids, distances):
    # Define values:
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for id in cloud_ids:
        base_cloud = input_descriptor[id,:].reshape((1,192))    
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
            print(i)
            print(line)
    print("VISUAL DESCRIPTORS LOADED")
    descriptors_reg = descriptors_reg.astype(np.float32)
    precision_reg = []
    recall_reg = []

if USE_SEM:  
    descriptors_sem = np.zeros((seq_len, 192))
    with open('descriptor_texts/sem_descriptors_kitti_'+seq_name+'.txt', 'r') as file:
        lines = file.readlines()
    for i, line in zip(range(len(lines)), lines):
        try:
            des = np.fromstring(line.strip('[]\n'), sep=';')
            descriptors_sem[i, :] = des
        except:
            print(i)
            print(line)
    print("VISUAL-SEMANTIC DESCRIPTORS LOADED")
    descriptors_sem = descriptors_sem.astype(np.float32)
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
            print(i)
            print(line)
    print("VELODYNE DESCRIPTORS LOADED")
    descriptors_velo = descriptors_velo.astype(np.float32)
    precision_velo = []
    recall_velo = []

if USE_VELO_SEM:  
    descriptors_velo_sem = np.zeros((seq_len, 192))
    with open('descriptor_texts/velo_sem_descriptors_kitti_'+seq_name+'.txt', 'r') as file:
        lines = file.readlines()
    for i, line in zip(range(len(lines)), lines):
        try:
            des = np.fromstring(line.strip('[]\n'), sep=';')
            descriptors_velo_sem[i, :] = des
        except:
            print(i)
            print(line)
    print("VELODYNE DESCRIPTORS LOADED")
    descriptors_velo_sem = descriptors_velo_sem.astype(np.float32)
    precision_velo_sem = []
    recall_velo_sem = []

cloud_ids = np.arange(seq_len)

# Loop through distance Thresholds:
thresholds = np.arange(0.0, SEARCH_RANGE, SEARCH_INTERVAL)
for thresh in tqdm(thresholds):
    if USE_REG:
        prec_reg, rec_reg = evaluate_match(descriptors_reg, cloud_ids, distances)
        precision_reg.append(prec_reg)
        recall_reg.append(rec_reg)
    if USE_SEM:
        prec_sem, rec_sem = evaluate_match(descriptors_sem, cloud_ids, distances)
        precision_sem.append(prec_sem)
        recall_sem.append(rec_sem)
    if USE_VELO:
        prec_velo, rec_velo = evaluate_match(descriptors_velo, cloud_ids, distances)
        precision_velo.append(prec_velo)
        recall_velo.append(rec_velo)
    if USE_VELO_SEM:
        prec_velo_sem, rec_velo_sem = evaluate_match(descriptors_velo_sem, cloud_ids, distances)
        precision_velo_sem.append(prec_velo_sem)
        recall_velo_sem.append(rec_velo_sem)
    

# Plot the Curve   
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')
ax.set_title("Precision - Recall Curve: Sequence "+seq_name)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
if USE_REG:
    reg_key, = ax.plot(recall_reg, precision_reg,  'b-')
    reg_key.set_label("Visual")
if USE_SEM:
    sem_key, = ax.plot(recall_sem, precision_sem,  'g-')
    sem_key.set_label("Visual-Semantic")
if USE_VELO:
    velo_key, = ax.plot(recall_velo, precision_velo,  'r-')
    velo_key.set_label("Velodyne")
if USE_VELO_SEM:
    velo_sem_key, = ax.plot(recall_velo_sem, precision_velo_sem,  'y-')
    velo_sem_key.set_label("Velodyne-Semantic")
ax.grid()
ax.legend()
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.01])


plt.show()

    