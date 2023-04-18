# Loop closure testing script
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import cv2 as cv

match_boundary = 50 # Number of frames before/after matching can occur
gt_threshold = 5.0

SEQ_NUM = 6
seq_name = f'{SEQ_NUM:02}'

SEARCH_RANGE = 2.0
SEARCH_INTERVAL = 0.001

matcher = cv.BFMatcher_create(cv.NORM_L2)

# Load Poses for GT Loop closures
poses = np.loadtxt('/home/march/devel/datasets/Kitti/odometry-2012/poses/'+seq_name+'.txt')
locations = np.zeros((poses.shape[0], 3))
for i in range(poses.shape[0]):
    base_pose = np.reshape(poses[0, :], (3,4))
    pose = np.reshape(poses[i, :], (3,4))
    locations[i, :] = pose[:,3]
print("GT POSES LOADED")
distances = distance_matrix(locations, locations)

#plt.imshow(distances, cmap='hot')


seq_len = len(locations)

# Load Pre-computed Descriptors for Evaluation
descriptors = np.zeros((seq_len, 192))
with open('descriptor_texts/velo_descriptors_kitti_'+seq_name+'.txt', 'r') as file:
    lines = file.readlines()
for i, line in zip(range(len(lines)), lines):
    try:
        des = np.fromstring(line.strip('[]\n'), sep=';')
        descriptors[i, :] = des
    except:
        print(i)
        print(line)
print("DESCRIPTORS LOADED")
descriptors = descriptors.astype(np.float32)

cloud_ids = np.arange(seq_len)

# Define Lists for Curve:
precision = []
recall = []

# Loop through distance Thresholds:
thresholds = np.arange(0.0, SEARCH_RANGE, SEARCH_INTERVAL)
for thresh in tqdm(thresholds):

    # Define values:
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for id in cloud_ids:
        base_cloud = descriptors[id,:].reshape((1,192))
        
        # Create valid matching set:
        if id-match_boundary <= 0:
            continue
        else:
            valid_descriptors = descriptors[:id-match_boundary, :]
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
    #print(tp, fp, tn, fn)
    # Calculate Precision and recall values:
    prec = tp/(tp+fp)
    rec = tp/(tp+fn)

    precision.append(prec)
    recall.append(rec)

# Plot the Curve   
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')
ax.set_title("Precision - Recall Curve")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.plot(recall, precision,  'b-')
ax.grid()
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])

# path = plt.figure()
# ax1 = path.add_subplot(111, projection='3d')
# ax1.set_title("GT Path")
# #ax1.set_box_aspect((np.ptp(locations[:,0]), np.ptp(locations[:,1]), np.ptp(locations[:,2])))
# ax1.scatter(locations[:,0], locations[:,1], locations[:,2])
# ax1.set_aspect('equal')
# ax1.set_xlabel("X Axis")
# ax1.set_ylabel("Y Axis")
# ax1.set_zlabel("Z Axis")


plt.show()
    