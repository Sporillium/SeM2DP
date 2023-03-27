# Loop closure testing script
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import matplotlib.pyplot as plt

match_boundary = 50 # Number of frames before matching can occur

# Load Poses for GT Loop closures
poses = np.loadtxt('/home/march/devel/datasets/Kitti/odometry-2012/poses/00.txt')
locations = []
for i in range(poses.shape[0]):
    pose = np.reshape(poses[i, :], (3,4))
    locations.append(pose[:,3])
print("GT POSES LOADED")

# Compute GT Matches for pose file:
gt = {}
for p in tqdm(range(len(locations))):
    if p < match_boundary:
        gt[p] = -1
        continue
    else:
        base_loc = locations[p]
        min_dist = 10000
        ind = -1
        for i in range(p-match_boundary, 0, -1):
            dist = np.linalg.norm(base_loc - locations[i])
            if dist < min_dist:
                min_dist = dist
                ind = i
        if min_dist < 10:
            gt[p] = ind
        else:
            gt[p] = -1
print("GT MATCHES COMPUTED")


# Load Pre-computed Descriptors for Evaluation
descriptors = {}
with open('descriptors.txt', 'r') as file:
    lines = file.readlines()
for i, line in zip(range(len(lines)), lines):
    try:
        des = np.fromstring(line.strip('[]\n'), sep=';')
        descriptors[i] = des
    except:
        print(i)
        print(line)
print("DESCRIPTORS LOADED")

precision = []
recall = []
# Define Thresholds and matching objects
#threshold = 0.5
neigh = NearestNeighbors(n_neighbors=1)
ranges = np.arange(0.1, 3.5, 0.1)
for threshold in tqdm(ranges):
    valid_frames = []
    matches = {}

    for frame in descriptors.keys():
        if frame < match_boundary:
            matches[frame] = -1
            continue
        else:
            valid_frames.append(descriptors[frame-match_boundary])
            neigh.fit(valid_frames)
            neigh_dist, neigh_ind = neigh.kneighbors([descriptors[frame]], 1)
            if neigh_dist < threshold:
                matches[frame] = neigh_ind[0][0]
            else:
                matches[frame] = -1

    TN = TP = FN = FP = 0
    for frame in matches.keys():
        base_frame = frame
        match_frame = matches[frame]

        if match_frame != -1:
            base_pose = locations[frame]
            match_pose = locations[matches[frame]]
            dist = np.linalg.norm(base_pose-match_pose)
            if dist < 10:
                TP += 1
            else:
                FP += 1
        else:
            gt_match_frame = gt[frame]
            if gt_match_frame != -1:
                FN += 1
            else:
                TN += 1

    prec = 0
    rec = 0

    try:
        rec = TP/(TP+FN)
    except:
        recall.append(-1)
    else:
        recall.append(rec)

    try:
        prec = TP/(TP+FP)
    except:
        precision.append(1)
    else:
        precision.append(prec)
    
    
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("Precision - Recall Curve")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.plot(recall, precision, 'b-')
ax.grid()

plt.show()
    