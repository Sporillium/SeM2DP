# Main executable part of the SEM2DP Program

# Custom File Imports:
import segmentation as seg
import stereo
import cloud
import m2dp

# Python Package imports
import numpy as np
import cv2 as cv
from tqdm import trange

# Define Execution flags:
SHOW_POSE_ESTIMATE = True

# ----- Main Execution Starts here -----
# Load processing objects:
segmentation_engine = seg.SegmentationEngine(model_id=6, use_gpu=False)
stereo_extractor = stereo.StereoExtractor(segmentation_engine, detector='SIFT', matcher='BF', camera_id=0, seq=0)
cloud_engine = cloud.CloudProcessor(stereo_extractor)

# Load Pose File:
try: 
    poses = np.loadtxt(stereo_extractor.poses)
except:
    pose_file = False
else:
    pose_file = True

# Define image regions:
IM_START = 0
IM_END = IM_START + 5

single_frames = {}
pose_changes = {}
point_cloud = []

# Process first frame in the sequence
single_frames[IM_START] = cloud_engine.processFrame(IM_START)
for point in single_frames[IM_START]:
    point.frame_num = IM_START

# Evaluate Pose estimates from points in the frame
if SHOW_POSE_ESTIMATE:
    origin = np.array([0, 0, 0])
    print("\nTimestep 0 Inf:\tX:{: .3f}\tY:{: .3f}\tZ:{: .3f}".format(origin[0], origin[1], origin[2]))

    if pose_file:
        base_origin = np.array([0, 0, 0, 1])
        trans = np.reshape(poses[IM_START,:], (3,4))
        base_origin = trans@base_origin.T
        ref_origin = base_origin.copy()
        base_origin = np.concatenate((base_origin, 1), axis=None)
        print("Timestep 0 GT:\tX:{: .3f}\tY:{: .3f}\tZ:{: .3f}".format((ref_origin[0]-base_origin[0]), (ref_origin[1]-base_origin[1]), (ref_origin[2]-base_origin[2])))
    
    print("Total points found in timestep 0: {}\n".format(len(single_frames[IM_START])))

for im in trange(IM_START+1, IM_END):
    single_frames[im] = cloud_engine.processFrame(im)
    for point in single_frames[im]:
        point.frame_num = im
    
    pose_changes[im], non_match_points = cloud_engine.processPoseChange(single_frames[im-1], single_frames[im])

    cloud_engine.merge_point_cloud(point_cloud, non_match_points, transform=pose_changes[im])

    if SHOW_POSE_ESTIMATE:
            origin = (pose_changes[im][:, 0:3] @ origin) + pose_changes[im][:, 3]
            print("Timestep {} Inf:\tX:{: .3f}\tY:{: .3f}\tZ:{: .3f}".format(im, origin[0], origin[1], origin[2]))

            if pose_file:
                trans = np.reshape(poses[im,:], (3,4))
                new_origin = trans@base_origin.T
                new_origin = np.concatenate((new_origin, 1), axis=None)
                print("Timestep {} GT:\tX:{: .3f}\tY:{: .3f}\tZ:{: .3f}".format(im, (ref_origin[2]-new_origin[2]), -(ref_origin[0]-new_origin[0]), -(ref_origin[1]-new_origin[1])))
            print("Total points found in timestep {}: {}\n".format(im, len(single_frames[im])))
        
point_cloud.extend(single_frames[IM_END-1])
print("Total Points in Point Cloud: {}".format(len(point_cloud)))

volumes = np.array([np.linalg.det(point.location_cov) for point in point_cloud])
volume_thresh = 1e-8
print("The Maximum 'Volume' is {}\nThe Minimum 'Volume' is {}".format(np.max(volumes), np.min(volumes)))
ii = np.where(volumes < volume_thresh)[0]
filtered_cloud = [point_cloud[i] for i in ii]
print("Total Points in Filtered Point Cloud: {}\n".format(len(filtered_cloud)))

print("COMPLETED")