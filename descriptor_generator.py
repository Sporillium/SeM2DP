# Code for evaluating different loop closure methods against ground truth

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
SHOW_POSE_ESTIMATE = False

# ----- Main Execution Starts here -----
# Load processing objects:
segmentation_engine = seg.SegmentationEngine(model_id=6, use_gpu=False)
stereo_extractor = stereo.StereoExtractor(segmentation_engine, detector='SIFT', matcher='BF', camera_id=0, seq=0)
cloud_engine = cloud.CloudProcessor(stereo_extractor)
signature_generator = m2dp.m2dp()

seq_leng = stereo_extractor.seq_len

# Load Pose File:
try: 
    poses = np.loadtxt(stereo_extractor.poses)
except:
    pose_file = False
else:
    pose_file = True

descriptors = {}
with open("descriptors.txt", 'w') as file:
    for im in trange(seq_leng):
        point_cloud = cloud_engine.processFrameNoSemantics(im)
        descriptors[im] = signature_generator.extractAndProcess(point_cloud)
        line = np.array2string(descriptors[im], max_line_width=10000, separator=';')
        file.write(line+"\n")

print(len(descriptors))
