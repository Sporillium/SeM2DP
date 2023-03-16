# Main executable part of the SEM2DP Program

# Custom File Imports:
import segmentation as seg
import stereo
import cloud
import m2dp

# Python Package imports
import numpy as np
import cv2 as cv

# ----- Main Execution Starts here -----
# Load processing objects:
segmentation_engine = seg.SegmentationEngine(model_id=6, use_gpu=False)
stereo_extractor = stereo.StereoExtractor(segmentation_engine, detector='SIFT', matcher='BF', camera_id=0, seq=0)
cloud_engine = cloud.CloudProcessor(stereo_extractor)

# Define image regions:
IM_START = 0
IM_END = IM_START + 5

single_frames = {}
pose_changes = {}
point_cloud = []

single_frames[IM_START] = cloud_engine.processFrame(IM_START)
for point in single_frames[IM_START]:
    point.frame_num = IM_START

print("COMPLETED")
print(len(single_frames[IM_START]))