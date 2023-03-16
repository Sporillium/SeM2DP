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
segmentation_engine = seg.SegmentationEngine(model_id=6, use_gpu=False)

stereo_extractor = stereo.StereoExtractor(segmentation_engine, detector='SIFT', matcher='BF', camera_id=0, seq=0)

cloud_engine = cloud.CloudProcessor(stereo_extractor)

print(type(cloud_engine.stereo_extractor))