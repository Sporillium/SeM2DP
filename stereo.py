# File for Functions to do with extracting features from stereo images

# Package imports
import cv2 as cv
from cv2 import xfeatures2d as xf2d
import numpy as np
import random
import yaml
from yaml.loader import SafeLoader

# Required parameters and constants


# ----- Class Definitions -----
class StereoExtractor:
    """
    Initialises a StereoExtractor Object, which is used analyse stereo image pairs and extract 3D features from them.

    Parameters:
    ----------
        detector: Specifies which 2D feature extractor to use for extracting features from the 2D images. Options include SIFT, SURF, and ORB.
                    Default value: 'SIFT'
        
        matcher: Specifies which method to use to match extracted features from the left and right images. Options are brute force (BF) or FLANN (NN).
                    Default value: 'BF'
        
        camera_id: Specifies which camera settings to use from camera_info.yaml. 0 = KITTI
                    Default value: 0

        seq: Specifies which sequence (if present) that the system should process. For image systems with only one sequence, None is specified.
                Default value: None 

    Instance Variables:
    ----------
        detector_id: String
        matcher_id: String
        camera_id: int

        cam_name: String
        cam_baseline: float
        cam_fl: float
        cam_xoff: float
        cam_y_off: float

        path_l: String
        path_r: String
        poses: String
        seq_len: int
                    
    Returns:
    ----------
        Instance of Stereo Extractor object object
    """
    def __init__(self, detector='SIFT', matcher='BF', camera_id=0, seq=None):
        self.detectorID = detector
        self.matcherID = matcher
        self.camera_id = camera_id

        with open('./model_info/camera_info.yaml') as f:
            camera_info = yaml.load(f, Loader=SafeLoader)[camera_id]

            self.cam_name = camera_info['name']
            self.cam_baseline = camera_info['baseline']
            self.cam_fl = camera_info['focal_length']
            self.cam_xoff = camera_info['x_off']
            self.cam_yoff = camera_info['y_off']
            
            if self.cam_name == 'kitti': # Check for Kitti Dataset and load correct sequence info
                sequences = camera_info['seq_id']
                len = camera_info['len']
                self.path_l = camera_info['image_path']+sequences[seq]+'image_2/'
                self.path_r = camera_info['image_path']+sequences[seq]+'image_3/'
                self.poses = camera_info['poses_path']+f'{seq:02}'+'.txt'
                self.seq_len = len[seq]
            else:
                self.path_l = camera_info['image_path_l']
                self.path_r = camera_info['image_path_r']
                self.poses = camera_info['poses_path']
                self.seq_len = camera_info['len']


    # ----- Method definitions -----


# ----- Function Definitions -----


# Code if executed as main

