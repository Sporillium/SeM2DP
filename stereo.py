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

        detector = Instance of OpenCV Feature detector
        matcher = Instance of OpenCV Feature matcher
                    
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
            f.close()
            
        # Define OpenCV Extractors from settings:
        if self.detectorID == 'SIFT':
            self.detector = cv.SIFT_create()
        elif self.detectorID == 'SURF':
            self.detector = xf2d.SURF_create()
        elif self.detectorID == 'ORB':
            self.detector = cv.ORB_create(nfeatures=1000)
        else:
            print("UNDEFINED FEATURE")
            exit(code=2)
        
        # Define OpenCV Matchers from Settings:
        if self.detectorID == 'SIFT' or self.detectorID == 'SURF':
            if self.matcherID == 'BF':
                self.matcher = cv.BFMatcher()
            elif self.matcherID == 'NN':
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
                search_params = dict(checks=100)
                self.matcher = cv.FlannBasedMatcher(index_params, search_params)

        elif self.detectorID == 'ORB':
            if self.matcherID == 'BF':
                self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
            elif self.matcherID == 'NN':
                FLANN_INDEX_LSH = 6
                index_params= dict(algorithm = FLANN_INDEX_LSH,
                                table_number = 12,
                                key_size = 20,
                                multi_probe_level = 2)
                search_params = dict(checks=100) 
                self.matcher = cv.FlannBasedMatcher(index_params, search_params)
        else:
            print("UNDEFINED MATCHER")
            exit(code=2)
    


    # ----- Method definitions -----
    def noiseMatrix(self, XL, YL, XR, YR):
        """
        Defines a noise matrix for the x and y directions of each image

        Parameters:
        ----------
            XL: standard deviation of the left image X noise.
            YL: standard deviation of the left image X noise.
            XR: standard deviation of the right image X noise.
            YR: standard deviation of the right image Y noise.

        """
        self.N = np.array([ [XL**2, 0, 0, 0],
                            [0, YL**2, 0, 0],
                            [0, 0, XR**2, 0],
                            [0, 0, 0, YR**2]])
        
    def measurement3DNoise(self, xl, yl, xr, yr):
        """
        Transforms pixel locations with added zero-mean noise in stereo rectified images into 3D coordinates with zero-mean noise

        Parameters:
        ----------
            xl: X coordinate of left image feature.
            yl: Y coordinate of left image feature.
            xr: X coordinate of right image feature.
            ye: Y coordinate of right image feature.

        Returns:
        ----------
            mu: Numpy array containing the location of the measured mean in 3D space, in the form [X, Y, Z].
            Q:  Numpy array containg the measurement covariance of the given point in 3D space.
        """
        X = (self.cam_fl*self.cam_baseline) / (xl - xr)
        Y = -(((xl-self.cam_xoff)*self.cam_baseline)/(xl - xr) - 0.5*self.cam_baseline)
        Z = -(((0.5*(yl+yr)-self.cam_yoff)*self.cam_baseline)/(xl-xr))

        mu = np.array([X,Y,Z])
        
        W = np.array([  [-(self.cam_fl*self.cam_baseline)/((xl-xr)**2) , 0 , (self.cam_fl*self.cam_baseline)/((xl-xr)**2) , 0],
                        [-(self.cam_baseline*(self.cam_xoff-xr))/((xl-xr)**2) , 0 , (self.cam_baseline*(self.cam_xoff-xl))/((xl-xr)**2) , 0],
                        [(self.cam_baseline*(yl - 2*self.cam_yoff + yr))/(2*(xl-xr)**2) , -self.cam_baseline/(2*(xl-xr)**2) , 
                            -(self.cam_baseline*(yl - 2*self.cam_yoff + yr))/(2*(xl-xr)**2) , -self.cam_baseline/(2*(xl-xr)**2)]])
        
        Q = W @ self.N @ W.T

        return mu, Q

# ----- Function Definitions -----


# Code if executed as main

