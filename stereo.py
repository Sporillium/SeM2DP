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
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=100)
                self.matcher = cv.FlannBasedMatcher(index_params, search_params)

        elif self.detectorID == 'ORB':
            if self.matcherID == 'BF':
                self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
            elif self.matcherID == 'NN':
                FLANN_INDEX_LSH = 6
                index_params= dict(algorithm=FLANN_INDEX_LSH,
                                table_number=12,
                                key_size=20,
                                multi_probe_level=2)
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
    
    def computeRatioTest(self, matches, ratio=0.7):
        """
        Computes the ratios test as defined by D. Lowe in the original SIFT paper

        Parameters:
        ----------
            matches: List of match pairs from the knn matcher
            ratio: difference ratio needed for computing.
                    Default value: 0.7
            
        Returns:
        ----------
            good: List of good matches after ratio test
        """
        good = []
        for m,n in matches:
            if m.distance < ratio*n.distance:
                good.append(m)
        return good
    
    def epipolarFilter(self, matches, kp_left, kp_right, filter_threshold=0.0, disp_threshold=0.5):
        """
        Applies a filter to point cloud based on the epipolar constraint in stereo-rectified images

        Parameters:
        ----------
            matches: list of matched points
            kp_left: key points from the left image
            kp_right: key points from the right image
            filter_threshold: Allowable tolerance for deviation in the image plane x-direction
                                Default value: 0.0
            disp_threshold: Allowable tolerance for deviation in the image plane y-direction
                                Default value: 0.5
        
        Returns:
        ----------
            ret_matches: List of filtered matches

        """
        ret_matches = []
        for mat in matches:
            (xL, yL) = kp_left[mat.queryIdx].pt
            (xR, yR) = kp_right[mat.trainIdx].pt

            if ((xL - xR) > filter_threshold) and (abs(yL - yR) < disp_threshold):
                ret_matches.append(mat)

        return ret_matches 

    def pointsFromImages(self, im_no):
        """
        Extracts points from pair of stereo images, filters them, and returns matched points, as well as descriptors

        Parameters:
        ----------
            im_no: ID Number of Image to Process
            
        Returns:
        ----------
            epi_matches: List of matched points
            kpL: keypoints from left image
            kpR: keypoints from right image
            desL: descriptors from left image
            desR: descriptors from right image
        """
        if self.cam_name != 'kitti':
            print("Behavior not implemented!")
            exit()
        else:
            image_str = f'{im_no:06}'

            imgL = cv.imread(self.path_l+image_str+".png", cv.IMREAD_UNCHANGED)
            imgR = cv.imread(self.path_r+image_str+".png", cv.IMREAD_UNCHANGED)
            imgL = cv.cvtColor(imgL, cv.COLOR_BGR2RGB)
            imgR = cv.cvtColor(imgR, cv.COLOR_BGR2RGB)

            kpL, desL = self.detector.detectAndCompute(imgL, None)
            kpR, desR = self.detector.detectAndCompute(imgR, None)

            if self.detectorID != 'ORB':
                initial_matches = self.matcher.knnMatch(desL, desR)
                initial_matches = self.computeRatioTest(initial_matches)
            else:
                initial_matches = self.matcher.match(desL, desR)
            
            epi_matches = self.epipolarFilter(initial_matches, kpL, kpR, filter_threshold=5.0)

        return epi_matches, kpL, kpR, desL, desR
    
    def semanticsFromImages(self, im_no, segmentation_engine, show_seg=False):
        """
        Extracts semantic distributions from pair of stereo images, and then returns the full class distributions

        Parameters:
        ----------
            im_no: ID Number of Image to Process
            segmentation_engine: Object that can process the semantic segmentation
            show_seg: Flag to display the segmentation results
                    Default value: False
        Returns:
        ----------
            distL: Distribution tensor of left image
            distR: Distribution tensor of right image
        """
        if self.cam_name != 'kitti':
            print("Behavior not implemented!")
            exit()
        else:
            image_str = f'{im_no:06}'

            imgL = cv.imread(self.path_l+image_str+".png", cv.IMREAD_UNCHANGED)
            imgR = cv.imread(self.path_r+image_str+".png", cv.IMREAD_UNCHANGED)
            imgL = cv.cvtColor(imgL, cv.COLOR_BGR2RGB)
            imgR = cv.cvtColor(imgR, cv.COLOR_BGR2RGB)

            imgL_proc = cv.GaussianBlur(imgL, (5,5), 0)
            imgR_proc = cv.GaussianBlur(imgR, (5,5), 0)

            # segment images and get the correct outputs:
            distL = segmentation_engine.segmentImageDist(imgL_proc)
            distR = segmentation_engine.segmentImageDist(imgR_proc)

            if show_seg: # Display segmentation results
                visL = segmentation_engine.segmentImageVis(imgL_proc)
                visL = cv.cvtColor(visL, cv.COLOR_BGR2RGB)
                cv.imshow("Left Image Segmentation", visL)
                cv.waitKey(0)
                cv.destroyAllWindows()

            return distL, distR

# ----- Function Definitions -----


# Code if executed as main

