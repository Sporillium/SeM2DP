# File for Functions to do with extracting features from stereo images

# Package imports
import cv2 as cv
from cv2 import xfeatures2d as xf2d
import numpy as np
import yaml
from yaml.loader import SafeLoader

# Required parameters and constants
FIXED_BLUR = np.array([ [0.0023997,  0.00590231, 0.01012841, 0.0121259,  0.01012841, 0.00590231, 0.0023997 ],
                        [0.00590231, 0.01451734, 0.02491186, 0.02982491, 0.02491186, 0.01451734, 0.00590231],
                        [0.01012841, 0.02491186, 0.04274892, 0.05117975, 0.04274892, 0.02491186, 0.01012841],
                        [0.0121259,  0.02982491, 0.05117975, 0.06127329, 0.05117975, 0.02982491, 0.0121259 ],
                        [0.01012841, 0.02491186, 0.04274892, 0.05117975, 0.04274892, 0.02491186, 0.01012841],
                        [0.00590231, 0.01451734, 0.02491186, 0.02982491, 0.02491186, 0.01451734, 0.00590231],
                        [0.0023997,  0.00590231, 0.01012841, 0.0121259,  0.01012841, 0.00590231, 0.0023997 ]])

DYNAMIC_CLASSES = [12, 20, 76, 80, 83, 90, 102, 103, 116, 127] # Dynamic Object Classes
                    #[Person, Car, Boat, Bus, Truck, Airplane, Van, Ship, Motorbike, Bicycle]
DYNAMIC_CLASSES_SUPER = [8, 9]
                    #[Creatures, Vehicles]
UNCLEAR_CLASSES = [2] # Classes that present non-fixed values
                    #[Sky]
UNCLEAR_CLASSES_SUPER = UNCLEAR_CLASSES
#np.seterr(all='raise')
# ----- Class Definitions -----
class StereoExtractor:
    """
    Initialises a StereoExtractor Object, which is used analyse stereo image pairs and extract 3D features from them.

    Parameters:
    ----------
        seg_engine: Segmentation_Engine object that performs semantic segmentation on the stereo images

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
        seg_engine: SegmentationEngine Object

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
    def __init__(self, seg_engine, detector='SIFT', matcher='BF', camera_id=0, seq=None):
        self.seg_engine = seg_engine

        self.N = noiseMatrix(1, 1, 1, 1)

        self.detectorID = detector
        self.matcherID = matcher
        self.camera_id = camera_id

        with open('./model_info/camera_info.yaml') as f:
            cameras = yaml.load(f, Loader=SafeLoader)['ID']
            camera_info = cameras[camera_id]

            self.cam_name = camera_info['name']
            self.cam_baseline = camera_info['baseline']
            self.cam_fl = camera_info['focal_length']
            self.cam_xoff = camera_info['x_off']
            self.cam_yoff = camera_info['y_off']
            
            if self.cam_name == 'kitti': # Check for Kitti Dataset and load correct sequence info
                sequences = camera_info['seq_id']
                len = camera_info['seq_len']
                self.path_l = camera_info['image_path']+sequences[seq]+'image_2/'
                self.path_r = camera_info['image_path']+sequences[seq]+'image_3/'
                self.poses = camera_info['poses_path']+f'{seq:02}'+'.txt'
                self.seq_len = len[seq]
            else:
                self.path_l = camera_info['image_path_l']
                self.path_r = camera_info['image_path_r']
                self.poses = camera_info['poses_path']
                self.seq_len = camera_info['seq_len']    
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
                initial_matches = self.matcher.knnMatch(desL, desR, 2)
                initial_matches = self.computeRatioTest(initial_matches)
            else:
                initial_matches = self.matcher.match(desL, desR)
            
            epi_matches = self.epipolarFilter(initial_matches, kpL, kpR, filter_threshold=5.0)

        return epi_matches, kpL, kpR, desL, desR, imgL
    
    def semanticsFromImages(self, im_no, show_seg=False):
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

            #imgL_proc = cv.GaussianBlur(imgL, (5,5), 0)
            #imgR_proc = cv.GaussianBlur(imgR, (5,5), 0)

            # segment images and get the correct outputs:
            distL = self.seg_engine.segmentImageDist(imgL)
            distR = self.seg_engine.segmentImageDist(imgR)

            if show_seg: # Display segmentation results
                visL = self.seg_engine.segmentImageVis(imgL)
                cv.imshow("Left Image Segmentation", visL)
                cv.waitKey(0)
                cv.destroyAllWindows()
            return distL, distR
    
    def semanticsMaxFromImages(self, im_no):
        """
        Extracts semantic distributions from pair of stereo images, and then returns the argmax for each pixel

        Parameters:
        ----------
            im_no: ID Number of Image to Process
            segmentation_engine: Object that can process the semantic segmentation
        Returns:
        ----------
            segL: Argmax Matrix of left image
            imgL: RGB Color Matrix of left image
        """
        if self.cam_name != 'kitti':
            print("Behavior not implemented!")
            exit()
        else:
            image_str = f'{im_no:06}'
            imgL = cv.imread(self.path_l+image_str+".png", cv.IMREAD_UNCHANGED)
            imgL = cv.cvtColor(imgL, cv.COLOR_BGR2RGB)
            # imgL_proc = cv.GaussianBlur(imgL, (5,5), 0)
            # segment images and get the correct outputs:
            segL = self.seg_engine.segmentImageMax(imgL)
            return segL, imgL
    
    def semanticFilter(self, matches, kp_left, kp_right, dist_l, dist_r, filter_threshold=0.5):
        """
        Applies a filter to point cloud based on semantic similarity between the left and right points

        Parameters:
        ----------
            matches: list of matched points
            kp_left: key points from the left image
            kp_right: key points from the right image
            dist_l: semantic distribution from the left image
            dist_r: semantic distribution from the right image
            filter_threshold: Allowable tolerance for KL Divergence between points
                                Default value: 0.5
        
        Returns:
        ----------
            ret_matches: List of filtered matches
            left_locs: List of feature locations in left image
            right_locs: List of feature locations in right image
            left_dists: List of feature distributions in left image
            right_dists: List of feature distributions in right image

        """
        ret_matches = []
        left_dists = []
        right_dists = []
        for mat in matches:
            feat_l = kp_left[mat.queryIdx]
            feat_r = kp_right[mat.trainIdx]

            pd_l = self.computeFeatureDist(feat_l, dist_l)
            pd_r = self.computeFeatureDist(feat_r, dist_r)

            if pd_l.sum() == 0 or pd_r.sum()==0:
                continue

            div = self.fastKLDiv(pd_l, pd_r)
            if div < filter_threshold:
                ret_matches.append(mat)
                left_dists.append(pd_l)
                right_dists.append(pd_r)
        
        return ret_matches, left_dists, right_dists
        
    def fastKLDiv(self, p, q):
        """
        Computes the Symmetrised Kullback-Liebler Divergence between two semantic distributions

        Parameters:
        ----------
            p: distribution 1
            q: distribution 2
        
        Returns:
        ----------
            Symmetrised KL Divergence of p and q
        """
        divP = 0.0
        divQ = 0.0
        try:
            divP += np.multiply(np.log(np.divide(p,q)),p).sum()
        except:
            print("Q Error", q)

        try:
            divQ += np.multiply(np.log(np.divide(q,p)),q).sum()
        except:
            print("P Error", p)
        
        return divP + divQ
    
    def computeFeatureDist(self, feat, dist):
        """
        Approximates the region of semantic information around a point as a single distribution over the given classes

        Parameters:
        ----------
            feat: OpenCV Image feature object
            dist: Full Pixel-wise semantic distribution over the whole image
        
        Returns:
        ----------
            retDist: Normalised distribution vector for the point measurement
        """
        (x, y) = feat.pt
        x = np.int16(x)
        y = np.int16(y)
        x_range = [x-3, x+4]
        y_range = [y-3, y+4]
        
        retDist = np.zeros(150)

        if(x_range[0] < 0 or x_range[1] > dist.shape[2]):
            #print("X out of range", x_range[0], x_range[1])
            return retDist
        elif(y_range[0] < 0 or y_range[1] > dist.shape[1]):
            #print("Y out of Range", y_range[0], y_range[1])
            return retDist
        else:
            dist_slice = dist[:, y_range[0]:y_range[1], x_range[0]:x_range[1]]
            retDist = np.tensordot(dist_slice, FIXED_BLUR, axes=((1,2),(0,1)))
            return self.normalizeVect(retDist)
    
    def normalizeVect(self, vect):
        """
        Normalises a vector

        Parameters:
        ----------
            vect: input vector
        
        Returns:
        ----------
            mat: Normalised vector
        """
        matSum = np.sum(vect)
        mat = vect/matSum
        return mat

    def excludeDynamicLabels(self, points):
        """
        Removes points of with specific class labels from list of points

        Parameters:
        ----------
            points: list of unfiltered points
            labels: list of labels to exclude
        
        Returns:
        ----------
            filtered_points: list of points with specific labels removed
        """
        filtered_points = []
        for point in points:
            if point.label not in DYNAMIC_CLASSES:
                filtered_points.append(point)
        return filtered_points
    
    def excludeUncertainLabels(self, points):
        """
        Removes points of with specific class labels from list of points

        Parameters:
        ----------
            points: list of unfiltered points
            labels: list of labels to exclude
        
        Returns:
        ----------
            filtered_points: list of points with specific labels removed
        """
        filtered_points = []
        for point in points:
            if point.label not in UNCLEAR_CLASSES:
                filtered_points.append(point)
        return filtered_points

    def excludeDynamicLabelsMatrix(self, points, isSuper=False):
        filtered_points = []
        if not isSuper:
            for i in range(points.shape[0]):
                if points[i, 3] not in DYNAMIC_CLASSES:
                    filtered_points.append(points[i, :])
        else:
            for i in range(points.shape[0]):
                if points[i, 3] not in DYNAMIC_CLASSES_SUPER:
                    filtered_points.append(points[i, :])
        return np.asarray(filtered_points)

    def excludeUncertainLabelsMatrix(self, points, isSuper=False):
        filtered_points = []
        if not isSuper:
            for i in range(points.shape[0]):
                if points[i, 3] not in UNCLEAR_CLASSES:
                    filtered_points.append(points[i, :])
        else:
            for i in range(points.shape[0]):
                if points[i, 3] not in UNCLEAR_CLASSES_SUPER:
                    filtered_points.append(points[i, :])
        return np.asarray(filtered_points)
                
# ----- Function Definitions -----
def noiseMatrix(XL, YL, XR, YR):
    """
    Defines a noise matrix for the x and y directions of each image

    Parameters:
    ----------
        XL: standard deviation of the left image X noise.
        YL: standard deviation of the left image X noise.
        XR: standard deviation of the right image X noise.
        YR: standard deviation of the right image Y noise.

    """
    N = np.array([  [XL**2, 0, 0, 0],
                    [0, YL**2, 0, 0],
                    [0, 0, XR**2, 0],
                    [0, 0, 0, YR**2]])
    return N
        

# Code if executed as main

