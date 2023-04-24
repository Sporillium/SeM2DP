# Functions for processing point clouds

# Package Imports
import numpy as np
import random

# Constants
SCALE_THRESHOLD = 0.99
PROBABILITY_THRESHOLD = 5

# Class Definitions
class CloudProcessor:
    def __init__(self, stereo_extractor):
        self.stereo_extractor = stereo_extractor
        self.cloud = None

    def loadCloud(self, base_cloud):
        self.cloud = base_cloud
    
    def processFrame(self, img, filter_unclear_classes=True, filter_dynamic_classes=False):
        """
        Extracts semantic features and computes 3D Point clouds for a single frame of an input stereo image

        Parameters:
        ----------
            img (int): ID of image/frame to process
            filter_unclear_classes (Boolean): Flag for filtering unclear or unmeasurable class points from the point cloud. 
                Default value: True
            filter_dynamic_classes (Boolean): Flag for filtering points on potentially dynamic objects from the point cloud.
                Default value: False
        
        Returns:
        ----------
            points (list): A list of extracted 3D Semantic points
        """
        distL, distR = self.stereo_extractor.semanticsFromImages(img)
        epi_mat, kpL, kpR, desL, desR = self.stereo_extractor.pointsFromImages(img)

        #print(type(distL))
        #print(distL.shape)

        sem_mat, left_dist, right_dist = self.stereo_extractor.semanticFilter(epi_mat, kpL, kpR, distL, distR)

        points = []
        for mat, i in zip(sem_mat, range(len(sem_mat))):
            featL = kpL[mat.queryIdx]
            featR = kpR[mat.trainIdx]
            descL = desL[mat.queryIdx]
            descR = desR[mat.trainIdx]

            (xl, yl) = featL.pt
            (xr, yr) = featR.pt
            dist = left_dist[i]

            mean, cov = self.stereo_extractor.measurement3DNoise(xl, yl, xr, yr)
            point = MeasuredPoint(mean, cov, sem_dist=dist, left_descriptor=descL, right_descriptor=descR)
            points.append(point)
        
        #print("Total points before filtering: ", len(points))
        if filter_unclear_classes:
            points = self.stereo_extractor.excludeUncertainLabels(points)
        
        if filter_dynamic_classes:
            points = self.stereo_extractor.excludeDynamicLabels(points)
        
        #print("Total points after filtering: ", len(points))

        point_locs = np.zeros((len(points), 3))
        semantics = np.zeros((len(points)))
        for point, i in zip(points, range(len(points))):
            point_locs[i, :] = point.location
            semantics[i] = point.label

        return point_locs, semantics
    
    def processFrameNoSemantics(self, img):
        """
        Extracts 2D point features and computes 3D Point clouds for a single frame of an input stereo image

        Parameters:
        ----------
            img (int): ID of image/frame to process

        Returns:
        ----------
            points (list): A list of extracted 3D Semantic points
        """
        epi_mat, kpL, kpR, desL, desR = self.stereo_extractor.pointsFromImages(img)

        points = []
        for mat in epi_mat:
            featL = kpL[mat.queryIdx]
            featR = kpR[mat.trainIdx]
            descL = desL[mat.queryIdx]
            descR = desR[mat.trainIdx]

            (xl, yl) = featL.pt
            (xr, yr) = featR.pt

            mean, cov = self.stereo_extractor.measurement3DNoise(xl, yl, xr, yr)
            point = MeasuredPoint(mean, cov, sem_dist=None, left_descriptor=descL, right_descriptor=descR)
            points.append(point)
        
        return points
    
    def processPoseChange(self, prev_cloud, curr_cloud):
        """
        Estimates the pose change between two consecutive images in a sequence of stereo images

        Parameters:
        ----------
            prev_cloud (list): List of points in the point cloud from the previous time step
            curr_cloud (list): List of points in the point cloud from the current time step
        
        Returns:
        ----------
            R_best (matrix): Rotation Matrix that best fits the point correspondences
            t_best (vector): Translation vector that best fits the point correspondences
            non_corres_points (list): List of points with no corresponding points
        """
        des_L_prev = [point.desc_l for point in prev_cloud]
        des_R_prev = [point.desc_r for point in prev_cloud]

        des_L_curr = [point.desc_l for point in curr_cloud]
        des_R_curr = [point.desc_r for point in curr_cloud]

        x_bar = [point.mean for point in prev_cloud]
        y_bar = [point.mean for point in curr_cloud]

        C = [point.cov for point in prev_cloud]
        P = [point.cov for point in curr_cloud]

        if self.stereo_extractor.detectorID != 'ORB':
            initial_matches_L = self.stereo_extractor.matcher.knnMatch(np.asarray(des_L_prev), np.asarray(des_L_curr), 2)
            initial_matches_R = self.stereo_extractor.matcher.knnMatch(np.asarray(des_R_prev), np.asarray(des_R_curr), 2)

            initial_matches_L = self.stereo_extractor.computeRatioTest(initial_matches_L)
            initial_matches_R = self.stereo_extractor.computeRatioTest(initial_matches_R)
        else:
            initial_matches_L = self.stereo_extractor.matcher.match(np.asarray(des_L_prev), np.asarray(des_L_curr))
            initial_matches_R = self.stereo_extractor.matcher.match(np.asarray(des_R_prev), np.asarray(des_R_curr))
        
        corres_L = np.zeros(len(des_L_prev))
        corres_R = np.zeros(len(des_R_prev))
        final_corr = np.zeros(min(len(des_L_prev), len(des_R_prev)))

        for matL in initial_matches_L:
                corres_L[matL.queryIdx] = matL.trainIdx
        for matR in initial_matches_R:
                corres_R[matR.queryIdx] = matR.trainIdx
        
        for i in range(len(final_corr)):
            if len(final_corr == len(des_L_prev)):
                final_corr[i] = corres_L[i] if np.equal(corres_L, corres_R)[i] else 0
            else:
                final_corr[i] = corres_R[i] if np.equal(corres_L, corres_R)[i] else 0

        c_best, R_best, T_best = probabilistic_outlier_removal(np.asarray(x_bar), np.asarray(C), 
                                                               np.asarray(y_bar), np.asarray(P), 
                                                               np.asarray(final_corr, dtype=np.int0), 
                                                               prob_func_chiu_log, num_iters=1000)
        ii = np.where(c_best == 0)[0]
        non_corres_points = [prev_cloud[i] for i in ii]

        return np.column_stack((R_best, T_best.T)), non_corres_points

    def merge_point_cloud(self, cloud_list, new_points, transform=None):
        """
        Extends point cloud with new points, and allows for transforming of points to new locations

        Parameters:
        ----------
            cloud_list (list): Main point cloud
            new_points (list): List of points to be added to the point cloud
            transform (Matrix): Transformation matrix to transform new points to a different location
                Default value: None
        
        Returns:
        ----------
            None
        """

        cloud_list.extend(new_points)
        if transform is not None:
            for point in cloud_list:
                point.transformLocation(transform)
        

class MeasuredPoint:
    """
    Initialises a measuredPoint Object, which stores information about the 3D position and classes of points extracted from stereo images.

    Parameters:
    ----------
        mean (vector): Mean location of point

        cov (matrix): Covariance of measured point location

        sem_dist (vector): Semantic distribution over the classes for the point
            Default value: None

        left_descriptor: Descriptor of the point component of the left image point
            Default value: None

        right_descriptor: Descriptor of the point component of the right image point
            Default value: None 

    Instance Variables:
    ----------
        mean: Vector
        cov: Matrix

        sem_dist: Vector
        label: Int
        
        desc_l: Descriptor
        desc_r: Descriptor

        location: Vector
        location_cov: Matrix
        
        frame_num: Int
        cluster: Int
               
    Returns:
    ----------
        Instance of MeasuredPoint object
    """
    def __init__(self, mean, cov, sem_dist=None, left_descriptor=None, right_descriptor=None):
        self.mean = mean
        self.cov = cov
        self.sem_dist = sem_dist
        self.label = np.argmax(sem_dist) if sem_dist is not None else None
        self.desc_l = left_descriptor
        self.desc_r = right_descriptor
        self.location = mean
        self.location_cov = cov
        self.frame_num = None
        self.cluster = None
    
    def transformLocation(self, transform):
        self.location = (transform[:, 0:3].T @ self.location) + transform[:, 3]
        self.location_cov = transform[:, 0:3].T @ self.location_cov @ transform[:, 0:3]


# Function Definitions
# ----- Implementation of Umeyama's Transformation Estimation Algorithm -----
def umeyama(src_points, dst_points):
    """
    Estimate Transformation Parameters between two sets of points in N-D space

    Parameters:
    ----------
        src_points: (M, N) array
            Coordinates of M source points in N-Dimensional Space
        dst_points: (M, N) array
            Corresponding M destination points in N-Dimensional Space
    
    Returns:
    ----------
        R: (N, N) array
            Least-squares rotational matrix between the sets of points
        t: (1, N) array
            Least squares translation vector between the sets of points
        c: float
            Least squares scaling parameter between the sets of points
    
    References:
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
    """
    num = src_points.shape[0]
    dim = src_points.shape[1]

    # Compute Mean of src_points and dist_points
    src_mean = src_points.mean(axis=0)
    dst_mean = dst_points.mean(axis=0)

    # Subtract the mean from src_points and dst_points
    src_demean = src_points - src_mean
    dst_demean = dst_points - dst_mean
    
    # Eq. (38). Covariance of src and dst
    A = np.matmul(dst_demean.T, src_demean) / num

    # Eq. (39).
    S = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        S[dim - 1] = -1
    
    # Create empty matrices
    R = np.eye(dim, dtype=np.double)
    
    # Compute the SVD of the Covariance Matrix
    U, D, V = np.linalg.svd(A)

    # Compute Transformation Parameters:
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan*R, 0, 0

    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            R = np.matmul(U, V)
        else:
            temp = S[dim - 1]
            S[dim - 1] = -1
            R = np.matmul(U, np.matmul(np.diag(S), V))
            S[dim - 1] = temp
    else:
        R = np.matmul(U, np.matmul(np.diag(S), V))

    # Estimate scale (Eq. (41) and (42))
    c = (1.0 / src_demean.var(axis=0).sum()) * np.matmul(D, S)
    T = dst_mean - c*(R @ src_mean)
    return R, T, c

# ----- Implementation of Brink's Probabilistic Outlier Removal -----
def probabilistic_outlier_removal(x, C, y, P, c, prob_func, num_iters=1000, return_transform=True):
    """
    Modified Implementation of RANSAC that makes use of Probabilistic associations between random samples

    Parameters:
    ---------
        x: (N, M) array
            Array of M-dimensional points of length N
        C: (N, (M, M)) array
            Array of MxM Covariance Matrices for N points given in x
        y: (N, M) array
            Array of Corresponding M-dimensional points of length N
        P: (N, (M, M)) array
            Array of MxM covariance Matrices for N points given in y
        c: (N, 1) array
            List containing correspondences between x and y, such that if c[i] = j,
            x[i] = y[j]. An entry of 0 indicates that there is no match for that
            specific point in x.
        prob_func: Function Object
            Chosen probability function for computing inlier association probability
        num_iters: Int
            Number of iterations to process, defaults to 1000
        return_transform: Boolean
            Flag to return the rotation matrix R and translation vector t corresponding
            to the best match from the outlier removal step. Defaults to True

    Returns:
    ---------
        c_best: (N, 1) array
            List containing correspondences from model containing the most inlier points
        R_best: (N, N) matrix
            Rotation matrix corresponding to best inlier count
        t_best: (N, 1) array
            Translation vector corresponding to best inlier count
    """
    best_inlier_count = 0
    c_best = None
    R_best = None
    T_best = None

    for it in range(num_iters):

        cor_product = 0
        while cor_product == 0:
            samples = np.asarray(random.sample(range(len(c)), 3))
            cor_product = np.prod(c[samples])
        
        x_sample = x[samples]
        y_sample = y[c[samples]]

        R, T, s = umeyama(x_sample, y_sample)
        inlier_count = 0
        c_temp = np.copy(c)

        if s > SCALE_THRESHOLD:
            for i in range(len(c_temp)):
                j = c_temp[i]
                if j > 0:
                    x_temp = (R @ x[i]) + T
                    C_temp = R @ C[i] @ R.T
                    p = prob_func(x_temp, C_temp, y[j], P[j])
                    if p < PROBABILITY_THRESHOLD:
                        inlier_count += 1
                    else:
                        c_temp[i] = 0

            if inlier_count > best_inlier_count:
                c_best = np.copy(c_temp)
                R_best = R
                T_best = T
                best_inlier_count = inlier_count
    
    if return_transform:
        return c_best, R_best, T_best
    else:
        return c_best

# ----- Implementation of Brink's improved Consensus measure ------
def prob_func_brink(x, C, y, P):
    """
    Function that determines the probability that two samples are of the same point. Method used by Brink (2012)

    Parameters:
    ---------
        x: (1, M) vector
            A M-dimensional mean vector for sample 1
        C: (M, M) array
            MxM Covariance Matrix for sample 1
        y: (1, M) array
            A M-dimensional mean vector for sample 2
        P: (M, M) array
            MxM Covariance Matrix for sample 2

    Returns:
    ---------
        prob: float
            Probability that x and y are measurements of the same point
    """
    C_inv = np.linalg.inv(C)
    C_det = np.linalg.det(C)

    P_inv = np.linalg.inv(P)
    P_det = np.linalg.det(P)

    cov = C_inv + P_inv
    cov_det = np.linalg.det(cov)

    s_1 = x.T @ C_inv @ x
    s_2 = y.T @ P_inv @ y
    s_3 = (x.T@C_inv + y.T@P_inv) @ cov @ (C_inv@x + P_inv@y)

    s = s_1 + s_2 - s_3

    prob = ( ( np.sqrt(C_det)*np.sqrt(P_det)) / np.sqrt(cov_det)) * np.exp(-0.5*s)

    return prob

# ----- Implementation of Chiu's Improved Consensus measure ------
def prob_func_chiu(x, C, y, P):
    """
    Function that determines the probability that two samples are of the same point. Method used by Chiu (2017)

    Parameters:
    ---------
        x: (1, M) vector
            A M-dimensional mean vector for sample 1
        C: (M, M) array
            MxM Covariance Matrix for sample 1
        y: (1, M) array
            A M-dimensional mean vector for sample 2
        P: (M, M) array
            MxM Covariance Matrix for sample 2

    Returns:
    ---------
        prob: float
            Probability that x and y are measurements of the same point
    """
    cov = C + P
    cov_inv = np.linalg.inv(cov)
    cov_det = np.linalg.det(cov)

    diff = x - y

    s = diff.T @ cov_inv @ diff

    prob = (1/np.sqrt(cov_det)) * np.exp(-0.5*s)

    return prob

# ----- Implementation of Chiu's Improved Consensus measure Neg Log Likelyhood ------
def prob_func_chiu_log(x, C, y, P):
    """
    Function that determines the negative log likelihood that two samples are of the same point. Method used by Chiu (2017)

    Parameters:
    ---------
        x: (1, M) vector
            A M-dimensional mean vector for sample 1
        C: (M, M) array
            MxM Covariance Matrix for sample 1
        y: (1, M) array
            A M-dimensional mean vector for sample 2
        P: (M, M) array
            MxM Covariance Matrix for sample 2

    Returns:
    ---------
        meas: float
            Probability that x and y are measurements of the same point, expressed as a negative log likelihood
    """
    cov = C + P
    cov_inv = np.linalg.inv(cov)
    cov_det = np.linalg.det(cov)

    diff = x - y

    meas = (diff.T @ cov_inv @ diff) + np.log(cov_det)

    return meas
