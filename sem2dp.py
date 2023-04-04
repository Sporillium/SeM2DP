# Implementation of a modified M2DP algorithm that takes semantic information into account

# Class implementing SeM2DP Descriptor

import numpy as np
import math
import matplotlib.pyplot as plt
import itertools as it
from sklearn.decomposition import PCA

# Parameters for M2DP
P = 4 # Azimuth angles [0, pi/p, 2pi/p ... pi] default 4
Q = 16 # Altitude angles [0, pi/2q, 2pi/2q ... pi/2] default 16
L = 8 # Number of concentric circles [l^2r, (l-1)^2r ... 2^2r, r] default 8
T = 16 # Number of bins per circle default 16

class sem2dp:
    # Constructor
    def __init__(self, P=P, Q=Q, L=L, T=T):
        self.p = P
        self.q = Q
        self.l = L
        self.t = T
        self.A = np.zeros((P*Q, 2*T*L))

        self.point_cloud = None
        self.mod_cloud = None

        self.centroid = None
        self.x_axis = None
        self.y_axis = None
    
    # Methods
    def readCloud(self, cloud):
        """
        Reads in point external point cloud

        Parameters:
        ----------
            point_cloud: Array of 3D points expressed in (x, y, z)

        Returns:
        ----------
            None
        """
        self.point_cloud = cloud

    def centroidPointCloud(self):
        """
        Calulates the Centroid of the Point Cloud, and shifts the origin to the centroid

        Parameters:
        ----------
            None

        Returns:
        ----------
            None
        """
        x = []
        y = []
        z = []
        tot = []
        labs = []
        for point in self.point_cloud:
            loc = point.location
            lab = point.label
            tot.append(loc)
            labs.append(lab)
            x.append(loc[0])
            y.append(loc[1])
            z.append(loc[2])
        
        x_cent = sum(x)/len(self.point_cloud)
        y_cent = sum(y)/len(self.point_cloud) 
        z_cent = sum(z)/len(self.point_cloud)

        self.centroid = np.array([x_cent, y_cent, z_cent])
        
        for i in range(len(tot)):
            tot[i] = tot[i] - self.centroid
        
        self.mod_cloud = tot
        self.label_map = labs
        #print(self.label_map)
    
    def cloudAxes(self):
        """
        Performs PCA using scikit_learn. Creates the axes used by M2DP using the 1st and 2nd PC's

        Parameters:
        ----------
            None

        Returns:
        ----------
            None
        """
        pca = PCA(n_components=3)
        pca.fit(self.mod_cloud)

        self.x_axis = pca.components_[(0)]
        self.y_axis = pca.components_[(1)]
    
    def projectPoints(self):
        """
        Projects the points from the point cloud onto a series of planes

        Parameters:
        ----------
            None

        Returns:
        ----------
            None
        """
        theta = np.linspace(0, np.pi, self.p)
        phi = np.linspace(0, np.pi/2, self.q)

        for th in theta:
            for ph in phi:
                x = np.cos(ph)*np.cos(th)
                y = np.cos(ph)*np.sin(th)
                z = np.sin(ph)

                norm = np.array([x, y, z])
                print(norm)

                dist_x = np.dot(self.x_axis, norm)
                proj_x = self.x_axis - (dist_x*norm)
                
                # Construct Perpendicular Axis
                proj_y = np.cross(proj_x, norm)

                # Construct basis vectors:
                norm_x = proj_x/np.linalg.norm(proj_x)
                norm_y = proj_y/np.linalg.norm(proj_y)

                proj_cloud = []
                for point in self.mod_cloud:
                    dist = np.dot(point, norm)
                    proj = point - (dist*norm)
                    proj_cloud.append(proj)
                
                flat_cloud = []
                for point in proj_cloud:
                    point_x = np.dot(point, norm_x)
                    point_y = np.dot(point, norm_y)
                    flat_cloud.append(np.array([point_x, point_y]))

                max_dist = 0
                for point in flat_cloud:
                    dist = math.dist([0,0], point)
                    if dist > max_dist:
                        max_dist = dist
                
                print(max_dist)

                #self.plotCloud(proj_cloud, proj_x, proj_y, norm)
                self.plotProjection(flat_cloud, max_dist)
                plt.show()
    
    def calculateDescriptor(self):
        """
        Projects the points from the point cloud onto a series of planes, and calculates the 
        M2DP descriptor from the various projections

        Parameters:
        ----------
            None

        Returns:
        ----------
            d: Vector of values that serves as the descriptor of the scene, of shape [(P*Q + L*T), 1]
        """
        theta = np.linspace(0, np.pi, self.p)
        phi = np.linspace(0, np.pi/2, self.q)
        bins = np.linspace(0, np.pi*2, self.t+1)

        for th,ph in it.product(range(len(theta)), range(len(phi))):
                
            # Construct Surface normal
            x = np.cos(phi[ph])*np.cos(theta[th])
            y = np.cos(phi[ph])*np.sin(theta[th])
            z = np.sin(phi[ph])
            norm = np.array([x, y, z])

            dist_x = np.dot(self.x_axis, norm)
            proj_x = self.x_axis - (dist_x*norm)
            
            # Construct Perpendicular Axis
            proj_y = np.cross(proj_x, norm)

            # Construct basis vectors:
            norm_x = proj_x/np.linalg.norm(proj_x)
            norm_y = proj_y/np.linalg.norm(proj_y)

            # Project cloud onto plane and change basis to planar coordinates
            flat_cloud = []
            for point in self.mod_cloud:
                dist = np.dot(point, norm)
                proj = point - (dist*norm)
                point_x = np.dot(proj, norm_x)
                point_y = np.dot(proj, norm_y)
                point_rad = math.dist([0,0],[point_x, point_y])
                point_theta = math.atan2(point_x, point_y)
                flat_cloud.append(np.array([point_rad, point_theta]))

            # Find the furthest point from Origin, and define biggest circle
            max_dist = 0
            for point in flat_cloud:
                if point[0] > max_dist:
                    max_dist = point[0]
            
            # Define circles
            r = max_dist/(self.l**2)
            circles = [0]
            for i in range(1, self.l, 1):
                rads = (i**2)*r
                circles.append(rads)
            circles.append(max_dist)

            # for c in range(1, len(circles), 1):
            #     for b in range(1, len(bins), 1):
            bin_labs = []
            list_counts = []
            list_labs = []
            for c,b in it.product(range(1, len(circles), 1), range(1, len(bins), 1)):
                in_bin = [p for p in flat_cloud if circles[c] >= p[0] > circles[c-1] and bins[b] >= p[1] > bins[b-1]]
                for i in range(len(flat_cloud)): 
                    if (circles[c] >= flat_cloud[i][0] > circles[c-1]) and (bins[b] >= flat_cloud[i][1] > bins[b-1]):
                        lab = self.label_map[i]
                        bin_labs.append(lab)
                    
                if len(bin_labs) == 0:
                    label = 0
                else:
                    label = mostFrequent(bin_labs)
                
                list_counts.append(len(in_bin))
                list_labs.append(label)
            combo_list = [None]*(len(list_counts)+len(list_labs))
            combo_list[::2] = list_counts
            combo_list[1::2] = list_labs
            combo = np.asarray(combo_list)
            self.A[(th*self.q + ph), :] = combo
        
        #print(A)
        u, s, vh = np.linalg.svd(self.A)
        #print(u.shape)
        #print(u)
        #print("\n\n")
        #print(vh.shape)
        #print(vh)
        #print("\n\n")
        d = np.concatenate((u.T[0,:], vh[0,:])).T
        #print(d.shape)
        return d
    
    def extractAndProcess(self, point_cloud):
        """
        Makes a general call to all processing methods in the M2DP class, and processes a given point cloud into a descriptor 

        Parameters:
        ----------
            point_cloud: Array or list of points in point cloud with 3D coordinates

        Returns:
        ----------
            descriptor: Vector of values that serves as the descriptor of the scene, of shape [(P*Q + L*T), 1]
        """
        # Read the point cloud into the object memory
        self.readCloud(point_cloud)

        # Calculate the centroid of the point cloud
        self.centroidPointCloud()

        # Define the principle axes of the point cloud using PCA
        self.cloudAxes()

        # Generate the descriptor using planar projection, and return
        descriptor = self.calculateDescriptor()
        return descriptor

def mostFrequent(List):
    return max(set(List), key=List.count)