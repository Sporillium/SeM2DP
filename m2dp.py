# Implementation of Basic M2DP Algorithm
# Reference: Li He, Xiaolong Wang and Hong Zhang, "M2DP: A Novel 3D Point Cloud Descriptor and Its Application in Loop Closure Detection", IROS 2016.

# Class implementing M2DP global scene signature descriptor

# Import packages 
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

A = np.zeros((P*Q, T*L))
print(A.shape)

class m2dp:
    # Constructor
    def __init__(self, P=P, Q=Q, L=L, T=T):
        self.p = P
        self.q = Q
        self.l = L
        self.t = T
        self.A = np.zeros((P*Q, T*L))

        self.point_cloud = None
        self.mod_cloud = None

        self.centroid = None
        self.x_axis = None
        self.y_axis = None
    
    # Methods
    def plotAxes(self):
        """
        Plots the point cloud, centroid, and x- and y-axes from the PCA

        Parameters:
        ----------
            None

        Returns:
        ----------
            None
        """
        fig = plt.figure()
        ax4 = fig.add_subplot(111, projection='3d')
        ax4.set_aspect('auto')
        ax4.set_title("3D Projection of Points")
        ax4.set_xlabel("X Axis")
        ax4.set_ylabel("Y Axis")
        ax4.set_zlabel("Z Axis")
        
        x = []
        y = []
        z = []

        for points in self.mod_cloud:
            x.append(points[0])
            y.append(points[1])
            z.append(points[2])
            
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)

        ax4.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))  # aspect ratio is 1:1:1 in data space   
        ax4.plot(x, y, z, 'bo')
        ax4.plot(0, 0, 0, 'go')

        ax4.plot([0,10*self.x_axis[0]], [0,10*self.x_axis[1]], [0,10*self.x_axis[2]], 'r-')
        ax4.plot([0,10*self.y_axis[0]], [0,10*self.y_axis[1]], [0,10*self.y_axis[2]], 'y-')

    def plotCloud(self, points, proj_x, proj_y, norm):
            """
            Plots a given point cloud

            Parameters:
            ----------
                points: list of points given as 3D vectors

            Returns:
            ----------
                None
            """
            fig = plt.figure()
            ax4 = fig.add_subplot(111, projection='3d')
            ax4.set_aspect('equal')
            ax4.set_title("3D Projection of Points")
            ax4.set_xlabel("X Axis")
            ax4.set_ylabel("Y Axis")
            ax4.set_zlabel("Z Axis")
            
            x = []
            y = []
            z = []

            for point in points:
                x.append(point[0])
                y.append(point[1])
                z.append(point[2])
            
                
            x = np.asarray(x)
            y = np.asarray(y)
            z = np.asarray(z)

            #ax4.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))  # aspect ratio is 1:1:1 in data space   
            ax4.plot(x, y, z, 'bo')
            #ax4.plot(self.centroid[0], self.centroid[1], self.centroid[2], 'go')
            ax4.plot(0, 0, 0, 'ro')

            #ax4.plot([self.centroid[0],self.x_axis[0]], [self.centroid[1],self.x_axis[1]], [self.centroid[2],proj_x[2]], 'r-')
            #ax4.plot([self.centroid[0],5*proj_x[0]], [self.centroid[1],5*proj_x[1]], [self.centroid[2],5*proj_x[2]], 'r-')
            #ax4.plot([self.centroid[0],self.centroid[0]+proj_y[0]], [self.centroid[1],self.centroid[1]+proj_y[1]], [self.centroid[2],self.centroid[2]+proj_y[2]], 'y-')

            #ax4.plot([self.centroid[0],self.x_axis[0]], [self.centroid[1],self.x_axis[1]], [self.centroid[2],proj_x[2]], 'r-')
            ax4.plot([0,5*proj_x[0]], [0,5*proj_x[1]], [0,5*proj_x[2]], 'r-')
            ax4.plot([0,5*proj_y[0]], [0,5*proj_y[1]], [0,5*proj_y[2]], 'r-')
            ax4.plot([0,5*norm[0]], [0,5*norm[1]], [0,5*norm[2]], 'g-')

    def plotProjection(self, points, max_dist):
        """
        Plots a given point cloud

        Parameters:
        ----------
            points: list of points given as 3D vectors

        Returns:
        ----------
            None
        """
        fig = plt.figure()
        ax4 = fig.add_subplot(111,)
        ax4.set_aspect('equal')
        ax4.set_title("2D Projection of Points on Plane")
        ax4.set_xlabel("X Axis")
        ax4.set_ylabel("Y Axis")
        ax4.set_xlim(-max_dist, max_dist)
        ax4.set_ylim(-max_dist, max_dist)
        
        x = []
        y = []
        for point in points:
            x.append(point[0])
            y.append(point[1])    
        x = np.asarray(x)
        y = np.asarray(y)

        bins = np.linspace(0, np.pi*2, self.t+1)
        for b in bins:
            ax4.plot([0,max_dist*np.cos(b)], [0,max_dist*np.sin(b)], 'k-')

        # Draw Circles:
        r = max_dist/(self.l**2)
        for i in range(1, self.l, 1):
            rads = (i**2)*r
            circle = plt.Circle((0,0), rads, fill=False)
            ax4.add_artist(circle)
        max_circle = plt.Circle((0,0), max_dist, fill=False)

        ax4.add_artist(max_circle)

        ax4.plot(x, y, 'bo')
        ax4.plot(0, 0, 'ro')

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
        for point in self.point_cloud:
            loc = point.location
            tot.append(loc)
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
            for c,b in it.product(range(1, len(circles), 1), range(1, len(bins), 1)):
                in_bin = [p for p in flat_cloud if circles[c] >= p[0] > circles[c-1] and bins[b] >= p[1] > bins[b-1]]
                A[(th*self.q + ph), (c-1)*self.t + (b-1)] = len(in_bin)
        
        #print(A)
        u, s, vh = np.linalg.svd(A)
        print(u.shape)
        #print(u)
        print("\n\n")
        print(vh.shape)
        #print(vh)
        print("\n\n")
        d = np.concatenate((u.T[0,:], vh[0,:])).T
        print(d.shape)
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







                
        
    
            


