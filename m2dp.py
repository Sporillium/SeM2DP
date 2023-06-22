# Implementation of Basic M2DP Algorithm
# Reference: Li He, Xiaolong Wang and Hong Zhang, "M2DP: A Novel 3D Point Cloud Descriptor and Its Application in Loop Closure Detection", IROS 2016.

# Class implementing M2DP global scene signature descriptor

# Import packages 
import numpy as np
from sklearn.decomposition import PCA

# Parameters for M2DP
P = 4 # Azimuth angles [0, pi/p, 2pi/p ... pi] default 4
Q = 16 # Altitude angles [0, pi/2q, 2pi/2q ... pi/2] default 16
L = 8 # Number of concentric circles [l^2r, (l-1)^2r ... 2^2r, r] default 8
T = 16 # Number of bins per circle default 16

def createDescriptor(data, T=16, L=8, P=4, Q=16):
    data = np.asarray(data)

    numT = T
    numR = L
    numP = P
    numQ = Q

    data_rot = PCARotationInvariant(data)

    azimuthList = np.linspace(-np.pi/2, np.pi, numP)
    elevationList = np.linspace(0,np.pi/2, numQ)

    rho2 = np.sum(np.power(data_rot, 2), 1)
    maxRho = np.sqrt(max(rho2))

    A = GetSignatureMatrix(azimuthList, elevationList, data_rot, numT, numR, maxRho)
    u,s,vh = np.linalg.svd(A)
    desM2DP  = np.concatenate((u[:,0], vh[0,:]))

    return desM2DP

def PCARotationInvariant(data):
    n = data.shape[0]

    md = np.mean(data, 0)

    data = data - np.tile(md,(n,1))

    pca = PCA(n_components=3)
    pca.fit(data)

    x_component = pca.components_[0, :]
    y_component = pca.components_[1, :]
    z_component = pca.components_[2, :]

    X = np.matmul(x_component, data.T)
    Y = np.matmul(y_component, data.T)
    Z = np.matmul(z_component, data.T)

    data_trans = np.array([X, Y, Z])
    
    return data_trans.T

def GetSignatureMatrix(azimuthList, elevationList, data, numT, numR, maxRho):
    A = np.zeros((len(azimuthList)*len(elevationList),numT*numR))
    n = 0

    thetaList = np.linspace(-np.pi,np.pi,numT+1)

    rhoList = np.linspace(0,np.sqrt(maxRho),numR+1)
    rhoList = rhoList**2
    rhoList[-1] = rhoList[-1] + 0.001

    for azm in azimuthList:
        for elv in elevationList:
            vecN = sph2cart(azm,elv,1)

            h = np.matmul(np.array([1, 0, 0]), vecN.T)
            c = h*vecN

            px = np.array([1, 0, 0]) - c
            py = np.cross(vecN, px)

            pdata = np.transpose(np.array([data@px, data@py]))
            
            theta,rho = cart2pol(pdata[:,0], pdata[:,1])

            bin, r_edges, deg_edges = np.histogram2d(rho, theta, bins=(rhoList, thetaList))
            bin = np.ravel(bin)

            A[n, :] = bin
            n += 1
    
    return A

def sph2cart(ph, th, rad):
    x = rad*np.cos(ph)*np.cos(th)
    y = rad*np.cos(ph)*np.sin(th)
    z = rad*np.sin(ph)
    return np.array([x, y, z])

def cart2pol(x, y):
    rad = np.sqrt(x**2 + y**2)
    th = np.arctan2(x, y)
    return th, rad

def createColorDescriptor(data, T=16, L=8, P=4, Q=16, C=16):
    data = np.asarray(data)
    pos_data = data[:, :3]
    col_data = data[:, 4:]

    numT = T
    numR = L
    numP = P
    numQ = Q
    numC = C

    data_rot = PCARotationInvariant(pos_data)

    azimuthList = np.linspace(-np.pi/2, np.pi, numP)
    elevationList = np.linspace(0,np.pi/2, numQ)

    rho2 = np.sum(np.power(data_rot, 2), 1)
    maxRho = np.sqrt(max(rho2))

    A = GetSignatureMatrixColor(azimuthList, elevationList, data_rot, numT, numR, maxRho, col_data, numC)
    u,s,vh = np.linalg.svd(A)
    desM2DP  = np.concatenate((u[:,0], vh[0,:]))

    return desM2DP

def GetSignatureMatrixColor(azimuthList, elevationList, pos_data, numT, numR, maxRho, col_data, numC):
    A = np.zeros((len(azimuthList)*len(elevationList),(numT*numR)+(numC*3*numR)))
    n = 0

    thetaList = np.linspace(-np.pi,np.pi,numT+1)

    rhoList = np.linspace(0,np.sqrt(maxRho),numR+1)
    rhoList = rhoList**2
    rhoList[-1] = rhoList[-1] + 0.001

    colList = np.linspace(0,256,numC+1)
    col_R = col_data[:, 0]
    col_G = col_data[:, 1]
    col_B = col_data[:, 2]

    for azm in azimuthList:
        for elv in elevationList:
            vecN = sph2cart(azm,elv,1)

            h = np.matmul(np.array([1, 0, 0]), vecN.T)
            c = h*vecN

            px = np.array([1, 0, 0]) - c
            py = np.cross(vecN, px)

            pdata = np.transpose(np.array([pos_data@px, pos_data@py]))
            
            theta,rho = cart2pol(pdata[:,0], pdata[:,1])

            space_bin, r_edges, deg_edges = np.histogram2d(rho, theta, bins=(rhoList, thetaList))
            space_bin = np.ravel(space_bin)

            R_bins, r_edges, cr_edges = np.histogram2d(rho, col_R, (rhoList, colList))
            G_bins, r_edges, cg_edges = np.histogram2d(rho, col_G, (rhoList, colList))
            B_bins, r_edges, cb_edges = np.histogram2d(rho, col_B, (rhoList, colList))

            col_bins = np.hstack((R_bins, G_bins, B_bins))
            col_bins = np.ravel(col_bins)

            A[n, :] = np.concatenate((space_bin, col_bins), axis=0)
            n += 1
    
    return A
                
        
    
            


