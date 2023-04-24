# Implementation of a modified M2DP algorithm that takes semantic information into account

# Class implementing SeM2DP Descriptor

import numpy as np
from sklearn.decomposition import PCA
#from m2dp import createDescriptor

# Parameters for M2DP
P = 4 # Azimuth angles [0, pi/p, 2pi/p ... pi] default 4
Q = 16 # Altitude angles [0, pi/2q, 2pi/2q ... pi/2] default 16
L = 8 # Number of concentric circles [l^2r, (l-1)^2r ... 2^2r, r] default 8
T = 16 # Number of bins per circle default 16

def createSemDescriptor(data, semantics, T=16, L=8, P=4, Q=16):
    # Seperate semantic labels from 3D Points:

    data = np.asarray(data)
    semantics = np.asarray(semantics)

    numT = T
    numR = L
    numP = P
    numQ = Q

    data_rot = PCARotationInvariant(data)

    azimuthList = np.linspace(-np.pi/2, np.pi, numP)
    elevationList = np.linspace(0,np.pi/2, numQ)

    rho2 = np.sum(np.power(data_rot, 2), 1)
    maxRho = np.sqrt(max(rho2))

    # Reintroduce the semantic information here
    A, S = GetSignatureMatrix(azimuthList, elevationList, data_rot, semantics, numT, numR, maxRho)
    u,s,vh = np.linalg.svd(A)
    desM2DP  = np.concatenate((u[:,0], vh[0,:]))

    return desM2DP, np.ravel(S).astype(np.uint8)

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

def GetSignatureMatrix(azimuthList, elevationList, data, semantics, numT, numR, maxRho):
    A = np.zeros((len(azimuthList)*len(elevationList),numT*numR))
    S = np.zeros((len(azimuthList)*len(elevationList),numT*numR))
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

            #print(theta.shape, rho.shape, semantics.shape)

            input_array = np.vstack((rho, theta, semantics))

            #print(input_array.shape)
 
            bins, edges = np.histogramdd(input_array.T, bins=(rhoList, thetaList, 150))
            #print(bins.shape)
            proj_bins = np.sum(bins, axis=2)
            #print(proj_bins.shape)

            bin_labs = np.argmax(bins, axis=2)

            bin = np.ravel(proj_bins)
            labs = np.ravel(bin_labs)

            A[n, :] = bin
            S[n,:] = labs
            n += 1
    
    return A, S

def sph2cart(ph, th, rad):
    x = rad*np.cos(ph)*np.cos(th)
    y = rad*np.cos(ph)*np.sin(th)
    z = rad*np.sin(ph)
    return np.array([x, y, z])

def cart2pol(x, y):
    rad = np.sqrt(x**2 + y**2)
    th = np.arctan2(x, y)
    return th, rad

def des_compress(descriptor):
    new_des = []
    zero_flag = False
    count = 0
    for val in descriptor:
        if val != 0 and zero_flag == False:
            new_des.append(val)

        elif val != 0 and zero_flag == True:
            zero_flag = False
            new_des.append(count)
            count = 0
            new_des.append(val)

        elif val == 0 and zero_flag == False:
            zero_flag = True
            new_des.append(256)
            count += 1

        elif val == 0 and zero_flag == True:
            count += 1
    
    return np.asarray(new_des)

def des_descompress(descriptor):
    new_des = np.zeros(8192)
    zero_flag = False
    pointer = 0
    for val in descriptor:
        if val != 256 and zero_flag == False:
            new_des[pointer] = val
            pointer += 1
        
        elif val != 256 and zero_flag == True:
            pointer += val
            zero_flag = False

        elif val == 256 and zero_flag == False:
            zero_flag = True
    
    return new_des
 
            