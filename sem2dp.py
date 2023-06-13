# Implementation of a modified M2DP algorithm that takes semantic information into account

# Class implementing SeM2DP Descriptor

import numpy as np
from sklearn.decomposition import PCA

# Parameters for M2DP
P = 4 # Azimuth angles [0, pi/p, 2pi/p ... pi] default 4
Q = 16 # Altitude angles [0, pi/2q, 2pi/2q ... pi/2] default 16
L = 8 # Number of concentric circles [l^2r, (l-1)^2r ... 2^2r, r] default 8
T = 16 # Number of bins per circle default 16

def createSemDescriptor(data, semantics, T=8, L=4, P=4, Q=16):
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
    S = GetSignatureMatrix(azimuthList, elevationList, data_rot, semantics, numT, numR, maxRho)
    return np.ravel(S).astype(np.uint8)
    
    # New, modified form of Signature
    # S = GetSignatureMatrixHisto(azimuthList, elevationList, data_rot, semantics, numT, numR, maxRho, 150)
    # u,s,vh = np.linalg.svd(S)
    # desM2DP  = np.concatenate((u[:,0], vh[0,:]))
    # return desM2DP  

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
            input_array = np.vstack((rho, theta, semantics))

            bins, edges = np.histogramdd(input_array.T, bins=(rhoList, thetaList, 150))
            bin_labs = np.argmax(bins, axis=2)

            labs = np.ravel(bin_labs)

            S[n,:] = labs
            n += 1
    
    return S

def sph2cart(ph, th, rad):
    x = rad*np.cos(ph)*np.cos(th)
    y = rad*np.cos(ph)*np.sin(th)
    z = rad*np.sin(ph)
    return np.array([x, y, z])

def cart2pol(x, y):
    rad = np.sqrt(x**2 + y**2)
    th = np.arctan2(x, y)
    return th, rad

def des_compress_new(descriptor):
    new_des = []
    for val, i in zip(descriptor, range(descriptor.shape[0])):
        if val != 0:
            new_des.append(i)
            new_des.append(val)
    return np.asarray(new_des), descriptor.shape[0]

def des_decompress_new(descriptor, des_size):
    new_des = np.zeros(des_size)
    indices = descriptor[::2]
    values = descriptor[1::2]
    new_des[indices] = values
    return new_des

def hamming_compare(base_descriptor, other_descriptors):
    n = other_descriptors.shape[0]
    base_descriptor = np.tile(base_descriptor,(n,1))
    
    dists = (base_descriptor != other_descriptors).sum(1) / other_descriptors.shape[1]
    
    return dists

def GetSignatureMatrixHisto(azimuthList, elevationList, data, semantics, numT, numR, maxRho, numS):
    S = np.zeros(((len(azimuthList)*len(elevationList)),((numT*numR)+(numS*numR))))
    n = 0

    thetaList = np.linspace(-np.pi,np.pi,numT+1)

    rhoList = np.linspace(0,np.sqrt(maxRho),numR+1)
    rhoList = rhoList**2
    rhoList[-1] = rhoList[-1] + 0.001

    semList = np.arange(0,numS+1)

    for azm in azimuthList:
        for elv in elevationList:
            vecN = sph2cart(azm,elv,1)

            h = np.matmul(np.array([1, 0, 0]), vecN.T)
            c = h*vecN

            px = np.array([1, 0, 0]) - c
            py = np.cross(vecN, px)
            pdata = np.transpose(np.array([data@px, data@py]))
            
            theta,rho = cart2pol(pdata[:,0], pdata[:,1])
            space_bin, r_edges, deg_edges = np.histogram2d(rho, theta, bins=(rhoList, thetaList))
            space_bin = np.ravel(space_bin)

            sem_bins, r_edges, s_edges = np.histogram2d(rho, semantics, bins=(rhoList, semList))
            sem_bins = np.ravel(sem_bins)

            combo = np.concatenate((space_bin, sem_bins), axis=0)

            S[n, :] = combo
            n += 1
    
    return S

def createSemDescriptorHisto(data, semantics, T=16, L=8, P=4, Q=16):
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

    # New, modified form of Signature
    S = GetSignatureMatrixHisto(azimuthList, elevationList, data_rot, semantics, numT, numR, maxRho, 150)
    u,s,vh = np.linalg.svd(S)
    desM2DP  = np.concatenate((u[:,0], vh[0,:]))
    return desM2DP

def GetSignatureMatrixHisto_Tune(azimuthList, elevationList, data, semantics, numT, numR, numC, numB, maxRho, numS,):
    S = np.zeros(((len(azimuthList)*len(elevationList)),((numT*numR)+(numS*numC*numB))))
    n = 0

    thetaList = np.linspace(-np.pi,np.pi,numT+1)
    binList = np.linspace(-np.pi,np.pi,numB+1)

    rhoList = np.linspace(0,np.sqrt(maxRho),numR+1)
    rhoList = rhoList**2
    rhoList[-1] = rhoList[-1] + 0.001

    circList = np.linspace(0,np.sqrt(maxRho),numC+1)
    circList = circList**2
    circList[-1] = circList[-1] + 0.001

    semList = np.arange(0,numS+1)

    for azm in azimuthList:
        for elv in elevationList:
            vecN = sph2cart(azm,elv,1)

            h = np.matmul(np.array([1, 0, 0]), vecN.T)
            c = h*vecN

            px = np.array([1, 0, 0]) - c
            py = np.cross(vecN, px)
            pdata = np.transpose(np.array([data@px, data@py]))
            
            theta,rho = cart2pol(pdata[:,0], pdata[:,1])
            space_bin, r_edges, deg_edges = np.histogram2d(rho, theta, bins=(rhoList, thetaList))
            space_bin = np.ravel(space_bin)

            input_array = np.vstack((rho, theta, semantics))
            sem_bins, edges = np.histogramdd(input_array.T, bins=(circList, binList, semList))
            sem_bins = np.ravel(sem_bins)

            combo = np.concatenate((space_bin, sem_bins), axis=0)

            S[n, :] = combo
            n += 1
    
    return S

def createSemDescriptorHisto_Tune(data, semantics, T=16, L=8, P=4, Q=16, C=8, B=1):
    data = np.asarray(data)
    semantics = np.asarray(semantics)

    numT = T
    numR = L
    numP = P
    numQ = Q
    numC = C
    numB = B

    data_rot = PCARotationInvariant(data)

    azimuthList = np.linspace(-np.pi/2, np.pi, numP)
    elevationList = np.linspace(0,np.pi/2, numQ)

    rho2 = np.sum(np.power(data_rot, 2), 1)
    maxRho = np.sqrt(max(rho2))

    # New, modified form of Signature
    S = GetSignatureMatrixHisto_Tune(azimuthList, elevationList, data_rot, semantics, numT, numR, numC, numB, maxRho, 150)
    u,s,vh = np.linalg.svd(S)
    desM2DP  = np.concatenate((u[:,0], vh[0,:]))
    return desM2DP