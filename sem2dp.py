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