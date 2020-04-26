#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
This script calculates and plots the preferred spatial frequencies of all the LGN cells, 
based on their spatial filter parameters.
Receptive field sizes are first loaded in for each LGN cell from a list 
of the "spatial size" parameter for each LGN cell. 
Here the list is contained in spatialfilter.csv, 
a subset of the LGN/LGN/lgn_full_col_cells_3.csv file (available from the LGN model). 
The preferred spatial frequencies are then computed for all LGN cells, 
based on the analytical formula for spatial frequency tuning of a DoG filter.

Parameters
----------
nu : spatial frequency
k: wavevector
Ac: amplitude of center Gaussian
As: amplitude of surround Gaussian
spatial_sizes: receptive field sizes 
               defined by the Allen model in 
               "LGN/LGN/lgn_full_col_cells_3.csv"
sigma_c: width of center Gaussian
         defined in the Allen model as the spatial size divided by 3
sigma_s: width of surround Gaussian
sc: scaling factor for determining sigma_s
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd


Ac = 1.73
As = 1*Ac
sc = 2.45

spatial_sizes = pd.read_csv('spatialfilter.csv', delimiter='\t')
sigma_c = spatial_sizes.values/3
sigma_s = sc*sigma_c

nu = np.arange(0, 1, 0.001)  
k = nu*2*np.pi  
mean_f = np.zeros(len(peaks),len(k))  # mean frequency response
peaks = np.zeros(len(sigma_c))  # preferred frequency

# Calculate spatial frequency tuning for each cell:
for i in range(len(peaks)):
    ind = 0
    
    for j in k:
        mean_f[i,ind] = (Ac*math.exp((-j**2 *sigma_c[i]**2)/2) 
                         - As*math.exp((-j**2 *(sigma_s[i])**2)/2))
        ind = ind + 1
    
    peaks_ind = np.argmax(mean_f[i,:])
    peaks[i] = nu[peaks_ind]

# Plot preferred frequencies:
plt.hist(peaks, 
         bins=250, 
         weights=np.zeros_like(peaks) + 1/peaks.size, 
         range=(0, 0.25))
plt.xlabel('Spatial frequency (cpd)')
plt.ylabel('Fraction of cells')

# Find median, minimum and maximum preferred frequency:
median = np.median(peaks)
minimum = np.amin(peaks)
maximum = np.amax(peaks)
