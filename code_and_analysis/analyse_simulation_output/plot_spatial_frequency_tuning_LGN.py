#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameters
----------
neuron_id: neuron to be analysed
simlength: length of simulation (ms)
tsteps_per_sec: temporal resolution (time steps per seconds)
tf: temporal frequency (Hz)
cpd_array: set of spatial frequencies for the tuning curve
normalise_rates: set to 'True' is rates should be normalised
"""


import h5py
import numpy as np
import matplotlib.pyplot as plt


neuron_id = 8018  
simlength = 3000 
tsteps_per_sec = 1000
tf = 4
cpd_array = np.array([0.005,0.04,0.08,0.12,0.16,0.2,0.24,0.28,0.32])
normalise_rates = False


#Firing rates (F0 and F1) for each stimuli size:
mean_F0 = np.zeros(len(cpd_array))
mean_F1 = np.zeros(len(cpd_array))


for i in range(len(cpd_array)):
    f = h5py.File('LGN/m1/%scpd/fullfield_3s_0d_0.5ps_%dtf.h5' \
                  % (cpd_array[i], tf), 'r')      
    fr = f['firing_rates_Hz']
    t = f['time']
    fr = np.asarray(fr)
    t = np.asarray(t)
    time = np.arange(len(fr[1, :]))
    ids = np.arange(len(fr[:, 1]))
            
    
    # Set measurement window:
    interval = 1000
    period_start = int(simlength/2) 
    period_end = period_start + interval
        
    
    # Calculate F1:
    signal = fr[neuron_id,period_start:period_end]
    ff = np.fft.fft(signal)   
    ff_freq = np.fft.fftfreq(signal.size)*tsteps_per_sec 
    amplitude_spectum = np.abs(ff)*2/(period_end - period_start)
    amp_ind = np.where(ff_freq == tf) 
    mean_F1[i] = amplitude_spectum[amp_ind] 
        
        
    # Calculate F0:
    mean_F0[i] = np.mean(fr[neuron_id, period_start:period_end])


# Normalise firing rates:   
if normalise_rates:
    mean_F0 = mean_F0/np.amax(mean_F0)
    mean_F1 = mean_F1/np.amax(mean_F1)
    ylab = 'Relative activity'
else:
    ylab = 'Firing rate (spikes/sec)'
    
# Plot firing rates as a function of spatial frequency:
plt.plot(cpd_array, mean_F1, linestyle='-', marker='.', 
         linewidth=0.2, c='mediumblue', label='F1')
plt.plot(cpd_array, mean_F0, linestyle='-', marker='.', 
         linewidth=0.2, c='tab:cyan', label='F0')
plt.xlabel('Spatial frequency (cpd)')
plt.ylabel(ylab)
plt.figlegend()
 
        
