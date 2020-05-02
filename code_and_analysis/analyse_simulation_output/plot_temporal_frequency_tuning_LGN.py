#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameters
----------
neuron_id: neuron to be analysed
simlength: length of simulation (ms)
tsteps_per_sec: temporal resolution (time steps per seconds)
sf: spatial frequency (cpd)
tuning_array: set of temporal frequencies for the tuning curve
normalise_rates: set to 'True' if rates should be normalised
window: measurement window for firing rates
"""


import h5py
import numpy as np
import matplotlib.pyplot as plt


neuron_id = 8018  
simlength = 3000 
tsteps_per_sec = 1000
sf = 0.04
tuning_array = np.array([1,2,4,8,15,30])
normalise_rates = False
stim_type = 'grating'
window = 1000


# Firing rates (F0 and F1) for each stimuli size:
mean_F0 = np.zeros(len(tuning_array))
mean_F1 = np.zeros(len(tuning_array))


for i in range(len(tuning_array)):
    f = h5py.File('LGN/m1/%spix/3s_0.2gray_155d_4tf_0.04cpd.h5' \
                  % (tuning_array[i]), 'r')
         
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
    amp_ind = np.where(ff_freq == tuning_array[i]) 
    mean_F1[i] = amplitude_spectum[amp_ind] 
        
        
    # Calculate time average (or F0):
    mean_F0[i] = np.mean(fr[neuron_id, period_start:period_end])


# Normalise firing rates:   
if normalise_rates and stim_type == 'grating':
    mean_F0 = mean_F0/np.amax(mean_F0)
    mean_F1 = mean_F1/np.amax(mean_F1)
    ylab = 'Relative activity'
else:
    ylab = 'Firing rate (spikes/sec)'
    
    
# Plot firing rates as a function of stimuli size:
if stim_type == 'grating':
    plt.plot(tuning_array, mean_F1, linestyle='-', marker='.', 
             linewidth=0.2, c='mediumblue', label='F1')
    plt.plot(tuning_array, mean_F0, linestyle='-', marker='.', 
             linewidth=0.2, c='tab:cyan', label='F0')
else:
    plt.plot(tuning_array, mean_F0, linestyle='-', marker='.', 
             linewidth=0.2, c='tab:cyan', label='Mean firing rate ')
plt.xlabel('Temporal frequency (Hz)')
plt.ylabel(ylab)
plt.figlegend()
 
        
