#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameters
----------
neuron_id: neuron to be analysed
simlength: length of simulation (ms)
tsteps_per_sec: temporal resolution (time steps per seconds)
tf: temporal frequency (Hz)
sf: spatial frequency (cpd)
tuning_array: set of stimuli diameters for the tuning curve
normalise_rates: set to 'True' if rates should be normalised
stim_type: type of stimuli, set to either 'flash' or 'grating'
grat_interval: measurement window for grating stimuli
flash_interval: measurement window for flashing stimuli
"""


import h5py
import numpy as np
import matplotlib.pyplot as plt


neuron_id = 8018  
simlength = 3000 
tsteps_per_sec = 1000
tf = 4
sf = 0.04
tuning_array = np.array([5,10,20,40,80,160,240])
normalise_rates = False
stim_type = 'grating'
grat_interval = 1000
flash_interval = 1200


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
    if stim_type == 'grating':
        interval = grat_interval
    elif stim_type == 'flash':
        interval = flash_interval
    period_start = int(simlength/2) 
    period_end = period_start + interval
        
    
    # Calculate F1:
    if stim_type == 'grating':
        signal = fr[neuron_id,period_start:period_end]
        ff = np.fft.fft(signal)   
        ff_freq = np.fft.fftfreq(signal.size)*tsteps_per_sec 
        amplitude_spectum = np.abs(ff)*2/(period_end - period_start)
        amp_ind = np.where(ff_freq == tf) 
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
plt.xlabel('Diameter (degrees)')
plt.ylabel(ylab)
plt.figlegend()
 
        
