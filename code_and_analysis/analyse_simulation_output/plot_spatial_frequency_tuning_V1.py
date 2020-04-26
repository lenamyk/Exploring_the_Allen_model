#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Parameters
----------
simlength: length of simulation (ms)
tsteps_per_sec: temporal resolution (time steps per seconds)
tf: temporal frequency (Hz)
dogparam: parameters of DoG filter
"""


import numpy as np
import matplotlib.pyplot as plt
import statistics


simlength = 3000  
tsteps_per_sec = 1000
dogparam = '1.73ac_1as_2.45sc'
tf = 4


# Number of trials:
trials = np.arange(20) 
# Set of stimuli:
tuning_array = np.array([0.005,0.04,0.08,0.12,0.16,0.2,0.24,0.28,0.32])


#Firing rates (F0 and F1) for each stimuli size:
mean_F0 = np.zeros(len(tuning_array))  
sd_F0 = np.zeros(len(tuning_array))  
mean_F1 = np.zeros(len(tuning_array))
sd_F1 = np.zeros(len(tuning_array))


#Decide if firing rates are normalised:
normalise_rates = False


# Example neuron id:
nrid = '23428'

for j in range(len(tuning_array)):
    matrix_of_firing_rates = np.zeros((simlength,len(trials)))
    time_average_per_trial = np.zeros(len(trials))
    mean_amplitude_per_trial = np.zeros(len(trials))
    fig = plt.figure()
    
    
    for i in trials:    
        # Read in spike output:
        filename = "m1/id%s/%scpd/3s_0d_%dtf_%s_tr%d.txt" \
        % (nrid,  tuning_array[j], tf, dogparam, i)
        a = open(filename, 'r+')
        spike_times = a.readlines()
        spike_times = np.array(spike_times, dtype=float)
        a.close()
    
    
        #Create discrete firing rate:
        t = np.linspace(0, simlength + 1, simlength)  
        firing_rate = np.zeros_like(t)
        entries = np.digitize(spike_times, t)
        firing_rate[entries] = 1*tsteps_per_sec
        matrix_of_firing_rates[:,i] = firing_rate
    
    
        #Calculate F0 per trial:
        time_average_per_trial[i] \
        = np.mean(matrix_of_firing_rates[200:, i])
    
    
        #Calculate F1 per trial:
        start_measuring = 500
        signal = firing_rate[start_measuring:simlength]
        ff = np.fft.fft(signal)   
        ff_freq = np.fft.fftfreq(signal.size)*tsteps_per_sec 
        amp_spectr = np.abs(ff)*2/(simlength - start_measuring) 
        ff_freq_positive = ff_freq[:int(len(ff_freq)/2)]
        amp_spectr_positive = amp_spectr[:int(len(amp_spectr)/2)]  
        mean_amplitude_per_trial[i] = np.interp(tf, 
                                                ff_freq_positive, 
                                                amp_spectr_positive)  
    
    
        # Plot firing rate for each trial:
        title = "trial %d" % (i + 1)
        ax = fig.add_subplot(len(trials) + 1, 1, i + 1)  
        ax.set_title(title, loc='left', x=1.01, y=0)
        ax.plot(t, matrix_of_firing_rates[:,i], linewidth=0.5)
        ax.set_xticklabels([])
            
        
    #Calculate trial averages of F0 and F1:
    mean_spikes = np.mean(matrix_of_firing_rates, axis=1)  
    mean_F0[j] = np.mean(mean_spikes)  
    sd_F0[j] = statistics.stdev(time_average_per_trial)
    mean_F1[j] = np.mean(mean_amplitude_per_trial)  
    sd_F1[j] = statistics.stdev(mean_amplitude_per_trial)


    # Plot trial averaged spike train:
    ax = fig.add_subplot(len(trials)+1, 1, len(trials)+1)
    ax.plot(t, mean_spikes,c='tab:orange', linewidth=0.5)
    ax.set_title('mean', loc='left', x=1.01, y=0)
    ax.set_title('%s cpd' %(tuning_array[j]), y=21)
    ax.set_xlabel('Time (ms)')
    plt.subplots_adjust(hspace=0)
    plt.tight_layout()

    
#Normalise rates:
if normalise_rates:
    sd_F1 = sd_F1/np.amax(mean_F1)
    sd_F0 = sd_F0/np.amax(mean_F0)
    mean_F1 = mean_F1/np.amax(mean_F1)
    mean_F0 = mean_F0/np.amax(mean_F0)
    ylab = 'Relative activity'
else:
    ylab = 'Firing rate (spikes/sec)'


# Plot F0 and F1 as a function of stimuli size:
fig = plt.figure()
plt.errorbar(tuning_array, mean_F0, yerr=sd_F0, marker='.', 
             label='F0', c='tab:cyan')
plt.errorbar(tuning_array, mean_F1, yerr=sd_F1, marker='.', 
             label='F1', c='mediumblue')
plt.xlabel('Spatial frequency (cpd)')
plt.ylabel(ylab)
plt.figlegend() 



