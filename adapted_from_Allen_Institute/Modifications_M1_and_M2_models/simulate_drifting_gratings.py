"""
Copyright 2017. Allen Institute. All rights reserved

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


import os
import sys
sys.path.append('lgnmodel')
sys.path.append('network2')
from optparse import OptionParser
from lgn_functions import *

import matplotlib.pyplot as plt

from network2.nxnetwork import NxNetwork
from network2.syn_network import SynNetwork
from network2.formats import ISeeFormat

import time

# To load networks already saved
#del LGN      #LGN not defined, so commented out this
LGN = SynNetwork.load('../LGN/lgn_full_col_cells_3.csv', format=ISeeFormat, positions=['x', 'y'])


radius = 40
duration = 3  #1.0 #0.01#3.0#0.5#3.0#2.05 (for flashes)
gray_screen = 0.2
cpd = 0.04
TF = 4.
for i in [155.2612]: #range(0, 360, 45):
    for j in [4]:
        #print 'Angle:', i
        #print 'TF: ', j
        TF = j
        direction = i
        contrast = 100.
        #output_file_name = 'fullfield_0.5sec_0.2cpd_0theta_0gray_4tf_1.06ac_1as_10sc_unnormed'
        output_file_name = 'testspontaneousrate'
        #output_file_name = '80pix_3sec_moving_155ang_4tf_0.04cpd_0.2gray_1.5ac_0.5as_2.45sc_unnormed'
        #output_file_name = '14pix_3sec_moving_155ang_4tf_0.04cpd_0.2gray'
        #output_file_name = 'full3_production_' + str(duration) + 'sec_SF' + str(cpd) + '_TF' + str(TF) +'_ori' + str(float(direction)) +'_c' + str(contrast) + '_gs' + str(gray_screen) #'fast_test''full2_gsOnly_' + str(gray_screen) #

        #print output_file_name
        stimulus = 'grating'
        trials = 20

        startTime = time.time()
        calculate_firing_rate(LGN, stimulus, output_file_name, duration, gray_screen, cpd, TF, direction, contrast, radius)        
        generate_spikes(LGN, trials, duration, output_file_name)
        #print 'Duration: ', time.time() - startTime
#print 'Done with spikes!'




