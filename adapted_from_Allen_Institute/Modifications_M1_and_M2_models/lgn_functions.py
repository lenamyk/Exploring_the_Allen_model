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

import csv
import os
import pickle
from random import *
from math import *
import math
import numpy as np
import numpy.matlib as np_matlib
import matplotlib.pyplot as plt
#import isee_engine.nwb as nwb
#import bmtk.simulator.utils.nwb as nwb
from lgnmodel.nwb_copy import *
import h5py as h5 


from lgnmodel.spatialfilter_surr import GaussianSpatialFilter, ArrayFilter   # Changed to Difference-of-Gaussians
from lgnmodel.temporalfilter import TemporalFilterCosineBump
from lgnmodel.linearfilter import SpatioTemporalFilter
from lgnmodel.transferfunction import ScalarTransferFunction, MultiTransferFunction
from lgnmodel.cellmodel import TwoSubfieldLinearCell, OnUnit, OffUnit
from lgnmodel.util_fns import create_ff_mov, get_data_metrics_for_each_subclass, create_grating_movie_list, get_tcross_from_temporal_kernel
from lgnmodel.moving_grating import GratingMovie, FullFieldFlashMovie # Define which visual input to use
import lgnmodel.poissongeneration as pg

from sympy.abc import x as symbolic_x
from sympy.abc import y as symbolic_y

import time

def calculate_FullFieldFlashMovie(LGN, fname):
    grey1 = 0.5
    grey2 = 1.0
    flashON = 0.25
    flashOFF = 0.25

    m_grey1 = FullFieldFlashMovie(np.arange(120), np.arange(240), 0., grey1, frame_rate=1000,max_intensity=0).full(t_max=grey1)
    m_grey2 = FullFieldFlashMovie(np.arange(120), np.arange(240), 0., grey2, frame_rate=1000,max_intensity=0).full(t_max=grey2)
    m_on  = FullFieldFlashMovie(np.arange(120), np.arange(240), 0., flashON, frame_rate=1000,max_intensity=0.8).full(t_max=flashON)
    m_off = FullFieldFlashMovie(np.arange(120), np.arange(240), 0., flashOFF, frame_rate=1000,max_intensity=-0.8).full(t_max=flashOFF)

    movie_to_show = m_grey1 + m_on + m_grey2 + m_off + m_grey2
    print(np.shape(movie_to_show.data))
    plt.plot(movie_to_show[:, 10, 35])
    plt.show()

    origin = (0.,0.)

    for counter, node in enumerate(LGN.nodes()):

        # For spatial filter and locations
        translate = (node[1]['position'][0], node[1]['position'][1])
        sigma = node[1]['spatial_size'] / 3.  # convert from degree to SD
        sigma = (sigma, sigma)
        spatial_filter = GaussianSpatialFilter(translate=translate, sigma=sigma, origin=origin)


        ################# Extract cell parameters needed      #################
        if node[1]['model_id'] == 'sONsOFF_001':

            # sON temporal filter
            sON_prs = {'opt_wts': [node[1]['weight_non_dom_0'], node[1]['weight_non_dom_1']],
                        'opt_kpeaks': [node[1]['kpeaks_non_dom_0'], node[1]['kpeaks_non_dom_1']],
                        'opt_delays': [node[1]['delay_non_dom_0'], node[1]['delay_non_dom_1']]}
            sON_filt_new = createOneUnitOfTwoSubunitFilter(sON_prs, 121.0)
            sON_sum = sON_filt_new[1]
            sON_filt_new = sON_filt_new[0]

            # tOFF temporal filter
            sOFF_prs = {'opt_wts': [node[1]['weight_dom_0'], node[1]['weight_dom_1']],
                        'opt_kpeaks': [node[1]['kpeaks_dom_0'], node[1]['kpeaks_dom_1']],
                        'opt_delays': [node[1]['delay_dom_0'], node[1]['delay_dom_1']]}
            sOFF_filt_new = createOneUnitOfTwoSubunitFilter(sOFF_prs, 115.0)
            sOFF_sum = sOFF_filt_new[1]
            sOFF_filt_new = sOFF_filt_new[0]

            amp_on = 1.0  # set the non-dominant subunit amplitude to unity
            spont = 4.0
            max_roff = 35.0
            max_ron = 21.0
            amp_off = -(max_roff / max_ron) * (sON_sum / sOFF_sum) * amp_on - (spont * (max_roff - max_ron)) / (
            max_ron * sOFF_sum)

            # Create sON subunit:
            xfer_fn_son = ScalarTransferFunction('Heaviside(s+' + str(0.5 * spont) + ')*(s+' + str(0.5 * spont) + ')')
            linear_filter_son = SpatioTemporalFilter(spatial_filter, sON_filt_new, amplitude=amp_on)
            scell_on = OnUnit(linear_filter_son, xfer_fn_son)

            # Create sOFF subunit:
            xfer_fn_soff = ScalarTransferFunction('Heaviside(s+' + str(0.5 * spont) + ')*(s+' + str(0.5 * spont) + ')')
            linear_filter_soff = SpatioTemporalFilter(spatial_filter, sOFF_filt_new, amplitude=amp_off)
            scell_off = OffUnit(linear_filter_soff, xfer_fn_soff)

            sep_ss_onoff_cell = create_two_sub_cell(linear_filter_soff, linear_filter_son, 0.5 * spont, 0.5 * spont,
                                                    node[1]['tuning_angle'], node[1]['sf_sep'], translate)
            cell = sep_ss_onoff_cell

            #t, f_tot = cell.evaluate(movie_to_show, downsample=1, separable = True)  # Taking the second movie which is 4 Hz


        elif node[1]['model_id'] == 'sONtOFF_001':
            # spatial_filter.get_kernel(np.arange(120), np.arange(240)).imshow()
            # sON temporal filter
            sON_prs = {'opt_wts': [node[1]['weight_non_dom_0'], node[1]['weight_non_dom_1']],
                       'opt_kpeaks': [node[1]['kpeaks_non_dom_0'], node[1]['kpeaks_non_dom_1']],
                       'opt_delays': [node[1]['delay_non_dom_0'], node[1]['delay_non_dom_1']]}
            sON_filt_new = createOneUnitOfTwoSubunitFilter(sON_prs, 93.5)
            sON_sum = sON_filt_new[1]
            sON_filt_new = sON_filt_new[0]

            # tOFF temporal filter
            tOFF_prs = {'opt_wts': [node[1]['weight_dom_0'], node[1]['weight_dom_1']],
                        'opt_kpeaks': [node[1]['kpeaks_dom_0'], node[1]['kpeaks_dom_1']],
                        'opt_delays': [node[1]['delay_dom_0'], node[1]['delay_dom_1']]}
            tOFF_filt_new = createOneUnitOfTwoSubunitFilter(tOFF_prs, 64.8)   #64.8
            tOFF_sum = tOFF_filt_new[1]
            tOFF_filt_new = tOFF_filt_new[0]

            amp_on = 1.0  # set the non-dominant subunit amplitude to unity
            spont = 5.5
            max_roff = 46.0
            max_ron = 31.0
            amp_off = -0.7 * (max_roff / max_ron) * (sON_sum / tOFF_sum) * amp_on - (spont * (max_roff - max_ron)) / (
            max_ron * tOFF_sum)

            # Create sON subunit:
            xfer_fn_son = ScalarTransferFunction('Heaviside(s+' + str(0.5 * spont) + ')*(s+' + str(0.5 * spont) + ')')
            linear_filter_son = SpatioTemporalFilter(spatial_filter, sON_filt_new, amplitude=amp_on)
            scell_on = OnUnit(linear_filter_son, xfer_fn_son)
            # linear_filter_son.spatial_filter.get_kernel(np.arange(120), np.arange(240)).imshow()

            # Create tOFF subunit:
            xfer_fn_toff = ScalarTransferFunction('Heaviside(s+' + str(0.5 * spont) + ')*(s+' + str(0.5 * spont) + ')')
            linear_filter_toff = SpatioTemporalFilter(spatial_filter, tOFF_filt_new, amplitude=amp_off)
            tcell_off = OffUnit(linear_filter_toff, xfer_fn_toff)
            # linear_filter_toff.spatial_filter.get_kernel(np.arange(120), np.arange(240)).kernel


            sep_ts_onoff_cell = create_two_sub_cell(linear_filter_toff, linear_filter_son, 0.5 * spont, 0.5 * spont,
                                                    node[1]['tuning_angle'], node[1]['sf_sep'], translate)
            # sep_ts_onoff_cell.show_temporal_filter()

            cell = sep_ts_onoff_cell
            #t, f_tot = cell.evaluate(movie_to_show, downsample=1, separable = True)  # Taking the second movie which is 4 Hz
        else:
            cell_type = node[1]['model_id'][0: node[1]['model_id'].find('_')]  # 'sON'  # 'tOFF'
            tf_str = node[1]['model_id'][node[1]['model_id'].find('_') + 1:]

            # For temporal filter
            wts    = [node[1]['weight_dom_0'], node[1]['weight_dom_1']]
            kpeaks = [node[1]['kpeaks_dom_0'], node[1]['kpeaks_dom_1']]
            delays = [node[1]['delay_dom_0'],  node[1]['delay_dom_1']]

            ################# End of extract cell parameters needed   #################

            # Get spont from experimental data
            exp_prs_dict = get_data_metrics_for_each_subclass(cell_type)
            subclass_prs_dict = exp_prs_dict[tf_str]
            spont_exp = subclass_prs_dict['spont_exp']
            spont_str = str(spont_exp[0])

            # Get filters
            transfer_function = ScalarTransferFunction('Heaviside(s+' + spont_str + ')*(s+' + spont_str + ')')
            temporal_filter = TemporalFilterCosineBump(wts, kpeaks, delays)

            if cell_type.find('ON') >= 0:
                amplitude = 1.0
                linear_filter = SpatioTemporalFilter(spatial_filter, temporal_filter, amplitude=amplitude)
                cell = OnUnit(linear_filter, transfer_function)
            elif cell_type.find('OFF') >= 0:
                amplitude = -1.0
                linear_filter = SpatioTemporalFilter(spatial_filter, temporal_filter, amplitude=amplitude)
                cell = OffUnit(linear_filter, transfer_function)

        t, f_tot = cell.evaluate(movie_to_show, downsample=1, separable=True)  # Taking the second movie which is 4 Hz

        if counter == 0:
            firing_rates = np.zeros([len(LGN.nodes()) + 1, len(t)])
            firing_rates[0, :] = t
            time = np.asarray(t)
        firing_rates[counter + 1, :] = f_tot
        

    f = h5.File(fname + '_f_tot.h5', 'w')
    f.create_dataset('time', data=time)
    f.create_dataset('firing_rates_Hz', data=firing_rates[1:, :])
    f.close()






def generate_positions(N, x0=0.0, x1=300.0, y0=0.0, y1=100.0):
    X = np.random.uniform(x0, x1, N)
    Y = np.random.uniform(y0, y1, N)
    return np.column_stack((X, Y))



def generate_positions_grids(N, X_grids, Y_grids, X_len, Y_len):
    widthPerTile  = X_len/X_grids
    heightPerTile = Y_len/Y_grids

    X = np.zeros(N * X_grids * Y_grids)
    Y = np.zeros(N * X_grids * Y_grids)

    counter = 0
    for i in range(X_grids):
        for j in range(Y_grids):
            X_tile = np.random.uniform(i*widthPerTile,  (i+1) * widthPerTile,  N)
            Y_tile = np.random.uniform(j*heightPerTile, (j+1) * heightPerTile, N)
            X[counter*N:(counter+1)*N] = X_tile
            Y[counter*N:(counter+1)*N] = Y_tile
            counter = counter + 1
    return np.column_stack((X, Y))



def get_filter_spatial_size(N, X_grids, Y_grids, size_range):

    spatial_sizes = np.zeros(N * X_grids * Y_grids)

    counter = 0
    for i in range(X_grids):
        for j in range(Y_grids):
            if len(size_range) == 1:
                sizes = np.ones(N) * size_range[0]
            else:
                sizes = np.random.triangular(size_range[0], size_range[0] + 1, size_range[1], N)
            spatial_sizes[counter * N:(counter + 1) * N] = sizes
            counter = counter + 1
    return spatial_sizes


def get_filter_temporal_params(N, X_grids, Y_grids, model):

    # Total number of cells
    N_total = N * X_grids * Y_grids

    # Jitter parameters
    jitter = 0.025
    lower_jitter = 1 - jitter
    upper_jitter = 1 + jitter

    # Directory of pickle files with saved parameter values
    basepath = '/data/mat/RamIyer/lgnmodel_backup/lgnmodel/examples/Jan2017/opt_pkl_files/chosen_best_fits/'

    # For two-subunit filter (sONsOFF and sONtOFF)
    sOFF_fn = os.path.join(basepath + 'sOFF_TF4_3.5_-2.0_10.0_60.0_15.0_ic.pkl')  # best chosen fit for sOFF 4 Hz
    tOFF_fn = os.path.join(basepath + 'tOFF_TF8_4.222_-2.404_8.545_23.019_0.0_ic.pkl')  # best chosen fit for tOFF 8 Hz
    sON_fn = os.path.join(basepath + 'sON_TF4_3.5_-2.0_30.0_60.0_25.0_ic.pkl')  # best chosen fit for sON 4 Hz

    sOFF_prs = pickle.load(open(sOFF_fn, 'rb'))
    tOFF_prs = pickle.load(open(tOFF_fn, 'rb'))
    sON_prs = pickle.load(open(sON_fn, 'rb'))

    # Choose cell type and temporal frequency
    if model == 'sONsOFF_001':

        kpeaks = sOFF_prs['opt_kpeaks']
        kpeaks_dom_0 = np.random.uniform(lower_jitter * kpeaks[0], upper_jitter * kpeaks[0], N_total)
        kpeaks_dom_1 = np.random.uniform(lower_jitter * kpeaks[1], upper_jitter * kpeaks[1], N_total)
        kpeaks =  sON_prs['opt_kpeaks']
        kpeaks_non_dom_0 = np.random.uniform(lower_jitter * kpeaks[0], upper_jitter * kpeaks[0], N_total)
        kpeaks_non_dom_1 = np.random.uniform(lower_jitter * kpeaks[1], upper_jitter * kpeaks[1], N_total)

        wts = sOFF_prs['opt_wts']
        wts_dom_0 = np.random.uniform(lower_jitter * wts[0], upper_jitter * wts[0], N_total)
        wts_dom_1 = np.random.uniform(lower_jitter * wts[1], upper_jitter * wts[1], N_total)
        wts =  sON_prs['opt_wts']
        wts_non_dom_0 = np.random.uniform(lower_jitter * wts[0], upper_jitter * wts[0], N_total)
        wts_non_dom_1 = np.random.uniform(lower_jitter * wts[1], upper_jitter * wts[1], N_total)

        delays = sOFF_prs['opt_delays']
        delays_dom_0 = np.random.uniform(lower_jitter * delays[0], upper_jitter * delays[0], N_total)
        delays_dom_1 = np.random.uniform(lower_jitter * delays[1], upper_jitter * delays[1], N_total)
        delays =  sON_prs['opt_delays']
        delays_non_dom_0 = np.random.uniform(lower_jitter * delays[0], upper_jitter * delays[0], N_total)
        delays_non_dom_1 = np.random.uniform(lower_jitter * delays[1], upper_jitter * delays[1], N_total)

        sf_sep = 6.
        sf_sep = np.random.uniform(lower_jitter * sf_sep, upper_jitter * sf_sep, N_total)
        tuning_angles = np.random.uniform(0, 360., N_total)

    elif model == 'sONtOFF_001':

        kpeaks = tOFF_prs['opt_kpeaks']
        kpeaks_dom_0 = np.random.uniform(lower_jitter * kpeaks[0], upper_jitter * kpeaks[0], N_total)
        kpeaks_dom_1 = np.random.uniform(lower_jitter * kpeaks[1], upper_jitter * kpeaks[1], N_total)
        kpeaks = sON_prs['opt_kpeaks']
        kpeaks_non_dom_0 = np.random.uniform(lower_jitter * kpeaks[0], upper_jitter * kpeaks[0], N_total)
        kpeaks_non_dom_1 = np.random.uniform(lower_jitter * kpeaks[1], upper_jitter * kpeaks[1], N_total)

        wts = tOFF_prs['opt_wts']
        wts_dom_0 = np.random.uniform(lower_jitter * wts[0], upper_jitter * wts[0], N_total)
        wts_dom_1 = np.random.uniform(lower_jitter * wts[1], upper_jitter * wts[1], N_total)
        wts = sON_prs['opt_wts']
        wts_non_dom_0 = np.random.uniform(lower_jitter * wts[0], upper_jitter * wts[0], N_total)
        wts_non_dom_1 = np.random.uniform(lower_jitter * wts[1], upper_jitter * wts[1], N_total)

        delays = tOFF_prs['opt_delays']
        delays_dom_0 = np.random.uniform(lower_jitter * delays[0], upper_jitter * delays[0], N_total)
        delays_dom_1 = np.random.uniform(lower_jitter * delays[1], upper_jitter * delays[1], N_total)
        delays = sON_prs['opt_delays']
        delays_non_dom_0 = np.random.uniform(lower_jitter * delays[0], upper_jitter * delays[0], N_total)
        delays_non_dom_1 = np.random.uniform(lower_jitter * delays[1], upper_jitter * delays[1], N_total)

        sf_sep = 4.
        sf_sep = np.random.uniform(lower_jitter * sf_sep, upper_jitter * sf_sep, N_total)
        tuning_angles = np.random.uniform(0, 360., N_total)

    else:
        cell_type = model[0: model.find('_')]    #'sON'  # 'tOFF'
        tf_str = model[model.find('_') + 1:]

        # Load pickle file containing params for optimized temporal kernel, it it exists
        file_found = 0
        for fname in os.listdir(basepath):
            if os.path.isfile(os.path.join(basepath, fname)):
                pkl_savename = os.path.join(basepath, fname)
                if (tf_str in pkl_savename.split('_') and pkl_savename.find(cell_type) >= 0 and pkl_savename.find('.pkl') >= 0):
                    file_found = 1
                    print(pkl_savename)
                    filt_file = pkl_savename

        if file_found != 1:
            print('File not found: Filter was not optimized for this sub-class')

        savedata_dict = pickle.load(open(filt_file, 'rb'))

        kpeaks = savedata_dict['opt_kpeaks']
        kpeaks_dom_0 = np.random.uniform(lower_jitter * kpeaks[0], upper_jitter * kpeaks[0], N_total)
        kpeaks_dom_1 = np.random.uniform(lower_jitter * kpeaks[1], upper_jitter * kpeaks[1], N_total)
        kpeaks_non_dom_0 = np.nan * np.zeros(N_total)
        kpeaks_non_dom_1 = np.nan * np.zeros(N_total)

        wts = savedata_dict['opt_wts']
        wts_dom_0 = np.random.uniform(lower_jitter * wts[0], upper_jitter * wts[0], N_total)
        wts_dom_1 = np.random.uniform(lower_jitter * wts[1], upper_jitter * wts[1], N_total)
        wts_non_dom_0 = np.nan * np.zeros(N_total)
        wts_non_dom_1 = np.nan * np.zeros(N_total)

        delays = savedata_dict['opt_delays']
        delays_dom_0 = np.random.uniform(lower_jitter * delays[0], upper_jitter * delays[0], N_total)
        delays_dom_1 = np.random.uniform(lower_jitter * delays[1], upper_jitter * delays[1], N_total)
        delays_non_dom_0 = np.nan * np.zeros(N_total)
        delays_non_dom_1 = np.nan * np.zeros(N_total)

        sf_sep = np.nan * np.zeros(N_total)
        tuning_angles =  np.nan * np.zeros(N_total)

    return np.column_stack((kpeaks_dom_0, kpeaks_dom_1, wts_dom_0, wts_dom_1, delays_dom_0, delays_dom_1,
                            kpeaks_non_dom_0, kpeaks_non_dom_1, wts_non_dom_0, wts_non_dom_1,
                            delays_non_dom_0, delays_non_dom_1, tuning_angles, sf_sep))



def get_tuning_angles(N, X_grids, Y_grids, model):

    total_N = N * X_grids * Y_grids

    if model == 'sONsOFF_001':
        tuning_angles = np_matlib.repmat((np.linspace(0, 360, N, endpoint=False)), 1, X_grids * Y_grids)[0]
    else:
        tuning_angles = np.nan * np.zeros(total_N)

    return tuning_angles


def calculate_firing_rate(LGN, stimulus, output_file_name, duration, gray_screen,  cpd, TF, direction, contrast,radius):

    origin = (0.,0.)

    # Gratings simulation
    movie_to_show = GratingMovie(120, 240).create_movie(t_min = 0, t_max = duration, gray_screen_dur = gray_screen, cpd = cpd, temporal_f = TF, theta = direction, contrast = contrast/100., radius = radius) #, row_size_new = 120, col_size_new = 240)
    # movie_to_show = FullFieldFlashMovie(range(120), range(240),1,2).full(t_max=2)
    # gr_dir_name = '/data/mat/antona/network/14-simulations/6-LGN_firing_rates_and_spikes/LGN_spike_trains/movies_gratings/res_192'
    # gr_mov_list = create_grating_movie_list(gr_dir_name)


    for counter, node in enumerate(LGN.nodes()):

        # For spatial filter and locations
        translate = (node[1]['position'][0], node[1]['position'][1])
        sigma = node[1]['spatial_size'] / 3.  # convert from degree to SD
        sigma = (sigma, sigma)
        spatial_filter = GaussianSpatialFilter(translate=translate, sigma=sigma, origin=origin)
        print('positions:', translate)
        print('neuron id:', node[1]['id'])


        ################# Extract cell parameters needed      #################
        if node[1]['model_id'] == 'sONsOFF_001':

            # sON temporal filter
            sON_prs = {'opt_wts': [node[1]['weight_non_dom_0'], node[1]['weight_non_dom_1']],
                        'opt_kpeaks': [node[1]['kpeaks_non_dom_0'], node[1]['kpeaks_non_dom_1']],
                        'opt_delays': [node[1]['delay_non_dom_0'], node[1]['delay_non_dom_1']]}
            sON_filt_new = createOneUnitOfTwoSubunitFilter(sON_prs, 121.0)
            sON_sum = sON_filt_new[1]
            sON_filt_new = sON_filt_new[0]

            # tOFF temporal filter
            sOFF_prs = {'opt_wts': [node[1]['weight_dom_0'], node[1]['weight_dom_1']],
                        'opt_kpeaks': [node[1]['kpeaks_dom_0'], node[1]['kpeaks_dom_1']],
                        'opt_delays': [node[1]['delay_dom_0'], node[1]['delay_dom_1']]}
            sOFF_filt_new = createOneUnitOfTwoSubunitFilter(sOFF_prs, 115.0)
            sOFF_sum = sOFF_filt_new[1]
            sOFF_filt_new = sOFF_filt_new[0]

            amp_on = 1.0  # set the non-dominant subunit amplitude to unity
            spont = 4.0
            max_roff = 35.0
            max_ron = 21.0
            amp_off = -(max_roff / max_ron) * (sON_sum / sOFF_sum) * amp_on - (spont * (max_roff - max_ron)) / (
            max_ron * sOFF_sum)

            # Create sON subunit:
            xfer_fn_son = ScalarTransferFunction('Heaviside(s+' + str(0.5 * spont) + ')*(s+' + str(0.5 * spont) + ')')
            linear_filter_son = SpatioTemporalFilter(spatial_filter, sON_filt_new, amplitude=amp_on)
            scell_on = OnUnit(linear_filter_son, xfer_fn_son)

            # Create sOFF subunit:
            xfer_fn_soff = ScalarTransferFunction('Heaviside(s+' + str(0.5 * spont) + ')*(s+' + str(0.5 * spont) + ')')
            linear_filter_soff = SpatioTemporalFilter(spatial_filter, sOFF_filt_new, amplitude=amp_off)
            scell_off = OffUnit(linear_filter_soff, xfer_fn_soff)

            sep_ss_onoff_cell = create_two_sub_cell(linear_filter_soff, linear_filter_son, 0.5 * spont, 0.5 * spont,
                                                    node[1]['tuning_angle'], node[1]['sf_sep'], translate)
            cell = sep_ss_onoff_cell

            #t, f_tot = cell.evaluate(movie_to_show, downsample=1, separable = True)  # Taking the second movie which is 4 Hz


        elif node[1]['model_id'] == 'sONtOFF_001':
            # spatial_filter.get_kernel(np.arange(120), np.arange(240)).imshow()
            # sON temporal filter
            sON_prs = {'opt_wts': [node[1]['weight_non_dom_0'], node[1]['weight_non_dom_1']],
                       'opt_kpeaks': [node[1]['kpeaks_non_dom_0'], node[1]['kpeaks_non_dom_1']],
                       'opt_delays': [node[1]['delay_non_dom_0'], node[1]['delay_non_dom_1']]}
            sON_filt_new = createOneUnitOfTwoSubunitFilter(sON_prs, 93.5)
            sON_sum = sON_filt_new[1]
            sON_filt_new = sON_filt_new[0]

            # tOFF temporal filter
            tOFF_prs = {'opt_wts': [node[1]['weight_dom_0'], node[1]['weight_dom_1']],
                        'opt_kpeaks': [node[1]['kpeaks_dom_0'], node[1]['kpeaks_dom_1']],
                        'opt_delays': [node[1]['delay_dom_0'], node[1]['delay_dom_1']]}
            tOFF_filt_new = createOneUnitOfTwoSubunitFilter(tOFF_prs, 64.8)   #64.8
            tOFF_sum = tOFF_filt_new[1]
            tOFF_filt_new = tOFF_filt_new[0]

            amp_on = 1.0  # set the non-dominant subunit amplitude to unity
            spont = 5.5
            max_roff = 46.0
            max_ron = 31.0
            amp_off = -0.7 * (max_roff / max_ron) * (sON_sum / tOFF_sum) * amp_on - (spont * (max_roff - max_ron)) / (
            max_ron * tOFF_sum)

            # Create sON subunit:
            xfer_fn_son = ScalarTransferFunction('Heaviside(s+' + str(0.5 * spont) + ')*(s+' + str(0.5 * spont) + ')')
            linear_filter_son = SpatioTemporalFilter(spatial_filter, sON_filt_new, amplitude=amp_on)
            scell_on = OnUnit(linear_filter_son, xfer_fn_son)
            # linear_filter_son.spatial_filter.get_kernel(np.arange(120), np.arange(240)).imshow()

            # Create tOFF subunit:
            xfer_fn_toff = ScalarTransferFunction('Heaviside(s+' + str(0.5 * spont) + ')*(s+' + str(0.5 * spont) + ')')
            linear_filter_toff = SpatioTemporalFilter(spatial_filter, tOFF_filt_new, amplitude=amp_off)
            tcell_off = OffUnit(linear_filter_toff, xfer_fn_toff)
            # linear_filter_toff.spatial_filter.get_kernel(np.arange(120), np.arange(240)).kernel


            sep_ts_onoff_cell = create_two_sub_cell(linear_filter_toff, linear_filter_son, 0.5 * spont, 0.5 * spont,
                                                    node[1]['tuning_angle'], node[1]['sf_sep'], translate)
            # sep_ts_onoff_cell.show_temporal_filter()

            cell = sep_ts_onoff_cell
            #t, f_tot = cell.evaluate(movie_to_show, downsample=1, separable = True)  # Taking the second movie which is 4 Hz
        else:
            cell_type = node[1]['model_id'][0: node[1]['model_id'].find('_')]  # 'sON'  # 'tOFF'
            tf_str = node[1]['model_id'][node[1]['model_id'].find('_') + 1:]

            # For temporal filter
            wts    = [node[1]['weight_dom_0'], node[1]['weight_dom_1']]
            kpeaks = [node[1]['kpeaks_dom_0'], node[1]['kpeaks_dom_1']]
            delays = [node[1]['delay_dom_0'],  node[1]['delay_dom_1']]

            ################# End of extract cell parameters needed   #################

            # Get spont from experimental data
            exp_prs_dict = get_data_metrics_for_each_subclass(cell_type)
            subclass_prs_dict = exp_prs_dict[tf_str]
            spont_exp = subclass_prs_dict['spont_exp']
            spont_str = str(spont_exp[0])

            # Get filters
            transfer_function = ScalarTransferFunction('Heaviside(s+' + spont_str + ')*(s+' + spont_str + ')')
            temporal_filter = TemporalFilterCosineBump(wts, kpeaks, delays)

            if cell_type.find('ON') >= 0:
                amplitude = 1.0
                linear_filter = SpatioTemporalFilter(spatial_filter, temporal_filter, amplitude=amplitude)
                cell = OnUnit(linear_filter, transfer_function)
            elif cell_type.find('OFF') >= 0:
                amplitude = -1.0
                linear_filter = SpatioTemporalFilter(spatial_filter, temporal_filter, amplitude=amplitude)
                cell = OffUnit(linear_filter, transfer_function)

        t, f_tot = cell.evaluate(movie_to_show, downsample=1, separable=True)  # Taking the second movie which is 4 Hz


        if counter == 0:
            firing_rates = np.zeros([len(LGN.nodes()) + 1, len(t)])
            firing_rates[0, :] = t
            time = np.asarray(t)
        firing_rates[counter + 1, :] = f_tot
        #print('counter:',counter,'rate:',f_tot[:3])

    # a = np.asarray(firing_rates)
    # np.savetxt(output_file_name + "_f_tot.csv", a, delimiter=" ")

    f = h5.File(output_file_name + '_f_tot.h5', 'w')
    f.create_dataset('time', data=time)
    f.create_dataset('firing_rates_Hz', data=firing_rates[1:, :])
    f.close()



def calculate_firing_rate_fast(LGN, stimulus, output_file_name, duration, gray_screen,  cpd, TF, direction, contrast):

    origin = (0.,0.)

    if (gray_screen + 4.0 * (1/float(TF)) ) > (gray_screen + 1.0):
        T = gray_screen + 4.0 * (1/float(TF))
    else:
        T = gray_screen + 1.0
    # Gratings simulation
    movie_to_show = GratingMovie(120, 240).create_movie(t_min = 0, t_max = T, gray_screen_dur = gray_screen, cpd = cpd, temporal_f = TF, theta = direction, contrast = contrast/100.) #, row_size_new = 120, col_size_new = 240)
    # movie_to_show = FullFieldFlashMovie(range(120), range(240),1,2).full(t_max=2)
    # gr_dir_name = '/data/mat/antona/network/14-simulations/6-LGN_firing_rates_and_spikes/LGN_spike_trains/movies_gratings/res_192'
    # gr_mov_list = create_grating_movie_list(gr_dir_name)


    for counter, node in enumerate(LGN.nodes()):

        # For spatial filter and locations
        translate = (node[1]['position'][0], node[1]['position'][1])
        sigma = node[1]['spatial_size'] / 3.  # convert from degree to SD
        sigma = (sigma, sigma)
        spatial_filter = GaussianSpatialFilter(translate=translate, sigma=sigma, origin=origin)


        ################# Extract cell parameters needed      #################
        if node[1]['model_id'] == 'sONsOFF_001':

            # sON temporal filter
            sON_prs = {'opt_wts': [node[1]['weight_non_dom_0'], node[1]['weight_non_dom_1']],
                        'opt_kpeaks': [node[1]['kpeaks_non_dom_0'], node[1]['kpeaks_non_dom_1']],
                        'opt_delays': [node[1]['delay_non_dom_0'], node[1]['delay_non_dom_1']]}
            sON_filt_new = createOneUnitOfTwoSubunitFilter(sON_prs, 121.0)
            sON_sum = sON_filt_new[1]
            sON_filt_new = sON_filt_new[0]

            # tOFF temporal filter
            sOFF_prs = {'opt_wts': [node[1]['weight_dom_0'], node[1]['weight_dom_1']],
                        'opt_kpeaks': [node[1]['kpeaks_dom_0'], node[1]['kpeaks_dom_1']],
                        'opt_delays': [node[1]['delay_dom_0'], node[1]['delay_dom_1']]}
            sOFF_filt_new = createOneUnitOfTwoSubunitFilter(sOFF_prs, 115.0)
            sOFF_sum = sOFF_filt_new[1]
            sOFF_filt_new = sOFF_filt_new[0]

            amp_on = 1.0  # set the non-dominant subunit amplitude to unity
            spont = 4.0
            max_roff = 35.0
            max_ron = 21.0
            amp_off = -(max_roff / max_ron) * (sON_sum / sOFF_sum) * amp_on - (spont * (max_roff - max_ron)) / (
            max_ron * sOFF_sum)

            # Create sON subunit:
            xfer_fn_son = ScalarTransferFunction('Heaviside(s+' + str(0.5 * spont) + ')*(s+' + str(0.5 * spont) + ')')
            linear_filter_son = SpatioTemporalFilter(spatial_filter, sON_filt_new, amplitude=amp_on)
            scell_on = OnUnit(linear_filter_son, xfer_fn_son)

            # Create sOFF subunit:
            xfer_fn_soff = ScalarTransferFunction('Heaviside(s+' + str(0.5 * spont) + ')*(s+' + str(0.5 * spont) + ')')
            linear_filter_soff = SpatioTemporalFilter(spatial_filter, sOFF_filt_new, amplitude=amp_off)
            scell_off = OffUnit(linear_filter_soff, xfer_fn_soff)

            sep_ss_onoff_cell = create_two_sub_cell(linear_filter_soff, linear_filter_son, 0.5 * spont, 0.5 * spont,
                                                    node[1]['tuning_angle'], node[1]['sf_sep'], translate)
            cell = sep_ss_onoff_cell

            #t, f_tot = cell.evaluate(movie_to_show, downsample=1, separable = True)  # Taking the second movie which is 4 Hz


        elif node[1]['model_id'] == 'sONtOFF_001':
            # spatial_filter.get_kernel(np.arange(120), np.arange(240)).imshow()
            # sON temporal filter
            sON_prs = {'opt_wts': [node[1]['weight_non_dom_0'], node[1]['weight_non_dom_1']],
                       'opt_kpeaks': [node[1]['kpeaks_non_dom_0'], node[1]['kpeaks_non_dom_1']],
                       'opt_delays': [node[1]['delay_non_dom_0'], node[1]['delay_non_dom_1']]}
            sON_filt_new = createOneUnitOfTwoSubunitFilter(sON_prs, 93.5)
            sON_sum = sON_filt_new[1]
            sON_filt_new = sON_filt_new[0]

            # tOFF temporal filter
            tOFF_prs = {'opt_wts': [node[1]['weight_dom_0'], node[1]['weight_dom_1']],
                        'opt_kpeaks': [node[1]['kpeaks_dom_0'], node[1]['kpeaks_dom_1']],
                        'opt_delays': [node[1]['delay_dom_0'], node[1]['delay_dom_1']]}
            tOFF_filt_new = createOneUnitOfTwoSubunitFilter(tOFF_prs, 64.8)   #64.8
            tOFF_sum = tOFF_filt_new[1]
            tOFF_filt_new = tOFF_filt_new[0]

            amp_on = 1.0  # set the non-dominant subunit amplitude to unity
            spont = 5.5
            max_roff = 46.0
            max_ron = 31.0
            amp_off = -0.7 * (max_roff / max_ron) * (sON_sum / tOFF_sum) * amp_on - (spont * (max_roff - max_ron)) / (
            max_ron * tOFF_sum)

            # Create sON subunit:
            xfer_fn_son = ScalarTransferFunction('Heaviside(s+' + str(0.5 * spont) + ')*(s+' + str(0.5 * spont) + ')')
            linear_filter_son = SpatioTemporalFilter(spatial_filter, sON_filt_new, amplitude=amp_on)
            scell_on = OnUnit(linear_filter_son, xfer_fn_son)
            # linear_filter_son.spatial_filter.get_kernel(np.arange(120), np.arange(240)).imshow()

            # Create tOFF subunit:
            xfer_fn_toff = ScalarTransferFunction('Heaviside(s+' + str(0.5 * spont) + ')*(s+' + str(0.5 * spont) + ')')
            linear_filter_toff = SpatioTemporalFilter(spatial_filter, tOFF_filt_new, amplitude=amp_off)
            tcell_off = OffUnit(linear_filter_toff, xfer_fn_toff)
            # linear_filter_toff.spatial_filter.get_kernel(np.arange(120), np.arange(240)).kernel


            sep_ts_onoff_cell = create_two_sub_cell(linear_filter_toff, linear_filter_son, 0.5 * spont, 0.5 * spont,
                                                    node[1]['tuning_angle'], node[1]['sf_sep'], translate)
            # sep_ts_onoff_cell.show_temporal_filter()

            cell = sep_ts_onoff_cell
            #t, f_tot = cell.evaluate(movie_to_show, downsample=1, separable = True)  # Taking the second movie which is 4 Hz
        else:
            cell_type = node[1]['model_id'][0: node[1]['model_id'].find('_')]  # 'sON'  # 'tOFF'
            tf_str = node[1]['model_id'][node[1]['model_id'].find('_') + 1:]

            # For temporal filter
            wts    = [node[1]['weight_dom_0'], node[1]['weight_dom_1']]
            kpeaks = [node[1]['kpeaks_dom_0'], node[1]['kpeaks_dom_1']]
            delays = [node[1]['delay_dom_0'],  node[1]['delay_dom_1']]

            ################# End of extract cell parameters needed   #################

            # Get spont from experimental data
            exp_prs_dict = get_data_metrics_for_each_subclass(cell_type)
            subclass_prs_dict = exp_prs_dict[tf_str]
            spont_exp = subclass_prs_dict['spont_exp']
            spont_str = str(spont_exp[0])

            # Get filters
            transfer_function = ScalarTransferFunction('Heaviside(s+' + spont_str + ')*(s+' + spont_str + ')')
            temporal_filter = TemporalFilterCosineBump(wts, kpeaks, delays)

            if cell_type.find('ON') >= 0:
                amplitude = 1.0
                linear_filter = SpatioTemporalFilter(spatial_filter, temporal_filter, amplitude=amplitude)
                cell = OnUnit(linear_filter, transfer_function)
            elif cell_type.find('OFF') >= 0:
                amplitude = -1.0
                linear_filter = SpatioTemporalFilter(spatial_filter, temporal_filter, amplitude=amplitude)
                cell = OffUnit(linear_filter, transfer_function)

        t, f_tot = cell.evaluate(movie_to_show, downsample=1, separable=True)

        sampling_rate = 1000.
        print('before', len(f_tot[int(-2 / float(TF) * sampling_rate):]))
        f_tot = np.append(f_tot, np.tile(f_tot[int(-1 / float(TF) * sampling_rate):], int(np.ceil((duration - T) * TF))) )
        print('after', f_tot[int(-1 / float(TF) * sampling_rate -2):])
        f_tot = f_tot[:duration*sampling_rate + 1]
        t = np.linspace(0, duration, len(f_tot))


        if counter == 0:
            firing_rates = np.zeros([len(LGN.nodes()) + 1, len(t)])
            firing_rates[0, :] = t
        firing_rates[counter + 1, :] = f_tot

    a = np.asarray(firing_rates)
    np.savetxt(output_file_name + "_f_tot.csv", a, delimiter=" ")


def generate_spikes(LGN, trials, duration, output_file_name):

    # f_tot = np.loadtxt(output_file_name + "_f_tot.csv", delimiter=" ")
    # t = f_tot[0, :]

    f = h5.File(output_file_name + "_f_tot.h5", 'r')
    f_tot = np.array(f.get('firing_rates_Hz'))

    # t = np.array(f.get('time'))
    # For h5 files that don't have time explicitly saved
    t = np.linspace(0, duration, f_tot.shape[1])

    print('firing rates has been generated')
    #create output file
   #f = nwb.create_blank_file(output_file_name + '_spikes.nwb', force=True)
    f = create_blank_file(output_file_name + '_spikes.nwb', force=True) 
   
    for trial in range(trials):
        for counter in range(len(LGN.nodes())):
            try:
                spike_train = np.array(f_rate_to_spike_train(t*1000., f_tot[counter, :], np.random.randint(10000), 1000.*min(t), 1000.*max(t), 0.1))
            except:
                spike_train = 1000.*np.array(pg.generate_inhomogenous_poisson(t, f_tot[counter, :], seed=np.random.randint(10000))) #convert to milliseconds and hence the multiplication by 1000

            #nwb.SpikeTrain(spike_train, unit='millisecond').add_to_processing(f, 'trial_%s' % trial)
            SpikeTrain(spike_train, unit='millisecond').add_to_processing(f, 'trial_%s' % trial)
    f.close()


# def generate_spikes(LGN, trials, output_file_name):
#
#     f_tot = np.loadtxt(output_file_name + "_f_tot.csv", delimiter=" ")
#     t = f_tot[0, :]
#
#     #create output file
#     f = nwb.create_blank_file(output_file_name + '_spikes.nwb', force=True)
#
#     for trial in range(trials):
#         for counter in range(len(LGN.nodes())):
#             try:
#                 spike_train = np.array(f_rate_to_spike_train(t*1000., f_tot[counter + 1, :], np.random.randint(10000), 1000.*min(t), 1000.*max(t), 0.1))
#             except:
#                 spike_train = 1000.*np.array(pg.generate_inhomogenous_poisson(t, f_tot[counter + 1, :], seed=np.random.randint(10000))) #convert to milliseconds and hence the multiplication by 1000
#
#             nwb.SpikeTrain(spike_train, unit='millisecond').add_to_processing(f, 'trial_%s' % trial)
#     f.close()


def f_rate_to_spike_train(t, f_rate, random_seed, t_window_start, t_window_end, p_spike_max):
  # t and f_rate are lists containing time stamps and corresponding firing rate values;
  # they are assumed to be of the same length and ordered with the time strictly increasing;
  # p_spike_max is the maximal probability of spiking that we allow within the time bin; it is used to decide on the size of the time bin; should be less than 1!

  if np.max(f_rate) * np.max(np.diff(t))/1000. > 0.1:   #Divide by 1000 to convert to seconds
      print('Firing rate to high for time interval and will not estimate spike correctly. Spikes will ' \
            'be calculated with the slower inhomogenous poisson generating fucntion')
      raise

  spike_times = []

  # Use seed(...) to instantiate the random number generator.  Otherwise, current system time is used.
  seed(random_seed)

  # Assume here for each pair (t[k], f_rate[k]) that the f_rate[k] value applies to the time interval [t[k], t[k+1]).
  for k in range(0, len(f_rate)-1):
    t_k = t[k]
    t_k_1 = t[k+1]
    if ((t_k >= t_window_start) and (t_k_1 <= t_window_end)):
      delta_t = t_k_1 - t_k
      av_N_spikes = f_rate[k] / 1000.0 * delta_t # Average number of spikes expected in this interval (note that firing rate is in Hz and time is in ms).

      if (av_N_spikes > 0):
        if (av_N_spikes <= p_spike_max):
          N_bins = 1
        else:
          N_bins = int(ceil(av_N_spikes / p_spike_max))

        t_base = t[k]
        t_bin = 1.0 * delta_t / N_bins
        p_spike_bin = 1.0 * av_N_spikes / N_bins
        for i_bin in range(0, N_bins):
          rand_tmp = random()
          if rand_tmp < p_spike_bin:
            spike_t = t_base + random() * t_bin
            spike_times.append(spike_t)

          t_base += t_bin

  return spike_times






def select_source_cells(sources, target, lgn_mean, lgn_models):

    target_id = target['id']
    source_ids = [s['id'] for s in sources]

    parametersDictionary= get_params_dictionary()
    pop_name = [key for key in parametersDictionary if key in target['pop_name']][0]

    # Check if target supposed to get a connection and if not, then no need to keep calculating.
    if np.random.random() > parametersDictionary[pop_name]['probability']:
        return [None] * len(source_ids)


    if target_id%250 == 0:
        print("connection LGN cells to L4 cell #", target_id)

    subfields_centers_distance_min = parametersDictionary[pop_name]['centers_d_min'] # 10.0
    subfields_centers_distance_max = parametersDictionary[pop_name]['centers_d_max'] #11.0  # 11.0
    subfields_centers_distance_L = subfields_centers_distance_max - subfields_centers_distance_min

    subfields_ON_OFF_width_min = parametersDictionary[pop_name]['ON_OFF_w_min'] #6.0  # 8.0 #10.0 #8.0 #8.0 #14.0 #15.0
    subfields_ON_OFF_width_max = parametersDictionary[pop_name]['ON_OFF_w_max'] #8.0  # 10.0 #12.0 #10.0 #15.0 #20.0 #15.0
    subfields_ON_OFF_width_L = subfields_ON_OFF_width_max - subfields_ON_OFF_width_min

    subfields_width_aspect_ratio_min = parametersDictionary[pop_name]['aspectRatio_min']# 2.8  # 1.9 #1.4 #0.9 #1.0
    subfields_width_aspect_ratio_max = parametersDictionary[pop_name]['aspectRatio_max']# 3.0  # 2.0 #1.5 #1.1 #1.0
    subfields_width_aspect_ratio_L = subfields_width_aspect_ratio_max - subfields_width_aspect_ratio_min

    # Convert to lin_degrees as what is used by the function select_source_cells below
    # There is a corresponding write-up with screen shots that explains the below conversion. Briefly:
    # From Niell et. al, chapter 29 of "The New Visual Neurosciences", Fig 29.1D, a square with ~0.5 mm on a side
    # corresponds to ~35 degrees in x (azimuth) and ~20 degrees in z (elevation).
    # Also, from the figure, we can assume that in both the azimuth and elevation directions, the scale
    # is approximately constant and not warped. Hence, for the x (azimuth) and z (elevation),
    # the visual degree traversed per mm of cortex can be determined:
    # In azimuth, 35/0.5 = 70 degs/mm
    # In elevation, 20/0.5 = 40 degs/mm
    # From this we can convert a translation in x & z in cortex to a translation in visual space.
    # For example, consider moving 0.85mm in the azimuth, the movement in visual space is then estimated
    # to be 0.85 * 70 = 59.5 degrees.
    # The x and z poistions are then converted to linear degrees: tan(x) * (180/pi)
    # Note that before the tangent is taken, the angle is converted to radians
    # The same conversion was done for the mean and dimensions.
    x_position_lin_degrees = convert_x_to_lindegs(target['position'][0])
    y_position_lin_degrees = convert_z_to_lindegs(target['position'][2])

    vis_x = lgn_mean[0] + ((x_position_lin_degrees))# - l4_mean[0]) / l4_dim[0]) * lgn_dim[0]
    vis_y = lgn_mean[1] + ((y_position_lin_degrees))# - l4_mean[2]) / l4_dim[2]) * lgn_dim[1]

    ellipse_center_x0 = vis_x #tar_cells[tar_gid]['vis_x']
    ellipse_center_y0 = vis_y #tar_cells[tar_gid]['vis_y']

    tuning_angle = float(target['tuning_angle'])
    tuning_angle = None if math.isnan(tuning_angle) else tuning_angle
    #tuning_angle = None if math.isnan(target['tuning_angle']) else target['tuning_angle']
    if tuning_angle is None:
        ellipse_b0 = (subfields_ON_OFF_width_min + random() * subfields_ON_OFF_width_L) / 2.0  # Divide by 2 to convert from width to radius.
        ellipse_b0 = 2.5 * ellipse_b0  # 1.5 * ellipse_b0
        ellipse_a0 = ellipse_b0  # ellipse_b0
        top_N_src_cells_subfield = 15  # 20
        ellipses_centers_halfdistance = 0.0
    else:
        tuning_angle_value = float(tuning_angle)
        ellipses_centers_halfdistance = (subfields_centers_distance_min + random() * subfields_centers_distance_L) / 2.0
        ellipse_b0 = (subfields_ON_OFF_width_min + random() * subfields_ON_OFF_width_L) / 2.0  # Divide by 2 to convert from width to radius.
        ellipse_a0 = ellipse_b0 * (subfields_width_aspect_ratio_min + random() * subfields_width_aspect_ratio_L)
        ellipse_phi = tuning_angle_value + 180.0 + 90.0  # Angle, in degrees, describing the rotation of the canonical ellipse away from the x-axis.
        ellipse_cos_mphi = math.cos(-math.radians(ellipse_phi))
        ellipse_sin_mphi = math.sin(-math.radians(ellipse_phi))
        top_N_src_cells_subfield = 8  # 10 #9

        ################################################################################################################################
        probability_sON = parametersDictionary[pop_name]['sON_ratio']
        if np.random.random() < probability_sON:
            cell_sustained_unit = 'sON_'
        else:
            cell_sustained_unit = 'sOFF_'

    cell_TF = np.random.poisson(parametersDictionary[pop_name]['poissonParameter'])
    while cell_TF <= 0:
        cell_TF = np.random.poisson(parametersDictionary[pop_name]['poissonParameter'])

    sON_subunits = np.array([1., 2., 4., 8.])
    sON_sum = np.sum(abs(cell_TF - sON_subunits))
    p_sON = (1 - abs(cell_TF - sON_subunits) / sON_sum) / (len(sON_subunits) - 1)

    sOFF_subunits = np.array([1., 2., 4., 8., 15.])
    sOFF_sum = np.sum(abs(cell_TF - sOFF_subunits))
    p_sOFF = (1 - abs(cell_TF - sOFF_subunits) / sOFF_sum) / (len(sOFF_subunits) - 1)

    tOFF_subunits = np.array([4., 8., 15.])
    tOFF_sum = np.sum(abs(cell_TF - tOFF_subunits))
    p_tOFF = (1 - abs(cell_TF - tOFF_subunits) / tOFF_sum) / (len(tOFF_subunits) - 1)

    # to match previous algorithm reorganize source cells by type
    cell_type_dict = {}
    for lgn_model in lgn_models:
        cell_type_dict[lgn_model] = [(src_id, src_dict) for src_id, src_dict in zip(source_ids, sources) if src_dict['model_id'] == lgn_model]

    lgn_models_subtypes_dictionary = {
        'sON_': {'sub_types':['sON_TF1', 'sON_TF2', 'sON_TF4', 'sON_TF8'], 'probabilities': p_sON},
        'sOFF': {'sub_types':['sOFF_TF1', 'sOFF_TF2', 'sOFF_TF4', 'sOFF_TF8', 'sOFF_TF15'], 'probabilities': p_sOFF},
        'tOFF': {'sub_types':['tOFF_TF4', 'tOFF_TF8', 'tOFF_TF15'], 'probabilities': p_tOFF},
    }
    ################################################################################################################################


    # For this target cell, if it has tuning, select the input cell types
    # Note these parameters will not matter if the cell does not have tuning but are calculated anyway
    # Putting it here instead of previous if-else statement for clarity
    # cumulativeP = np.cumsum(connectivityRatios[pop_name]['probabilities'])
    # lgn_model_idx = np.where((np.random.random() < np.array(cumulativeP)) == True)[0][0]
    # sustained_Subunit = connectivityRatios[pop_name]['lgn_models'][lgn_model_idx][0]
    # transient_Subunit = connectivityRatios[pop_name]['lgn_models'][lgn_model_idx][1]
    src_cells_selected = {}
    for src_type in cell_type_dict.keys():
        src_cells_selected[src_type] = []

        if (tuning_angle is None):
            ellipse_center_x = ellipse_center_x0
            ellipse_center_y = ellipse_center_y0
            ellipse_a = ellipse_a0
            ellipse_b = ellipse_b0
        else:
            if ('tOFF_' in src_type[0:5]):
                ellipse_center_x = ellipse_center_x0 + ellipses_centers_halfdistance * ellipse_sin_mphi
                ellipse_center_y = ellipse_center_y0 + ellipses_centers_halfdistance * ellipse_cos_mphi
                ellipse_a = ellipse_a0
                ellipse_b = ellipse_b0
            elif ('sON_' in src_type[0:5] or 'sOFF_' in src_type[0:5]):
                ellipse_center_x = ellipse_center_x0 - ellipses_centers_halfdistance * ellipse_sin_mphi
                ellipse_center_y = ellipse_center_y0 - ellipses_centers_halfdistance * ellipse_cos_mphi
                ellipse_a = ellipse_a0
                ellipse_b = ellipse_b0
            else:
                # Make this a simple circle.
                ellipse_center_x = ellipse_center_x0
                ellipse_center_y = ellipse_center_y0
                # Make the region from which source cells are selected a bit smaller for the transient_ON_OFF cells, since each
                # source cell in this case produces both ON and OFF responses.
                ellipse_b = ellipses_centers_halfdistance/2.0 #0.01 #ellipses_centers_halfdistance + 1.0*ellipse_b0 #0.01 #0.5 * ellipse_b0 # 0.8 * ellipse_b0
                ellipse_a = ellipse_b0 #0.01 #ellipse_b0


        # Find those source cells of the appropriate type that have their visual space coordinates within the ellipse.
        for src_id, src_dict in cell_type_dict[src_type]:
            x, y = (src_dict['position'][0], src_dict['position'][1])

            x = x - ellipse_center_x
            y = y - ellipse_center_y

            x_new = x
            y_new = y
            if tuning_angle is not None:
                x_new = x * ellipse_cos_mphi - y * ellipse_sin_mphi
                y_new = x * ellipse_sin_mphi + y * ellipse_cos_mphi

            if (((x_new / ellipse_a) ** 2 + (y_new / ellipse_b) ** 2) <= 1.0):
                if (tuning_angle is not None):
                    if (src_type == 'sONsOFF_001' or src_type == 'sONtOFF_001'):
                        src_tuning_angle = float(src_dict['tuning_angle'])
                        delta_tuning = abs(abs(abs(180.0 - abs(tuning_angle_value - src_tuning_angle) % 360.0) - 90.0) - 90.0)
                        if (delta_tuning < 15.0):
                            src_cells_selected[src_type].append(src_id)

                    # elif src_type in ['sONtOFF_001']:
                    #     src_cells_selected[src_type].append(src_id)

                    elif cell_sustained_unit in src_type[:5]:
                        selection_probability = get_selection_probability(src_type, lgn_models_subtypes_dictionary)
                        if np.random.random() < selection_probability:
                            src_cells_selected[src_type].append(src_id)

                    elif 'tOFF_' in src_type[:5]:
                        selection_probability = get_selection_probability(src_type, lgn_models_subtypes_dictionary)
                        if np.random.random() < selection_probability:
                            src_cells_selected[src_type].append(src_id)

                else:
                    if (src_type == 'sONsOFF_001' or src_type == 'sONtOFF_001'):
                        src_cells_selected[src_type].append(src_id)
                    else:
                        selection_probability = get_selection_probability(src_type, lgn_models_subtypes_dictionary)
                        if np.random.random() < selection_probability:
                            src_cells_selected[src_type].append(src_id)

    select_cell_ids = [id for _, selected in src_cells_selected.items() for id in selected]

    # if len(select_cell_ids) > 30:
    #     select_cell_ids = np.random.choice(select_cell_ids, 30, replace=False)
    nsyns_ret = [parametersDictionary[pop_name]['N_syn'] if id in select_cell_ids else None for id in source_ids]
    return nsyns_ret



def get_selection_probability(src_type, lgn_models_subtypes_dictionary):

    current_model_subtypes = lgn_models_subtypes_dictionary[src_type[0:4]]['sub_types']
    current_model_probabilities = lgn_models_subtypes_dictionary[src_type[0:4]]['probabilities']
    lgn_model_idx = [i for i, model in enumerate(current_model_subtypes) if src_type == model][0]
    return current_model_probabilities[lgn_model_idx]


def convert_x_to_lindegs(xcoords):
    return np.tan(0.07 * np.array(xcoords) * np.pi / 180.) * 180.0 / np.pi

def convert_z_to_lindegs(zcoords):
    return np.tan(0.04 * np.array(zcoords) * np.pi / 180.) * 180.0 / np.pi


def get_params_dictionary():

    params_dict = {
        'i1Htr3a'  : {'probability': 0.588, 'poissonParameter': 2.0, 'sON_ratio': 0.75, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 8.50, 'ON_OFF_w_max': 11.0, 'aspectRatio_min': 2.2, 'aspectRatio_max': 2.4, 'N_syn': 10},
        'e23'      : {'probability': 0.789, 'poissonParameter': 1.5, 'sON_ratio': 0.90, 'centers_d_min': 4.00, 'centers_d_max': 6.00, 'ON_OFF_w_min': 7.50, 'ON_OFF_w_max': 9.50, 'aspectRatio_min': 3.4, 'aspectRatio_max': 3.6, 'N_syn': 15},
        'i23Pvalb' : {'probability': 0.824, 'poissonParameter': 2.0, 'sON_ratio': 0.75, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 10.0, 'ON_OFF_w_max': 13.0, 'aspectRatio_min': 1.6, 'aspectRatio_max': 1.8, 'N_syn': 15},
        'i23Sst'   : {'probability': 0.000, 'poissonParameter': 2.0, 'sON_ratio': 0.75, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 8.50, 'ON_OFF_w_max': 11.0, 'aspectRatio_min': 2.2, 'aspectRatio_max': 2.4, 'N_syn': 00},
        'i23Htr3a' : {'probability': 0.000, 'poissonParameter': 2.0, 'sON_ratio': 0.75, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 8.50, 'ON_OFF_w_max': 11.0, 'aspectRatio_min': 2.2, 'aspectRatio_max': 2.4, 'N_syn': 00},
        'e4'       : {'probability': 1.000, 'poissonParameter': 2.0, 'sON_ratio': 0.90, 'centers_d_min': 4.00, 'centers_d_max': 6.00, 'ON_OFF_w_min': 7.50, 'ON_OFF_w_max': 9.50, 'aspectRatio_min': 3.4, 'aspectRatio_max': 3.6, 'N_syn': 80},
        'i4Pvalb'  : {'probability': 1.000, 'poissonParameter': 2.0, 'sON_ratio': 0.75, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 10.0, 'ON_OFF_w_max': 13.0, 'aspectRatio_min': 1.6, 'aspectRatio_max': 1.8, 'N_syn': 75},
        'i4Sst'    : {'probability': 0.333, 'poissonParameter': 2.0, 'sON_ratio': 0.75, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 8.50, 'ON_OFF_w_max': 11.0, 'aspectRatio_min': 2.2, 'aspectRatio_max': 2.4, 'N_syn': 25},
        'i4Htr3a'  : {'probability': 0.444, 'poissonParameter': 2.0, 'sON_ratio': 0.75, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 8.50, 'ON_OFF_w_max': 11.0, 'aspectRatio_min': 2.2, 'aspectRatio_max': 2.4, 'N_syn': 20},
        'e5'       : {'probability': 1.000, 'poissonParameter': 1.5, 'sON_ratio': 0.50, 'centers_d_min': 8.00, 'centers_d_max': 12.0, 'ON_OFF_w_min': 12.0, 'ON_OFF_w_max': 16.0, 'aspectRatio_min': 1.6, 'aspectRatio_max': 1.8, 'N_syn': 15},
        'i5Pvalb'  : {'probability': 1.000, 'poissonParameter': 2.0, 'sON_ratio': 0.50, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 10.0, 'ON_OFF_w_max': 13.0, 'aspectRatio_min': 1.6, 'aspectRatio_max': 1.8, 'N_syn': 20},
        'i5Sst'    : {'probability': 0.000, 'poissonParameter': 2.0, 'sON_ratio': 0.50, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 8.50, 'ON_OFF_w_max': 11.0, 'aspectRatio_min': 2.2, 'aspectRatio_max': 2.4, 'N_syn': 00},
        'i5Htr3a'  : {'probability': 0.000, 'poissonParameter': 2.0, 'sON_ratio': 0.50, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 8.50, 'ON_OFF_w_max': 11.0, 'aspectRatio_min': 2.2, 'aspectRatio_max': 2.4, 'N_syn': 00},
        'e6'       : {'probability': 0.778, 'poissonParameter': 1.5, 'sON_ratio': 0.90, 'centers_d_min': 3.00, 'centers_d_max': 4.00, 'ON_OFF_w_min': 9.00, 'ON_OFF_w_max': 11.0, 'aspectRatio_min': 3.4, 'aspectRatio_max': 3.6, 'N_syn': 15},
        'i6Pvalb'  : {'probability': 0.818, 'poissonParameter': 2.0, 'sON_ratio': 0.75, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 10.0, 'ON_OFF_w_max': 13.0, 'aspectRatio_min': 1.6, 'aspectRatio_max': 1.8, 'N_syn': 10},
        'i6Sst'    : {'probability': 0.000, 'poissonParameter': 2.0, 'sON_ratio': 0.75, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 8.50, 'ON_OFF_w_max': 11.0, 'aspectRatio_min': 2.2, 'aspectRatio_max': 2.4, 'N_syn': 00},
        'i6Htr3a'  : {'probability': 0.000, 'poissonParameter': 2.0, 'sON_ratio': 0.75, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 8.50, 'ON_OFF_w_max': 11.0, 'aspectRatio_min': 2.2, 'aspectRatio_max': 2.4, 'N_syn': 00}
        }

    # params_dict = {
    #     'i1Htr3a'  : {'probability': 0.588, 'poissonParameter': 2.0, 'sON_ratio': 0.75, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 8.50, 'ON_OFF_w_max': 11.0, 'aspectRatio_min': 2.2, 'aspectRatio_max': 2.4, 'N_syn': 10},
    #     'e23'      : {'probability': 0.789, 'poissonParameter': 1.5, 'sON_ratio': 0.90, 'centers_d_min': 8.00, 'centers_d_max': 12.0, 'ON_OFF_w_min': 7.50, 'ON_OFF_w_max': 9.50, 'aspectRatio_min': 3.4, 'aspectRatio_max': 3.6, 'N_syn': 15},
    #     'i23Pvalb' : {'probability': 0.824, 'poissonParameter': 2.0, 'sON_ratio': 0.75, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 10.0, 'ON_OFF_w_max': 13.0, 'aspectRatio_min': 1.6, 'aspectRatio_max': 1.8, 'N_syn': 15},
    #     'i23Sst'   : {'probability': 0.000, 'poissonParameter': 2.0, 'sON_ratio': 0.75, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 8.50, 'ON_OFF_w_max': 11.0, 'aspectRatio_min': 2.2, 'aspectRatio_max': 2.4, 'N_syn': 00},
    #     'i23Htr3a' : {'probability': 0.000, 'poissonParameter': 2.0, 'sON_ratio': 0.75, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 8.50, 'ON_OFF_w_max': 11.0, 'aspectRatio_min': 2.2, 'aspectRatio_max': 2.4, 'N_syn': 00},
    #     'e4'       : {'probability': 1.000, 'poissonParameter': 2.0, 'sON_ratio': 0.90, 'centers_d_min': 8.00, 'centers_d_max': 12.0, 'ON_OFF_w_min': 7.50, 'ON_OFF_w_max': 9.50, 'aspectRatio_min': 3.4, 'aspectRatio_max': 3.6, 'N_syn': 80},
    #     'i4Pvalb'  : {'probability': 1.000, 'poissonParameter': 2.0, 'sON_ratio': 0.75, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 10.0, 'ON_OFF_w_max': 13.0, 'aspectRatio_min': 1.6, 'aspectRatio_max': 1.8, 'N_syn': 75},
    #     'i4Sst'    : {'probability': 0.333, 'poissonParameter': 2.0, 'sON_ratio': 0.75, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 8.50, 'ON_OFF_w_max': 11.0, 'aspectRatio_min': 2.2, 'aspectRatio_max': 2.4, 'N_syn': 25},
    #     'i4Htr3a'  : {'probability': 0.444, 'poissonParameter': 2.0, 'sON_ratio': 0.75, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 8.50, 'ON_OFF_w_max': 11.0, 'aspectRatio_min': 2.2, 'aspectRatio_max': 2.4, 'N_syn': 20},
    #     'e5'       : {'probability': 1.000, 'poissonParameter': 1.5, 'sON_ratio': 0.50, 'centers_d_min': 8.00, 'centers_d_max': 12.0, 'ON_OFF_w_min': 12.0, 'ON_OFF_w_max': 16.0, 'aspectRatio_min': 1.6, 'aspectRatio_max': 1.8, 'N_syn': 15},
    #     'i5Pvalb'  : {'probability': 1.000, 'poissonParameter': 2.0, 'sON_ratio': 0.50, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 10.0, 'ON_OFF_w_max': 13.0, 'aspectRatio_min': 1.6, 'aspectRatio_max': 1.8, 'N_syn': 20},
    #     'i5Sst'    : {'probability': 0.000, 'poissonParameter': 2.0, 'sON_ratio': 0.50, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 8.50, 'ON_OFF_w_max': 11.0, 'aspectRatio_min': 2.2, 'aspectRatio_max': 2.4, 'N_syn': 00},
    #     'i5Htr3a'  : {'probability': 0.000, 'poissonParameter': 2.0, 'sON_ratio': 0.50, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 8.50, 'ON_OFF_w_max': 11.0, 'aspectRatio_min': 2.2, 'aspectRatio_max': 2.4, 'N_syn': 00},
    #     'e6'       : {'probability': 0.778, 'poissonParameter': 1.5, 'sON_ratio': 0.90, 'centers_d_min': 4.00, 'centers_d_max': 8.00, 'ON_OFF_w_min': 9.00, 'ON_OFF_w_max': 11.0, 'aspectRatio_min': 2.8, 'aspectRatio_max': 3.0, 'N_syn': 15},
    #     'i6Pvalb'  : {'probability': 0.818, 'poissonParameter': 2.0, 'sON_ratio': 0.75, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 10.0, 'ON_OFF_w_max': 13.0, 'aspectRatio_min': 1.6, 'aspectRatio_max': 1.8, 'N_syn': 15},
    #     'i6Sst'    : {'probability': 0.000, 'poissonParameter': 2.0, 'sON_ratio': 0.75, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 8.50, 'ON_OFF_w_max': 11.0, 'aspectRatio_min': 2.2, 'aspectRatio_max': 2.4, 'N_syn': 00},
    #     'i6Htr3a'  : {'probability': 0.000, 'poissonParameter': 2.0, 'sON_ratio': 0.75, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 8.50, 'ON_OFF_w_max': 11.0, 'aspectRatio_min': 2.2, 'aspectRatio_max': 2.4, 'N_syn': 00}
    #     }

    # params_dict = {
    #     'i1Htr3a'  : {'probability': 1.000, 'poissonParameter': 2.0, 'sON_ratio': 0.75, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 8.50, 'ON_OFF_w_max': 11.0, 'aspectRatio_min': 2.2, 'aspectRatio_max': 2.4, 'N_syn': 10},
    #     'e23'      : {'probability': 1.000, 'poissonParameter': 1.5, 'sON_ratio': 1.00, 'centers_d_min': 8.00, 'centers_d_max': 8.00, 'ON_OFF_w_min': 6.50, 'ON_OFF_w_max': 6.50, 'aspectRatio_min': 3.8, 'aspectRatio_max': 4.0, 'N_syn': 15},
    #     'i23Pvalb' : {'probability': 1.000, 'poissonParameter': 2.0, 'sON_ratio': 0.75, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 10.0, 'ON_OFF_w_max': 13.0, 'aspectRatio_min': 1.6, 'aspectRatio_max': 1.8, 'N_syn': 15},
    #     'i23Sst'   : {'probability': 0.000, 'poissonParameter': 2.0, 'sON_ratio': 0.75, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 8.50, 'ON_OFF_w_max': 11.0, 'aspectRatio_min': 2.2, 'aspectRatio_max': 2.4, 'N_syn': 00},
    #     'i23Htr3a' : {'probability': 0.000, 'poissonParameter': 2.0, 'sON_ratio': 0.75, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 8.50, 'ON_OFF_w_max': 11.0, 'aspectRatio_min': 2.2, 'aspectRatio_max': 2.4, 'N_syn': 00},
    #     'e4'       : {'probability': 1.000, 'poissonParameter': 2.0, 'sON_ratio': 1.00, 'centers_d_min': 8.00, 'centers_d_max': 8.00, 'ON_OFF_w_min': 6.50, 'ON_OFF_w_max': 6.50, 'aspectRatio_min': 3.8, 'aspectRatio_max': 4.0, 'N_syn': 80},
    #     'i4Pvalb'  : {'probability': 1.000, 'poissonParameter': 2.0, 'sON_ratio': 0.75, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 10.0, 'ON_OFF_w_max': 13.0, 'aspectRatio_min': 1.6, 'aspectRatio_max': 1.8, 'N_syn': 75},
    #     'i4Sst'    : {'probability': 1.000, 'poissonParameter': 2.0, 'sON_ratio': 0.75, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 8.50, 'ON_OFF_w_max': 11.0, 'aspectRatio_min': 2.2, 'aspectRatio_max': 2.4, 'N_syn': 25},
    #     'i4Htr3a'  : {'probability': 1.000, 'poissonParameter': 2.0, 'sON_ratio': 0.75, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 8.50, 'ON_OFF_w_max': 11.0, 'aspectRatio_min': 2.2, 'aspectRatio_max': 2.4, 'N_syn': 20},
    #     'e5'       : {'probability': 1.000, 'poissonParameter': 1.5, 'sON_ratio': 1.00, 'centers_d_min': 8.00, 'centers_d_max': 12.0, 'ON_OFF_w_min': 12.0, 'ON_OFF_w_max': 16.0, 'aspectRatio_min': 1.6, 'aspectRatio_max': 1.8, 'N_syn': 15},
    #     'i5Pvalb'  : {'probability': 1.000, 'poissonParameter': 2.0, 'sON_ratio': 0.50, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 10.0, 'ON_OFF_w_max': 13.0, 'aspectRatio_min': 1.6, 'aspectRatio_max': 1.8, 'N_syn': 20},
    #     'i5Sst'    : {'probability': 0.000, 'poissonParameter': 2.0, 'sON_ratio': 0.50, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 8.50, 'ON_OFF_w_max': 11.0, 'aspectRatio_min': 2.2, 'aspectRatio_max': 2.4, 'N_syn': 00},
    #     'i5Htr3a'  : {'probability': 0.000, 'poissonParameter': 2.0, 'sON_ratio': 0.50, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 8.50, 'ON_OFF_w_max': 11.0, 'aspectRatio_min': 2.2, 'aspectRatio_max': 2.4, 'N_syn': 00},
    #     'e6'       : {'probability': 1.000, 'poissonParameter': 1.5, 'sON_ratio': 1.00, 'centers_d_min': 8.00, 'centers_d_max': 8.00, 'ON_OFF_w_min': 6.50, 'ON_OFF_w_max': 6.50, 'aspectRatio_min': 3.8, 'aspectRatio_max': 4.0, 'N_syn': 15},
    #     'i6Pvalb'  : {'probability': 1.000, 'poissonParameter': 2.0, 'sON_ratio': 0.75, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 10.0, 'ON_OFF_w_max': 13.0, 'aspectRatio_min': 1.6, 'aspectRatio_max': 1.8, 'N_syn': 15},
    #     'i6Sst'    : {'probability': 0.000, 'poissonParameter': 2.0, 'sON_ratio': 0.75, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 8.50, 'ON_OFF_w_max': 11.0, 'aspectRatio_min': 2.2, 'aspectRatio_max': 2.4, 'N_syn': 00},
    #     'i6Htr3a'  : {'probability': 0.000, 'poissonParameter': 2.0, 'sON_ratio': 0.75, 'centers_d_min': 6.00, 'centers_d_max': 10.0, 'ON_OFF_w_min': 8.50, 'ON_OFF_w_max': 11.0, 'aspectRatio_min': 2.2, 'aspectRatio_max': 2.4, 'N_syn': 00}
    #     }

    return params_dict




def read_dat_file(filename, type_mapping={'transient_ON': 'tON_001', 'transient_OFF': 'tOFF_001', 'transient_ON_OFF': 'tONOFF_001'}):
    positions_table = {val: [] for val in type_mapping.values()}
    offset_table = {val: [] for val in type_mapping.values()}
    with open(filename, 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ')
        for row in csvreader:
            model_type = type_mapping.get(row[0], None)
            if model_type:
                positions_table[model_type].append([float(row[1]), float(row[2])])
                offset_table[model_type].append([float(row[3]), float(row[4])])

    return positions_table, offset_table


def calc_tuning_angle(offset_vect):
    offset_sum = sum(offset_vect)
    if offset_sum == 0:
        return None
    else:
        tmp_vec = offset_vect / np.sqrt(offset_vect[0]**2 + offset_vect[1]**2)
        return (360.0 + 180.0 * np.arctan2(tmp_vec[1], tmp_vec[0]) / np.pi) % 360.0

##Create temporal filter given input parameters in a dictionary
def create_temporal_filter(inp_dict):
    opt_wts = inp_dict['opt_wts']
    opt_kpeaks = inp_dict['opt_kpeaks']
    opt_delays = inp_dict['opt_delays']
    temporal_filter = TemporalFilterCosineBump(opt_wts, opt_kpeaks, opt_delays)

    return temporal_filter

##Create two subunit cell object given parameters
def create_two_sub_cell(dom_lf, non_dom_lf, dom_spont, non_dom_spont, onoff_axis_angle, subfield_separation, dom_location):
    dsp = str(dom_spont)
    ndsp = str(non_dom_spont)
    two_sub_transfer_fn = MultiTransferFunction((symbolic_x, symbolic_y),'Heaviside(x+'+dsp+')*(x+'+dsp+')+Heaviside(y+'+ndsp+')*(y+'+ndsp+')')
    two_sub_cell = TwoSubfieldLinearCell(dom_lf, non_dom_lf, subfield_separation=subfield_separation,
                                         onoff_axis_angle=onoff_axis_angle, dominant_subfield_location=dom_location,
                                         transfer_function = two_sub_transfer_fn)
    return two_sub_cell

def createOneUnitOfTwoSubunitFilter(prs, ttp_exp):
    filt = create_temporal_filter(prs)
    tcross_ind = get_tcross_from_temporal_kernel(filt.get_kernel(threshold=-1.0).kernel)
    filt_sum = filt.get_kernel(threshold=-1.0).kernel[:tcross_ind].sum()

    # Calculate delay offset needed to match response latency with data and rebuild temporal filter
    del_offset = ttp_exp - tcross_ind
    if del_offset >= 0:
        delays = prs['opt_delays']
        delays[0] = delays[0] + del_offset
        delays[1] = delays[1] + del_offset
        prs['opt_delays'] = delays
        filt_new = create_temporal_filter(prs)
    else:
        print('del_offset < 0')

    return (filt_new, filt_sum)
