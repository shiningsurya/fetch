#!/usr/bin/env python3
"""

SB:
- original h5plotter was shipped in the pysigproc which I dropped as a requisite in this branch. 
    So I am adding a similar functionality here
- All parallelizations left to the user. This is just a plotting functionality so is parallelization even required here?

"""
import os
import sys
import gc
import argparse

import tqdm

import h5py

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs

from candmaker import normalize

def plot_h5(h5_file, show=False, save=True, detrend=True):
    """
    Plot the h5 candidate file
    :param h5_file: Address of the candidate h5 file
    :param show: Argument to display the plot
    :param save: Argument to save the plot
    :param detrend: Optional argument to detrend the frequency-time array
    :return:
        with h5py.File(h5_file, 'r') as f:
            dm_time = np.array(f['data_dm_time'])
            if detrend:
                freq_time = s.detrend(np.array(f['data_freq_time'])[:, ::-1].T)
            else:
                freq_time = np.array(f['data_freq_time'])[:, ::-1].T
            dm_time[dm_time != dm_time] = 0
            freq_time[freq_time != freq_time] = 0
    """
    pass

def plotter (hf, usemedian=True):
    """actual worker

    Arguments
    ---------
    hf: :h5py.File: instance
    """
    ### setup 
    gs   = mgs.GridSpec(3, 2, width_ratios=[4, 1], height_ratios=[1, 1, 1])
    axpp = fig.add_subplot (gs[0,0])
    axdd = fig.add_subplot (gs[1,0])
    axbt = fig.add_subplot (gs[2,0])
    axtt = fig.add_subplot (gs[:,1])
    ### data
    dd = normalize (np.array(hf['data_freq_time'], dtype=np.float32), usemedian=usemedian)
    bt = normalize (np.array(hf['data_dm_time'], dtype=np.float32), usemedian=usemedian)
    ###
    to_print = []
    for key,value in hf.attrs.items():
        to_print.append(f'{key} : {value}')
    str_print = '\n'.join(to_print) + '\n'
    ### meta data
    fch1, foff, nchan, dm, cand_id, tsamp, snr, width = hf.attrs['fch1'], \
                                                        hf.attrs['foff'], hf.attrs['nchans'], \
                                                        hf.attrs['dm'],   hf.attrs['cand_id'], \
                                                        hf.attrs['tsamp'],hf.attrs['snr'],\
                                                        hf.attrs['width']
    if width > 1:
        ts = np.linspace(-128,128,256) * tsamp * width*1000 / 2
    else:
        ts = np.linspace(-128,128,256) * tsamp* 1000
    ### plotting
    axpp.step (ts, dd.mean (1), where='mid', c='k', ls='-')
    axdd.imshow(dd.T, aspect='auto', extent=[ts[0], ts[-1], fch1, fch1 + (nchan * foff)], interpolation='none',cmap='jet')
    axbt.imshow(bt, aspect='auto', extent=[ts[0], ts[-1], 0, 2*dm], interpolation='none', origin='lower', cmap='jet')
    axtt.text(0.2, 0, str_print, fontsize=14, ha='left', va='bottom', wrap=True)
    ### labels
    axpp.set_ylabel('Flux [a.u.]')
    axdd.set_ylabel('Frequency [MHz]')
    axbt.set_ylabel(r'DM [pc cm$^{-3}$]')
    axbt.set_xlabel('Time [ms]')
    axtt.axis('off')
    ### sharex
    axbt.get_shared_x_axes().join (axbt, axdd)
    axbt.get_shared_x_axes().join (axbt, axpp)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='Be verbose', action='store_true')

    parser.add_argument('h5', help='HDF5 candidates.', nargs='+',)

    # parser.add_argument('-c', '--cand_file', help='candidate file. It can be either HEIMDALL or PRESTO candidates.', type=str, required=True)
    # parser.add_argument('-f', '--fil_file', help='filterbank file. ', type=str, required=True)
    # parser.add_argument('-k', '--kill_file', help='Numpy readable kill file. ', type=str, default=None)

    parser.add_argument('-O', '--odir', help='Output file directory for plots', type=str, default='./')
    values = parser.parse_args()
    #################
    if not os.path.exists (values.odir):
        os.mkdir (values.odir)
    ##########################################################
    fig = plt.figure(figsize=(15, 8))
    ###
    for f in tqdm.tqdm (values.h5, desc='cand', unit='c'):
        bf  = os.path.basename (f)
        with h5py.File (f, mode='r') as hf:
            try:
                plotter (hf,)
                fig.savefig (os.path.join (values.odir, bf.replace('.h5', '.png')), bbox_inches='tight')
            except:
                print (f"caught exception file={f}")
            finally:
                fig.clf()

