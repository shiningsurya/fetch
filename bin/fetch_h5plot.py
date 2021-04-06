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
    axpp.step (ts, dd.mean (1), where='mid', c='k', linestyle='-')
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

