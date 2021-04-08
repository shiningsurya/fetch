#!/usr/bin/env python3
"""

SB:
- `dm_time` `freq_time` are now computed using cython code
- To reduce the large IO/memory requirements, decimation-in-time is done prior as filterbank is read. Meaning, decimation happens before de-dispersion
- - Since decimation-in-time happens before de-disperion/bowtie plane, the "tsamp" any of the those codes sees is different for every candidate. This hurts us because FDMT/Incoherent code as of now follows a class approach. 
- - Also, we should NOT do any frequency crunching prior de-dispersion because our DM sensitivity affects
- DM, SNR optimizations are left out since interest is speed
- Instead of job-based parallelism, it is left to the user to implement parallelism by running multiple instances
- Is hdf5 really needed? Why not use json or hex-encoded json like in PEASOUP?

"""

import gc
import argparse

import tqdm

import numpy as np
import pandas as pd

# for filterbank support
import sigpyproc as spp

## fetch imports
from fetch import cand as fetch_cands
# from fetch import btdd as fetch_btdd
from fetch import btdd_m as fetch_btdd

## kind of sucks that `scikit-image` is a dependency only because of this `block_reduce`
from skimage.measure import block_reduce

import matplotlib.pyplot as plt

print ()

#############
ODIR  = "./"
MASK  = None
TSIZE = 256
DSIZE = 256
FSIZE = 256
#############

def center_crop (data, axis=0, take=256):
    """
    Crops from the center along `axis`
    """
    ds    = data.shape
    start = ds[axis]//2 - take//2
    sl    = []
    for idim, dim in enumerate(ds):
        if idim == axis:
            sl.append (slice(start, start+take))
        else:
            sl.append (slice(0, dim))
    return data[tuple(sl)]

def normalize(data, usemedian=True):
    """
    Noramlise the data by unit standard deviation and zero median
    :param data: data
    :return:
    """
    data = np.array(data, dtype=np.float32)
    if usemedian:
        data -= np.median (data)
    else:
        data -= np.mean (data)
    data /= np.std(data)
    return data

def worker (packed):
    """
    Generates h5 file of candidate with resized frequency-time and DM-time arrays
    :param cand_val: List of candidate parameters (fil_name, snr, width, dm, label, tcand(s))
    :type cand_val: Candidate
    :return: None

    TODO: Add option to use cand.resize for reshaping FT and DMT

    QQQ: is h5 really needed?

    """
    #####
    fil_name, cands = packed
    
    # read filterbank 
    fil     = spp.FilReader (fil_name)
    fh      = fil.header

    tstart  = fh['tstart']
    tsamp   = fh['tsamp']
    fch1,foff,nchans  = [fh[x] for x in ['fch1', 'foff', 'nchans']]

    tcands  = tqdm.tqdm (cands.index, desc='Candidates', unit='cand')
    for idx in tcands:
        gc.collect ()
        ret    = dict()

        SIGMA  = cands.sigma[idx]
        DM     = cands.dm[idx]
        TIME   = cands.time[idx]
        WIDTH  = cands.width_units[idx]

        wfac   = max (1, WIDTH // 2)

        ret['tcand'] = TIME
        ret['dm']    = DM
        ret['snr']   = SIGMA
        ret['width'] = WIDTH
        ret['cand_id']    = f'cand_tstart_{tstart:.12f}_tcand_{TIME:.7f}_dm_{DM:.5f}_snr_{SIGMA:.5f}'
        ret['label'] = None

        btdd = fetch_btdd.BTDD (
                fch1=fch1, foff=foff, nchans=nchans,
                tsamp=tsamp * wfac, 
                ndm=DSIZE
        )

        btdd (DM)
        """
        NB: `FETCH:candmaker.py` which makes use of `pysigproc` to do the candidate computations
            dedisperses to the top-of-the-band and performs a center crop. 

        We take care here to match whatever we get here with whatever that gives.
        That having being said, we take a different approach, 
        we slice asymmetrically. 

        We need decimated 128 samples before the trigger => `128*wfac` samples before the trigger
        we need decimated 128 samples after the trigger and after de-dispersion/bowtie
        ==> thankfully, since we are taking care with start_slice, we just blindly take 256 samples and it will match up.
        """
        pre_take     = wfac * TSIZE // 2
        post_take    = wfac * ( btdd.max_d + WIDTH + TSIZE//2 )

        start_sample = int ((TIME / tsamp) - pre_take)
        width_sample = int (pre_take + post_take)

        ## check if start_sample is before the start of obs
        if start_sample < 0:
            width_sample = width_sample - start_sample
            start_sample = 0

        # print (f"takes pre={pre_take} post={post_take} width_sample={width_sample} delays={btdd.bt_delays[-1]} wfac={wfac}")
        # print (f" half_take={half_take} disp_sample={disp_sample} chunk_sample={chunk_sample} delays={btdd.bt_delays[-1]}")
        # print (f"delays={btdd.bt_delays[-1]} WIDTH={WIDTH} wfac={wfac} width_sample={width_sample}")

        fb           = decimated_read (fil, start_sample, width_sample, wfac, ffac=1, mask=MASK)
        # fb           = fil.readBlock (start_sample, width_sample)

        bt,dd      = btdd.work (fb)

        ## normalize
        bt = normalize (bt)
        dd = normalize (dd)

        # fig = plt.figure ()
        # plt.imshow (bt, aspect='auto', cmap='jet', origin='lower')
        # plt.show ()
        # adsfasdf

        ## crop
        ## XXX We no longer use center_crop because we slice carefully
        #  since our slicing is symmetric, our cropping is also symmetric
        # ret['bt'] = center_crop (bt, axis=1, take=TSIZE)
        # ret['dd'] = center_crop (dd, axis=1, take=TSIZE)
        ret['bt'] = bt[:,:TSIZE]
        ## XXX we have only done time crunching until now. 
        #  we do frequency crunching+cropping at the same time here
        ret['dd'] = block_reduce (dd[:,:TSIZE], (int(nchans//FSIZE), 1), func=np.mean, cval=np.median(dd))

        if ret['bt'].shape != (DSIZE, TSIZE) or ret['dd'].shape != (FSIZE, TSIZE):
            print (cands.loc[idx])
            print (f"payload.shapes bt={ret['bt'].shape} dd={ret['dd'].shape} width={WIDTH} wfac={wfac} fb={fb.shape}")
            raise ValueError (" Size is not matching")

        ## save
        fetch_cands.save_cand_h5 (ret, out_dir=ODIR, fil_header=fh)
        ## call the `h5plotter` if you want to plot

    return None

def decimated_read (fil, start, width, tfac, ffac=1, gulp=1024, mask=None):
    """
    Reads filterbank and decimates in chunks to reduce IO/memory burden

    Arguments
    ---------
        fil: SigPyProc FilReader instance
        start: unsigned int start of the samples
        width: unsigned int number of samples to read

        tfac: decimation in time-axis (axis=1)
        ffac: decimation in freq-axis (axis=0)

        gulp: int number of samples to read in a gulp
        mask: array of nchans 

    Returns
    -------
        filterbank: numpy.ndarray of float32
    """
    nchans_in    = fil.header['nchans']
    nchans_ou    = nchans_in // ffac 
    nsamps_ou    = ( width // tfac ) + 1
    # print (f" width={width} tfac={tfac} ou={nsamps_ou}")

    ofb          = np.zeros ((nchans_ou, nsamps_ou), dtype=np.float32)

    A,B          = 0,0
    rp           = fil.readPlan (gulp, start=start, nsamps=width, verbose=False,)
    for iread, _, payload in rp:
        gfb      = payload.reshape ((-1, nchans_in)).T
        if MASK is not None:
            gfb[MASK] = 0

        hfb      = block_reduce (gfb, (ffac, tfac), func=np.mean, cval=np.median(gfb))
        B        = A + hfb.shape[1]
        # print (f" decimated_read shapes in={gfb.shape} io={hfb.shape} A={A} B={B} ofb={ofb.shape} tfac={tfac} ffac={ffac}")
        ofb[:,A:B] = hfb
        A        = B

    return ofb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='Be verbose', action='store_true')

    parser.add_argument('-fs', '--frequency_size', type=int, help='Frequency size after rebinning', default=FSIZE)
    parser.add_argument('-ts', '--time_size', type=int, help='Time length after rebinning', default=TSIZE)
    parser.add_argument('-ds', '--dm_size', type=int, help='Number of DMs', default=DSIZE)

    parser.add_argument('-c', '--cand_file', help='candidate file. It can be either HEIMDALL or PRESTO candidates.', type=str, required=True)
    parser.add_argument('-f', '--fil_file', help='filterbank file. ', type=str, required=True)
    parser.add_argument('-k', '--kill_file', help='Numpy readable kill file. ', type=str, default=None)

    parser.add_argument('--snlow', help='Minimum S/N', type=float, default=5.0)
    parser.add_argument('--snhigh', help='Maximum S/N', type=float, default=300.0)

    parser.add_argument('--dmlow', help='Minimum DM', type=float, default=0.0)
    parser.add_argument('--dmhigh', help='Maximum DM', type=float, default=2000.0)

    parser.add_argument('--wdlow', help='Minimum boxcar width in log2 (int)', type=int, default=1)
    parser.add_argument('--wdhigh', help='Maximum boxcar width in log2 (int)', type=int, default=16)


    parser.add_argument('-o', '--fout', help='Output file directory for candidate h5', type=str)
    values = parser.parse_args()

    ####################################
    ### args logic
    TSIZE = values.time_size
    FSIZE = values.frequency_size
    DSIZE = values.dm_size
    ODIR  = values.fout

    ####################################
    ### read cand logic

    if values.cand_file.endswith ('cand'):
        cands = fetch_cands.read_heimdall_cands (values.cand_file)
    elif values.cand_file.endswith ('singlepulse'):
        cands = fetch_cands.read_presto_cands (values.cand_file)
    else:
        raise ValueError ('Candidate file format not understood.')

    ### cands selection logic
    cmask = fetch_cands.masker (cands, values.dmlow, values.dmhigh, values.snlow, values.snhigh, values.wdlow, values.wdhigh)
    mands = pd.DataFrame (cands[cmask])
    
    ####################################
    ### kill mask logic
    if values.kill_file is not None:
        MASK = np.loadtxt (values.kill_file, dtype=np.uint32)

    ####################################
    ### run
    worker ([values.fil_file, mands])
