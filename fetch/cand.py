"""
candidate reading methods
courtesy of TRISHUL

Added hdf5 candidate output method
"""

import os

import h5py

import pandas as pd

HEIMDALL_COLS = ['sigma','index','time','boxcar','dm_index','dm','group_id','sample_start','sample_stop']
HEIMDALL_DT_FORMATS = {
        'datetime+beam': "%Y-%m-%d-%H:%M:%S_{beam}.cand",
        'datetime': "%Y-%m-%d-%H:%M:%S.cand",
}
PRESTO_SPS_COLS = ['dm', 'sigma', 'time', 'index', 'width_units']

def masker (df, dmlow=None, dmhigh=None, snlow=None, snhigh=None, wdlow=None, wdhigh=None):
    """Applies mask"""
    mask = df.time >= 0.0       # all true mask
    #
    if dmlow:
        mask = mask & (df.dm >= dmlow)
    if dmhigh:
        mask = mask & (df.dm <= dmhigh)
    if snlow:
        mask = mask & (df.sigma >= snlow)
    if snhigh:
        mask = mask & (df.sigma <= snhigh)
    if wdlow:
        mask = mask & (df.width_units >= wdlow)
    if wdhigh:
        mask = mask & (df.width_units <= wdhigh)
    #
    return mask

def read_heimdall_cands (filename,):
    """Reads candidates written by Heimdall

    Arguments
    ---------
    filename: str
        Filename of the candidate file
    parse: str
        Notes the filename format
        Can be either None or keys of HEIMDALL_DT_FORMATS
        #TODO
    """
    df = pd.read_csv (filename, delim_whitespace=True, names=HEIMDALL_COLS)
    df['width_units'] = 2 ** df['boxcar']
    return df

def read_presto_cands (filename,):
    """Reads candidates written by PRESTO:single_pulse_search

    Arguments
    ---------
    filename: str
        Filename of the candidate file
    """
    df = pd.read_csv (filename, delim_whitespace=True, comment='#', names=PRESTO_SPS_COLS)
    return df

def save_cand_h5(payload, out_dir=None, fnout=None, fil_header={}):
    """
    Generates an h5 file of the candidate object
    :param out_dir: Output directory to save the h5 file
    :param fnout: Output name of the candidate file
    :return:
    """
    if fnout is None:
        fnout = payload['cand_id'] + '.h5'

    if out_dir is not None:
        fnout = os.path.join (out_dir, fnout)

    with h5py.File(fnout, 'w') as f:
        for k in ['tcand','dm','snr','width','cand_id', 'label']:
            fp = payload[k]
            if fp is not None:
                f.attrs[k] = fp
            else:
                f.attrs[k] = b'None'

        # Copy over filterbank header information as attributes
        for k,v in fil_header.items():
            if v is not None:
                f.attrs[k]   = v
            else:
                f.attrs[k]   = b'None'

        ## what is called `dd`
        ## the transpose is necessary because of axis
        freq_time_dset = f.create_dataset('data_freq_time', data=payload['dd'].T, dtype=payload['dd'].dtype, compression="lzf")
        freq_time_dset.dims[0].label = b"time"
        freq_time_dset.dims[1].label = b"frequency"

        ## what is called `bt`
        dm_time_dset = f.create_dataset('data_dm_time', data=payload['bt'], dtype=payload['bt'].dtype, compression="lzf")
        dm_time_dset.dims[0].label = b"dm"
        dm_time_dset.dims[1].label = b"time"

    return fnout
