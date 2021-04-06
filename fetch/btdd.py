
"""

Do not use this module anymore.

Perfer using `btdd_m` module as that provides the capability to break the dm-axis.
"""

import numpy as np

import fetch.fdmt as fetch_fdmt
import fetch.incoherent as fetch_incoherent

DM_FAC = 4.148E-3

BT_INCOH = False
"""
FDMT is finnicky. 
If the max-delay is less than number-of-channels, FDMT is no longer the best. 
In that case, we would have to take a staggered approach

BT_INCOH is the switch which does not use FDMT to compute bowtie. Instead it does repeated de-dispersion.
It is slow. It is not exposed to the user and the supporting code will be removed soon.
"""

class BTDD:
    """
    Class to streamline computation of bowtie plane and de-dispersed filterbank.

    FDMT algorithm requires the number of DM-trials to be less than number-of-chans, 
    but since 
        (1) FETCH DM range is [0., 2 *DM] where DM is the dm of the burst,
        (2) and when dealing with data with high time resolution and low frequency resolution

    The FDMT implementation here is that of a staggarred approach.
    Bowtie (0..D) -------------------------> Bowtie (0..A)
        |
        \-----> Incoherent (A) ------------> Bowtie (0..A)
                    |
                    \---> Incoherent (A) --> Bowtie (0..A)

    where D is some large DM-delay-full-band in time units
    where A is the number of channels

    In some future, we could optimize like anything since 
    for a given obsparam (nchans,fch1,foff), bowtie(0..A) can be resolved in one pass

    This class doesnot make use of the optimization trick.
    Bowtie (A..B)  --> Incoherent (A) & Bowtie (B-A)
    Incoherent (C) --> Incoherent (A) & Incoherent (C-A)
    where A,B,C are delays


    Because FETCH DM range always start at DM=0
    """
    def __init__ (self,
            fch1=None,foff=None,nchans=None,
            tsamp=None, 
            ndm=None,
        ):
        """

        Arguments
        ---------
        -All freq in MHz
        """
        self.fch1    = fch1
        self.foff    = foff
        self.nchans  = nchans
        self.f_axis  = self.fch1 + (np.arange (nchans, dtype=np.float32) * self.foff)
        ##
        self.fullband_delay = DM_FAC * (  (self.f_axis[self.nchans-1]/1E3)**-2 - (self.f_axis[0]/1E3)**-2 )
        self.inband_delay = np.zeros (nchans, dtype=np.float32)
        for i,f in enumerate (self.f_axis):
            self.inband_delay[i] = DM_FAC * ( (f/1E3)**-2 - (self.f_axis[0]/1E3)**-2 )
        ##
        self.tsamp   = tsamp
        ##
        self.ndm     = ndm
        ##
        self.fdmt    = fetch_fdmt.FDMT (self.fch1, self.foff, self.nchans)
        self.bt_delays = None
        self.dd_delays = None
        self.max_d     = None
        if BT_INCOH:
            self.bb_delays = np.zeros ((self.ndm, self.nchans), dtype=np.uint64)
        else:
            self.bb_delays = None


    def __call__ (self, dm):
        """
        Sets DM and calculates the delay arrays
        """
        ## dm calculations
        dm_axis        = dm + np.linspace (-dm, dm, self.ndm)
        self.bt_delays      = np.zeros (self.ndm, dtype=np.uint32)

        if BT_INCOH:
            for i,idm in enumerate (dm_axis):
                self.bt_delays[i] = int (idm * self.fullband_delay / self.tsamp)
                self.bb_delays[i] = np.uint64 (self.inband_delay * idm / self.tsamp)
        else:
            for i,idm in enumerate (dm_axis):
                self.bt_delays[i] = int (idm * self.fullband_delay / self.tsamp)

        self.dd_delays      = np.uint64 (self.inband_delay * dm / self.tsamp)
        self.max_d          = self.bt_delays[-1]

    def work (self, fb):
        """
        fb is (nchans, nsamps)
        """
        nchans, nsamps = fb.shape
        if BT_INCOH:
            osamps         = int(nsamps - self.bb_delays.max())
            bt             = np.zeros ((self.ndm, osamps), dtype=fb.dtype)
            for i in range (self.ndm):
                bt[i,:osamps]      = fetch_incoherent.Dedisperser (fb, self.bb_delays[i]).mean (axis=0)[:osamps]
        else:
            bt             = self.fdmt.Bowtie_delays (fb, self.bt_delays)
        dd             = fetch_incoherent.Dedisperser (fb, self.dd_delays)

        ## reset delay arrays
        self.bt_delays = None
        self.dd_delays = None
        return bt,dd
