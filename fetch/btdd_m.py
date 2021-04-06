
"""

Should I cython?

FETCH seems to this:
    - For a candidate at DM, FETCH.bowtie plane goes from 0 to 2*DM with ndm steps
    - the time resolution in the bowtie and the de-dispersed filterbank is half-the-width
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

`_m` is where I modularize the code
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
            dmbreak=None,
            verbose=False,
        ):
        """

        Arguments
        ---------
        -All freq in MHz
        """
        self.v       = verbose
        ##
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
        self.ndm     = ndm
        if dmbreak:
            self.dmbreak = dmbreak
        else:
            self.dmbreak = int (0.80 * self.nchans)
        ##
        self.fdmt    = fetch_fdmt.FDMT (self.fch1, self.foff, self.nchans)
        ##
        self.base_bt_delays = np.arange (self.dmbreak, dtype=np.uint32)
        self.base_incoh_dm  = self.nchans * self.tsamp / self.fullband_delay
        self.base_dd_delays = np.uint64 (self.inband_delay * self.dmbreak/ self.fullband_delay)
        if self.v:
            print (f"FDMT dmbreak={self.dmbreak} dmbreak_time={self.dmbreak*self.tsamp*1E3:3.2f} ms incoh.dm={self.base_incoh_dm} pc/cc")
        self.dm_axis        = np.zeros (self.ndm, dtype=np.float32)
        self.bt_delays      = np.zeros (self.ndm, dtype=np.uint32)
        self.dd_delays      = np.zeros (self.nchans, dtype=np.uint64)
        self.max_d     = None


    def __call__ (self, dm):
        """
        Sets DM and calculates the delay arrays

        BTDD is designed to for with FETCH DM sampling
        """
        ## reset
        self.__reset__ ()
        ## bt calculations
        self.dm_axis[:]     = dm + np.linspace (-dm, dm, self.ndm)
        for i,idm in enumerate (self.dm_axis):
            self.bt_delays[i] = int (idm * self.fullband_delay / self.tsamp)
        ## dd calculations
        self.dd_delays      = np.uint64 (self.inband_delay * dm / self.tsamp)
        ## misc calculations
        self.max_d     = self.bt_delays[-1]
        self.steps     = ( self.max_d // self.dmbreak ) + 1
        self.max_d     = self.steps * self.nchans
        ## we would want integral number of calls
        if self.v:
            print (f" max dm={2*dm} delay={self.max_d}")

    def __reset__ (self):
        self.max_d = 0
        self.dm_axis[:]   = 0.
        self.bt_delays[:] = 0
        self.dd_delays[:] = 0

    def work (self, fb):
        """
        fb is (nchans, nsamps)

        Divide `max_d` into steps of `nchans`

        The caller is in charge of the cropping. All calls to `Dedisperser` and `Bowtie` are considered valid.

        Performing `dd` computation outside FDMT, but can put it inside with some more logic
        """
        nchans, nsamps = fb.shape
        osamps         = int(nsamps - self.max_d)
        ret_bt         = np.zeros ((self.ndm, osamps), dtype=fb.dtype)
        ###

        ## dd computation
        ret_dd         = fetch_incoherent.Dedisperser (fb, self.dd_delays)

        ## bt computation

        # reference to the fb
        bb    = fb
        for istep in range (self.steps):
            # compute delays in this iteration
            I = istep * self.dmbreak
            J = I + self.dmbreak

            # compute bt
            bt    = self.fdmt.Bowtie_delays (bb, self.base_bt_delays)
            # take valid slices
            for idm,idelay in enumerate (self.bt_delays):
                if idelay >= I and idelay <= J:
                    ret_bt[idm] = bt[idelay % self.dmbreak,:osamps]

            # compute dd
            bb   = fetch_incoherent.Dedisperser (bb, self.base_dd_delays)

        return ret_bt, ret_dd
