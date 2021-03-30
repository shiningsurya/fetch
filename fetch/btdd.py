
"""

Should I cython?

FETCH seems to this:
    - For a candidate at DM, FETCH.bowtie plane goes from -DM to DM with ndm steps
"""

import numpy as np

import fetch.fdmt as fetch_fdmt
import fetch.incoherent as fetch_incoherent

DM_FAC = 4.148E-3

class BTDD:
    """
    Class to streamline computation of bowtie plane and de-dispersed filterbank.

    This class doesnot make use of the optimization trick.
    Bowtie (A..B)  --> Incoherent (A) & Bowtie (B-A)
    Incoherent (C) --> Incoherent (A) & Incoherent (C-A)
    where A,B,C are delays

    Because FETCH DM range always start at DM=0
    """
    def __init__ (self,
            fch1=None,foff=None,nchans=None,
            tsamp=None, nsamps=None,
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


    def __call__ (self, dm):
        """
        Sets DM and calculates the delay arrays
        """
        ## dm calculations
        dm_axis        = dm + np.linspace (-dm, dm, self.ndm)
        self.bt_delays      = np.zeros (self.ndm, dtype=np.uint32)
        for i,idm in enumerate (dm_axis):
            self.bt_delays[i] = int (idm * self.fullband_delay / self.tsamp)
        self.dd_delays      = np.uint64 (self.inband_delay * dm / self.tsamp)

    def work (self, fb):
        """
        fb is (nchans, nsamps)
        """
        ## actual algo
        bt             = self.fdmt.Bowtie_delays (fb, self.bt_delays)
        dd             = fetch_incoherent.Dedisperser (fb, self.dd_delays)

        ## reset delay arrays
        self.bt_delays = None
        self.dd_delays = None
        return bt,dd
