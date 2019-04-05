# ====================================================================================== #
# Module for loading Timme (2016) data set downloaded from CRCNS. Represents spontaneous
# activity in rat hippocampal disassociated cultures.
# Author: Eddie Lee, edlee@alumni.princeton.edu
# ====================================================================================== #
import pandas as pd
import numpy as np
import os
from scipy.io import loadmat
from scipy.interpolate import interp1d
from numba import njit
DATADR = os.path.expanduser('~')+'/Dropbox/Research/py_lib/data_sets/neuron_timme'


@njit
def count_zeros_at_end(x):
    i = 0
    while i<len(x) and x[-1-i]==0:
        i += 1
    return i

def split_at_zeros(x, n_zeros):
    """Split x at any sequence of at least n_zeros. And return subsets pruned such that no
    returned sequence is sandwiched by zeros."""

    assert n_zeros>=1
    assert x.ndim==1
    y = []
    countingZeros = True
    done = False
    i = 0
    startix = 0
    
    # don't consider any initial zeros
    while  i<x.size and x[i]==0:
        i += 1
        startix += 1

    while not done:
        if countingZeros:
            zerocounter = 0
            while i<x.size and x[i]==0:
                zerocounter += 1
                i += 1
            if zerocounter>=n_zeros:
                split = True 
            else:
                split = False
            countingZeros = False
        else:
            if split:
                y.append(x[startix:i])
                y[-1] = y[-1][:-count_zeros_at_end(y[-1])]
                startix = i
                split = False
                
            while i<x.size and x[i]:
                i += 1
            countingZeros = True

        if i==x.size:
            y.append(x[startix:i])
            done = True

    # don't consider any ending zeros
    if y[-1][-1]==0:
        i = y[-1].size-1
        while  i>0 and y[-1][i]==0:
            i -= 1
        y[-1] = y[-1][:i+1]

    return y


class NeuronData():
    def __init__(self, culture, div):
        """
        Load an instance of data.

        Parameters
        ----------
        culture : int
        div : int
        """
        
        data = loadmat('%s/Culture%dDIV%d.mat'%(DATADR,culture,div))
        self.spikes = [i[0].ravel() for i in data['data'][0][0][0]]
        self.n_neurons = data['data'][0][0][1].ravel()[0] 
        self.binsize = data['data'][0][0][2].ravel()[0]
        self.nbins = data['data'][0][0][3].ravel()[0]
        self.recording_id = data['data'][0][0][4][0]
        self.datatype = data['data'][0][0][5][0]
        self.exp_sys = data['data'][0][0][6][0]
        self.channel = data['data'][0][0][7].ravel()
        self.params = data['data'][0][0][8]
        self.interspikeInterval = None

    def calculate_interspike(self):
        """Network-wide interspike interval considering all spikes by all neurons in a
        single time series. This value is typically used for deciding how to discretize
        the time series.
        """

        if self.interspikeInterval is None:
            sortedSpikeTimes = np.concatenate(self.spikes).tolist()
            sortedSpikeTimes.sort()
            self.interspikeInterval = np.diff(sortedSpikeTimes).mean()*self.binsize
        return self.interspikeInterval

    def cat_avalanches(self, n_zeros, dt=None, min_len=10):
        """
        Cluster neuron spikes into avalanches given time bin discretization and using time
        contiguous cascades.

        Parameters
        ----------
        n_zeros : int
        dt : float, None
            If not specified, the average interspike interval from the data is used.
        min_len : int, 10
            Min number of contiguous bins for an avalanche to fire across to be included.

        Returns
        -------
        list of ndarrays
            Each ndarray specified how many different neurons were firing at each point in
            time.
        """
        
        if dt is None:
            dt = self.calculate_interspike()
        assert dt>=self.binsize, "Specified dt cannot be smaller than recording discreteness."

        # round each spike timing to the nearest bin
        spikes_ = self.spikes[:]
        factor = dt/self.binsize
        for i in range(self.n_neurons):
            spikes_[i] = np.around(spikes_[i]/factor).astype(int)
        assert all([s[-1]<1e7 for s in spikes_]), "Too large of an array formed."
        
        binnedCounts = np.zeros(1+max([s[-1] for s in spikes_]), dtype=int)
        for i in range(self.n_neurons):
            binnedCounts[spikes_[i]] += 1
        
        # identify contiguous sections defined by where there are no more than n bins that are empty
        avalanches = split_at_zeros(binnedCounts, n_zeros)
        avalanches = [a for a in avalanches if len(a)>min_len]
        # remove 0 if it's the first entry
        avalanches = [a[1:] if a[0]==0 else a for a in avalanches]

        # cache avalanches
        self._avalanches = avalanches

        return avalanches

    def binary_time_series(self, dt=None):
        """Cluster neuron spikes into avalanches given time bin discretization and using
        time contiguous cascades.

        Parameters
        ----------
        dt : float, None
            If not specified, the average interspike interval from the data is used.

        Returns
        -------
        ndarray of type uint8
            Each row is a different neuron.
        """
        
        if dt is None:
            dt = self.calculate_interspike()
        assert dt>=self.binsize, "Specified dt cannot be smaller than recording discreteness."

        # round each spike timing to the nearest bin
        spikes_ = self.spikes[:]
        factor = dt/self.binsize
        for i in range(self.n_neurons):
            spikes_[i] = np.around(spikes_[i]/factor).astype(int)
        
        T = max([i[-1] for i in spikes_])
        assert T*self.n_neurons<1e8, "Binary array will be too large."
        X = np.zeros((self.n_neurons, T+1), dtype=np.uint8)
        for i in range(self.n_neurons):
            X[i,spikes_[i]] = 1
        
        return X

    def avalanches(self, dt=None, min_len=10):
        """
        Cluster neuron spikes into avalanches given time bin discretization and using time
        contiguous cascades.

        Parameters
        ----------
        dt : float, None
            If not specified, the average interspike interval from the data is used.
        min_len : int, 10
            Min number of contiguous bins for an avalanche to fire across to be included.

        Returns
        -------
        list of ndarrays
            Each ndarray specifies how many different neurons were firing at each point in
            time.
        """
        
        if dt is None:
            dt = self.calculate_interspike()
        assert dt>=self.binsize, "Specified dt cannot be smaller than recording discreteness."

        # round each spike timing to the nearest bin
        spikes_ = self.spikes[:]
        factor = dt/self.binsize
        for i in range(self.n_neurons):
            spikes_[i] = np.around(spikes_[i]/factor).astype(int)
        assert all([s[-1]<1e7 for s in spikes_]), "Too large of an array formed."
        
        binnedCounts = np.zeros(1+max([s[-1] for s in spikes_]), dtype=int)
        for i in range(self.n_neurons):
            binnedCounts[spikes_[i]] += 1
        
        # identify contiguous sections where at least one neuron is firing in every time bin
        avalanches = np.split(binnedCounts, np.where(binnedCounts==0)[0])
        avalanches = [a for a in avalanches if len(a)>min_len]
        # remove 0 if it's the first entry
        avalanches = [a[1:] if a[0]==0 else a for a in avalanches]

        # cache avalanches
        self._avalanches = avalanches

        return avalanches

    def rate_profile(self, dt=None, n_interpolate=101, return_error=True):
        """Return average of linearly interpolated cumulative profile.
        
        Parameters
        ----------
        n_interpolate : int, 101
        insert_zero : bool, False
            If True, insert 0 at beginning of trajectory to pin start at 0.
        return_error : bool, False

        Returns
        -------
        ndarray
            Cum profile.
        ndarray
            Discretized time.
        int
            Number of avalanches averaged.
        ndarray (optional)
            If return_error is True.
        """

        if dt is None and not '_avalanches' in self.__dict__.keys():
            self.avalanches()
        elif dt:
            self.avalanches(dt=dt)
        av = self._avalanches

        # interpolate each avalanche
        t = np.linspace(0, 1, n_interpolate)

        traj = np.zeros((len(av),t.size))
        for i,a in enumerate(av):
            traj[i] = interp1d(np.linspace(0,1,a.size), a/a.sum()*a.size)(t)

        avgTraj = traj.mean(0)
        if return_error:
            return avgTraj, t, len(av), traj.std(ddof=1,axis=0)
        return avgTraj, t, len(av)

    def cum_profile(self, dt=None,
                    n_interpolate=101,
                    correct_bias=True,
                    min_duration=10,
                    return_error=False):
        """Return average of linearly interpolated cumulative profile.
        
        Parameters
        ----------
        n_interpolate : int, 101
        min_duration : int, 4
        return_error : bool, False

        Returns
        -------
        ndarray
            Cum profile.
        ndarray
            Discretized time.
        int
            Number of avalanches averaged.
        ndarray (optional)
            If return_error is True.
        """

        if dt is None and not '_avalanches' in self.__dict__.keys():
            self.avalanches()
        elif dt:
            self.avalanches(dt=dt)
        av = self._avalanches

        # interpolate each avalanche
        t = np.linspace(0, 1, n_interpolate)

        traj = np.zeros((len(av),t.size))
        # if no zero inserted, must account for lattice bias of 1/S at t=0 explicitly
        for i,a in enumerate(av):
            if a.size>=min_duration:
                x = np.linspace(0,1,a.size)
                x = np.insert(x, range(x.size), x)[1:]
                y = np.cumsum(a)/a.sum()
                y = np.insert(y, range(y.size), y)[:-1]

                y -= 1/a.size
                y[-1] -= 1/a.size
                y /= y[-1]

                traj[i] = interp1d(x,y)(t)

        avgTraj = traj.mean(0)
        if return_error:
            return avgTraj, t, len(av), traj.std(ddof=1,axis=0)
        return avgTraj, t, len(av)
