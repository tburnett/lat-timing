"""
"""
import numpy as np
import pandas as pd

class CellBinner(object):
    """Base class for creation of bins to define cells"""
    def __init__(self, rec):
        self.rec=rec
        
    def __call__(self, rec):
        """return a bin index"""
        raise NotImplementedError(f'Derived class {self.__class__} must implement')

class TimeBinner(CellBinner):
    
    def __init__(self, start, stop, step):
        self.nbins = nbins= int((stop-start+step/8)/step)
        self.bins = np.linspace(start, start+step*nbins, nbins+1)
    
    def __repr__(self):
        return f'{self.__class__}: {self.bins}'
    
    def __call__(self, times):
        """Returns bin index, where 0 means <=start, nbins+1 means >stop """

        return np.searchsorted(self.bins, times)
    
    @property 
    def start(self): return self.bins[0]
    
    @property
    def stop(self): return self.bins[-1]
    

class EnergyBinner(CellBinner):
    def __call__(self, band_ids):
        pass
        
        
class BinnedWeights(object):
    """ Generate a list of cells, with access to cell data
        weights
    """
    
    def __init__(self, data, bins=None):
        """
        Use time binning and data (a TimedData) object to generate list of cells
        """
        self.data = data 
        self.source_name = data.source_name
        self.verbose = data.verbose
        
        if bins is None:
            # get predefined bin data and corresponding fractional exposure
            # (this case using default step, range) 
            bins = data.edges
        elif np.isscalar(bins):
            # scalar is step
            step = bins
            tstart = data.time_bins[0]
            time_range = data.time_bins[-1]-tstart
            nbins = int(time_range/step)+1; 
            if data.verbose>0:
                print(f'Selecting {nbins} intervals of {step} days')
            bins = tstart + np.arange(nbins+1)*step 
        else:
            pass # check that bins makes sense:
        exposure = data.get_binned_exposure(bins)
        self.bins=bins
        self.N = len(bins)-1 # number of bins
        self.bin_centers = 0.5*(self.bins[1:]+self.bins[:-1])
        self.fexposure = exposure/np.sum(exposure)   

        # get the photon data with good weights, not NaN
        if 'weight' in data.photon_data:
            w = data.photon_data.weight
            good = np.logical_not(np.isnan(w))
            self.photons = data.photon_data.loc[good]
            self.weights = w = self.photons.weight.values
            # estimates for total signal and background
            self.S = np.sum(w)
            self.B = np.sum(1-w)
        else:
            # no weights specified
            self.photons = data.photon_data
            self.S = self.B = 0
            self.weights=[]
            
        # use photon times to get indices of bin edges
        self.edges = np.searchsorted(self.photons.time, self.bins)
        
        
    def __repr__(self):
        return f'''{self.__class__}:  
        {len(self.fexposure)} intervals from {self.bins[0]:.1f} to {self.bins[-1]:.1f} for source {self.source_name}
        S {self.S:.2f}  B {self.B:.2f} '''

    def __getitem__(self, i):
        """ get info for ith time bin and return dict with 
            t : MJD
            tw: bin width, 
            fexp: exposure as fraction of total, 
            n : number of photons in bin
            w : weights 
            S,B:  value
        """
        k   = self.edges        
        if len(self.weights)==0: 
            # No weights
            wts = []
            n = len(self.photons[k[i]:k[i+1]])
        else:
            wts = self.weights[k[i]:k[i+1]]
            n = len(wts)
        exp = self.fexposure[i]
        tw  = self.bins[i+1]-self.bins[i]

        return dict(
                t=self.bin_centers[i], # time
                tw = tw,  # bin width
                fexp=exp*self.N, # exposure as a fraction of mean, for filtering
                n=n, # number of photons in bin
                w=wts,
                S= exp*self.S,
                B= exp*self.B,               
                )

    def __len__(self):
        return self.N

    def test_plots(self):
        """Make a set of plots of exposure, counts, properties of weights, if any
        """
        import matplotlib.pyplot as plt

        has_weights = len(self.weights)>0
        fig, axx = plt.subplots( 5 if has_weights else 3, 1, 
                    figsize=(12,10 if has_weights else 6), 
                    sharex=True,
                    gridspec_kw=dict(hspace=0,top=0.95),)
        times=[]; vals = []

        for cell in self:
            t, e, n, w = [cell[q] for q in 't fexp n w'.split()]
            if e==0:
                continue
            times.append(t)
            v =  [e, n, n/e ]
            if has_weights:
                v= v + [ w.mean(), np.sum(w**2)/sum(w)]
            vals.append(v)
        vals = np.array(vals).T
        labels =  ['rel exp','counts','count rate']
        if has_weights: labels = labels +  ['mean weight', 'rms/mean weight']
        for ax, v, ylabel in zip(axx, vals,labels):
            ax.plot(times, v, '+b')
            ax.set(ylabel=ylabel)
            ax.grid(alpha=0.5)
        axx[-1].set(xlabel='MJD')
        fig.suptitle(self.source_name)