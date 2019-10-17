"""process weights for timed data
"""

import os, sys
import numpy as np
# from scipy import optimize
# from numpy import linalg
import pandas as pd

import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord


#local
from data_management import Data
import keyword_options


class BinnedWeights(object):
    """ manage access to weights"""
    
    def __init__(self, data):
        
        # get predefined bin data and corresponding fractional exposure 
        bins, exposure = data.binner()
        self.bins=bins
        self.N = len(bins)-1 # number of bins
        self.bin_centers = 0.5*(self.bins[1:]+self.bins[:-1])
        self.fexposure = exposure/np.sum(exposure)
        self.source_name = data.source_name

        # get the photon data with good weights, not NaN
        w = data.photon_data.weight
        good = np.logical_not(np.isnan(w))
        self.photons = data.photon_data.loc[good]

        # use photon times to get indices of bin edges
        self.weights = w = self.photons.weight
        self.edges = np.searchsorted(self.photons.time, self.bins)
        
        # estimates for total signal and background
        self.S = np.sum(w)
        self.B = np.sum(1-w)

        
    def __repr__(self):
        return f'''{self.__class__}:  
        {len(self.fexposure)} intervals from {self.bins[0]:.1f} to {self.bins[-1]:.1f} for source {self.source_name}
        S {self.S:.2f}  B {self.B:.2f} '''

    def __getitem__(self, i):
        """ get info for ith time bin and return dict with time, exposure, weights and S,B value
        """
        k = self.edges        
        wts = self.weights[k[i]:k[i+1]]
        exp=self.fexposure[i]

        return dict(
                t=self.bin_centers[i], # time
                exp=exp*self.N,        # exposure as a fraction of mean, for filtering
                w=wts,
                S= exp*self.S,
                B= exp*self.B,               
                )

    def __len__(self):
        return self.N

    def test_plots(self):
        """  plots of properties of the weight distribution"""
        fig, axx = plt.subplots(4,1, figsize=(12,6), sharex=True,
                                         gridspec_kw=dict(hspace=0))
        times=[]; vals = []
        for cell in self:
            t, e, w = [cell[q] for q in 't exp w'.split()]
            times.append(t)
            vals.append( (e, len(w), len(w)/e , w.mean(), np.sum(w**2)/sum(w)))
        vals = np.array(vals).T
        print( np.array(vals).shape)
        for ax, v, ylabel in zip(axx, vals, ['rel exp','counts','rate', 'mean weight', 'rms/mean weight']):
            ax.plot(times, v, 'ob')
            ax.set(ylabel=ylabel)
            ax.grid(alpha=0.5)
        fig.suptitle(self.source_name)


class Main(object):

    defaults=(
        ('verbose', 1,),
        ('source_name', 'Geminga','name of the source'),
        ('mjd_range', (51910, 54800),'MJD limits to use'),
        ('interval', 5, 'days per time bin'),
        ('weight_file', '../data/geminga_weights.pkl', 'location of pointlike weight file'),
        ('min_exposure', 0.2, 'ignore bins with exposure less than this')
    )
    
    @keyword_options.decorate(defaults)
    def __init__(self,  **kwargs):
        keyword_options.process(self, kwargs)

        sc = SkyCoord.from_name(self.source_name).galactic
        self.__dict__.update(l=sc.l.value, b=sc.b.value) 

        # load the data and produce the photon data dataframe
        self.data = data= Data(self)

        # add the weights to the photon dataframe
        wtd = data.add_weights(self.weight_file)
        # get info from weights file
        vals = [wtd[k] for k in 'model_name roi_name source_name source_lb '.split()]
        lbformat = lambda lb: '{:.2f}, {:.2f}'.format(*lb)
        self.model_info='\n  {}\n  {}\n  {}\n  {}'.format(vals[0], vals[1], vals[2], lbformat(vals[3]))

        # add a energy band column, filter out photons with NaN         
        gd = data.photon_data
        gd.loc[:,'eband']=(gd.band.values//2).clip(0,7)

        ok = np.logical_not(pd.isna(gd.weight))
        self.photons = gd.loc[ok,:]
        
    def all_weights(self):
        w = self.data.photon_data.weight
        good = np.logical_not(np.isnan(w))
        return w[good]

    def binned_weights(self):
        """ return a BinnedWeight object for access to each set of binned weights
        The object can be indexed, or used in a for loop
        bw[i] returns a  dict (t, exp, w, S, B)
        where t   : bin center time (MJD)
              exp : associated exposure as fraction of mean
              w   : array of weights for the time range
              S,B : predicted source, background counts for this bin
        """
        return BinnedWeights(self.data,)

        
    def corner_plot(self, weighted=False, **kwargs):
        """produce a "corner" plot.
        """
        import corner
        fig, axx=plt.subplots(3,3, figsize=(10,10))
        corner.corner(self.photons['eband radius weight'.split()], 
                    range=[(-0.75,7.25),(0,5),(0,1)],
                    bins=(16,20,20),
                    label_kwargs=dict(fontsize=14), 
                    labels=['Energy band index', 'Radius [deg]', 'weight'],
                    weights=self.photons.weight if weighted else None,
                    show_titles=False, top_ticks=True, color='blue', fig=fig, 
                    hist_kwargs=dict(histtype='stepfilled', facecolor='lightgrey',edgecolor='blue', lw=2),
                      **kwargs);
        fig.set_facecolor('white')
        use_weights='weighted' if weighted else 'not weighted'
        fig.suptitle(f'{self.source_name} analysis\n  {len(self.photons)} events\n  {use_weights} \nModel info'+self.model_info,
                    fontsize=14, ha='left',
                    fontdict=dict(family='monospace'), #didn't work
                    )
        
        