"""process weights for timed data
"""

import os, sys
import numpy as np
from scipy import optimize
from numpy import linalg
import pandas as pd

import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord

import corner

#local
from data_management import Data
import keyword_options


class BinnedWeights(object):
    """ manage access to weights"""
    
    def __init__(self, data):
        # get predefined bin data and corresponding fractional exposure 
        time_bins, exposure = data.binner()
        self.bin_centers = 0.5*(time_bins[1:]+time_bins[:-1])
        self.fexposure = exposure/np.sum(exposure)
        self.source_name = data.source_name

        # get the photon data with good weights
        w = data.photon_data.weight
        good = np.logical_not(np.isnan(w))
        self.photons = data.photon_data.loc[good]

        # use photon times to get indices of bin edges
        self.weights = self.photons.weight
        self.edges = np.searchsorted(self.photons.time, time_bins)
        
    def __getitem__(self, i):
        k = self.edges        
        wts = self.weights[k[i]:] if i>len(k)-3 else self.weights[k[i]:k[i+2]] 
        return self.bin_centers[i], self.fexposure[i], wts.values
    
    def __len__(self):
        return len(self.bin_centers)

    def test_plots(self):
        """ some test plots of properties of the weight distribution"""
        fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(10,8), sharex=True)
        for t, e, w in self:
            ax1.plot(t, (len(w)/e)*1e6, 'ob')
            ax2.plot(t, w.mean(), 'ob')
            ax3.plot(t, np.sum(w**2)/np.sum(w), 'ob')
        ax1.set(ylabel='rate [arg units]')
        ax2.set(ylabel='mean weight', ylim=(0,1))
        ax3.set(ylabel='rms/mean weight',ylim=(0,1))
        fig.suptitle(self.source_name)


class Main(object):

    defaults=(
        ('verbose', 1,),
        ('source_name', 'Geminga','name of the source'),
        ('mjd_range', (51910, 54800),'MJD limits to use'),
        ('interval', 5, 'days per time bin'),
        ('weight_file', '../data/geminga_weights.pkl', 'location of pointlike weight file'),
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
        The object has a index
        so bw[i] returns a tuple (t, e, w)
        where t : bin center time (MJD)
              e : associated exposure
              w : array of weights for the time range
        """
        return BinnedWeights(self.data)


    def weight_binner(self, step=None, start=None, stop=None,):
        pass
    
    def solve(self,  exposure_fraction, estimate=[0,0], fix_beta=False, **fmin_kw):
        """SOlve for alpha and beta for all of current selected data set
           TODO: extend to set of data segments
        
        parameters:
           exposure_fraction : Contribution to total exposure
           estimate : initial values for (alpha,beta)
           fix_beta : if True, only fit for alpha
           
        returns:
             fits for alpha,beta
        """
        return LogLike(self.weights(), exposure_fraction).solve(estimate, fix_beta, **fmin_kw)

        
    def corner_plot(self, weighted=False, **kwargs):
        """produce a "corner" plot.
        """
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
        
        