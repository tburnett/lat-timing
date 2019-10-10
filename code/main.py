"""
Process time data set
Expect to find set of files created by uw/data/timed_data/create_timed_data to generate files with times for all 
Extract a single data set around a cone
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import keyword_options
import exposure 
from data_management import Data, MJD


class Main(object):
    """Top-level processing for photon data
    """
 
    plt.rc('font', size=12)
    defaults=(
        ('verbose', 1, 'verbosity level'),
        ('radius',  5, 'cone radius for selection [deg]'),
        ('interval', 2, 'Binning time interval [days]'),
        ('mjd_range', None, 'Range of MJD: default all data'),

       )
    
    @keyword_options.decorate(defaults)
    def __init__(self, name, position=None, **kwargs  ):
        """Set up combined data from set of monthly files

        name :    string, source name 
        position : an (l,b) pair [optional- if not present, use name to look up] 

        """
        keyword_options.process(self,kwargs)

        self._set_geometry(name, position)
        self.data = Data(self)
        self.df = self.data.photon_data



    def _set_geometry(self, name, position):
        self.name=name
        if position is None:
            skycoord = SkyCoord.from_name(name)
            gal = skycoord.galactic
            self.l,self.b = (gal.l.value, gal.b.value)
        else:
            self.l,self.b = position
        if self.verbose>0:
            print(f'Selected position: (l,b)=({self.l:.3f},{self.b:.3f}), radius={self.radius}')
    

    def plot_normalized_rate(self, step=None, min_exposure_factor=0.3, data_cut=None):

        # get binned data, cut out low exposure bins
        dfm = self.data.binner(step=step,cut=data_cut)
        exp =    dfm.exp.values
        exp_mean = exp.mean()
        mask = exp > min_exposure_factor*exp_mean
        interval = step or self.interval
        if self.verbose>0:
            print(f'exposure minimum factor, {min_exposure_factor}, removes {sum(~mask)}/{len(mask)} intervals')
            
        dfm = dfm.loc[mask,:]
        rel_exp = dfm.exp.values/exp_mean
        t=       dfm.time.values
        counts = dfm.counts.values
        ratio = counts/rel_exp
        y = ratio/ratio.mean()


        fig, (ax1,ax2)= plt.subplots(2,1, figsize=(15,5), sharex=True,gridspec_kw=dict(hspace=0) )
        ax1.plot(t, rel_exp, '+'); ax1.grid(alpha=0.5)
        ax1.set(ylabel=f'Exposure per {interval} day', ylim=(0,None))
        ax1.text(0.01, 0.05, f'mean exposure: {exp_mean/interval:.2e} / day',
                     transform=ax1.transAxes)
        dy = y/np.sqrt(counts) 
        ax2.errorbar(t, y, yerr=dy,  fmt='+');
        ax2.set(xlabel=r'$\mathrm{MJD}$', ylabel='Relative flux')
        ax2.axhline(1, color='grey')
        ax2.grid(alpha=0.5)
        fig.suptitle(f'Normalized flux for {self.name}')
    

    def plot_time(self, delta_max=2, delta_t=2, xlim=None):
        """
        """
        df = self.df

        t = df.time.values
        ta,tb=t[0],t[-1]
        Nbins = int((tb-ta)/float(delta_t))

        fig,ax= plt.subplots(figsize=(15,5))
        hkw = dict(bins = np.linspace(ta,tb,Nbins), histtype='step')
        ax.hist(t, label='E>100 MeV', **hkw)
        ax.hist(t[(df.delta<delta_max) & (df.band>0)], label='delta<{} deg'.format(delta_max), **hkw)
        ax.set(xlabel=r'$\mathrm{MJD}$', ylabel='counts per {:.0f} day'.format(delta_t))
        if xlim is not None: ax.set(xlim=xlim)
        ax.legend()
        ax.set_title('{} counts vs. time'.format(self.name))

    def plot_delta(self, cumulative=False, squared=True):
        plt.rc('font', size=12)
        df = self.df
        fig,ax = plt.subplots(figsize=(6,3))
        x = df.delta**2 if squared else df.delta
        hkw = dict(bins=np.linspace(0, 25 if squared else 5, 100), 
                   histtype='step',lw=2,cumulative=cumulative)
        ax.hist(x, label='E>100 MeV', **hkw)
        ax.hist(x[df.band>8], label='E>1 GeV', **hkw)
        ax.set(yscale='log', xlabel='delta**2 [deg^2]' if squared else 'delta [deg]', 
            ylabel='cumulative counts' if cumulative else 'counts'); 
        ax.legend(loc='upper left' if cumulative else 'upper right');


### Code that must be run in FermiTools context to create the database
#from uw/data import binned_data
# def create_timed_data(
#         monthly_ft1_files='/afs/slac/g/glast/groups/catalog/P8_P305/zmax105/*.fits',
#         outfolder='$FERMI/data/P8_P305/time_info/',
#         overwrite=False  ):
#     """
#     """
#     files=sorted(glob.glob(monthly_ft1_files))
#     assert len(files)>0, 'No ft1 files found at {}'.format(monthly_ft1_files)
#     gbtotal = np.array([os.stat(filename).st_size for filename in files]).sum()/2**30
#     print '{} FT1 files found, {} GB total'.format(len(files), gbtotal)
#     outfolder = os.path.expandvars(outfolder)
#     if not os.path.exists(outfolder):
#         os.makedirs(outfolder)
#     os.chdir(outfolder)   
#     for filename in files:
#         m = filename.split('_')[-2]
#         outfile = 'month_{}.pkl'.format(m)
#         if not overwrite and os.path.exists(outfile) :
#             print 'exists: {}'.format(outfile)
#             continue
#         print 'writing {}'.format(outfile),
#         tr = binned_data.ConvertFT1(filename).time_record()
#         pickle.dump(tr, open(outfile, 'w'))
