"""
Process time data set
Expect to find set of files created by uw/data/timed_data/create_timed_data to generate files with times for all 
Extract a single data set around a cone with TimedData
"""

import os, glob, pickle
import healpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord
import keyword_options
import exposure, data_management


class TimedData(object):
    """Create a data set at a given position
    """
 
    plt.rc('font', size=12)
    defaults=(
        ('verbose',2,'verbosity level'),
        ('radius', 5, 'cone radius for selection [deg]'),
        ('interval', 2, 'Binning time interval [days]'),
        ('mjd_range', None, 'Range of MJD'),
       )
    
    @keyword_options.decorate(defaults)
    def __init__(self, name, position=None, **kwargs  ):
        """Set up combined data from set of monthly files

        name :    string, source name 
        position : an (l,b) pair [optional- if not present, use name to look up] 

        """
        keyword_options.process(self,kwargs)
        self.name=name
        if position is None:
            skycoord = SkyCoord.from_name(name)
            gal = skycoord.galactic
            self.l,self.b = (gal.l.value, gal.b.value)
        else:
            self.l,self.b = position
        if self.verbose>1:
            print(f'Selected position: (l,b)=({self.l:.3f},{self.b:.3f}), radius={self.radius}')

        self.data = data_management.Data(self, mjd_range=self.mjd_range)
        self.df = self.data.photon_df
        self.binned_exposure = self._get_exposure()

    def plot_time(self, delta_max=2, delta_t=2, xlim=None):
        """
        """
        df = self.df

        t = data_management.MJD(df.time)
        ta,tb=t[0],t[-1]
        Nbins = int((tb-ta)/float(delta_t))

        fig,ax= plt.subplots(figsize=(15,5))
        hkw = dict(bins = np.linspace(ta,tb,Nbins), histtype='step')
        ax.hist(t, label='E>100 MeV', **hkw)
        ax.hist(t[(df.delta<delta_max) & (df.band>0)], label='delta<{} deg'.format(delta_max), **hkw);
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

    def _get_exposure(self ):
        # get the livetime history wnd associated GTI

        lt = exposure.Livetime(self.data.ft2_files, self.data.gti_files, verbose=self.verbose)
        
        # use it and the current direction to get the expoosure
        edf = lt.get_exposure(SkyCoord(self.l,self.b,unit='deg',frame='galactic' )) # the exposure data frame
        self.tstart=data_management.MJD(edf.tstart.values) #todo convert previously
        self.tstop =data_management.MJD(edf.tstop.values)
        exp    = edf.exposure.values
        if self.mjd_range is not None:

            m = np.searchsorted(self.tstop, self.mjd_range[1])+1
            self.tstart = self.tstart[:m]
            self.tstop  = self.tstop[:m]
            exp = exp[:m]
        
        #use cumulative exposure over 30-sec bins to integrate over larger periods
        cumexp = np.concatenate(([0],np.cumsum(exp * (self.tstop-self.tstart))))

        time_range = self.tstop[-1]-self.tstart[0]
        nbins = int(time_range/self.interval)+1; print(f'Selecting {nbins} intervals of {self.interval} days')
        self.time_bins = self.tstart[0] + np.arange(nbins+1)*self.interval 

        # get index into tstop array of the bin edges
        edge_index = np.searchsorted(self.tstop, self.time_bins)
        # now the exposure integrated over the intervals
        return  np.diff(cumexp[edge_index])
        
    def _data_binner(self, cut=None):
        """ use time intervals defined by exposure calculation to generate equivalent bins
        TODO: allow cuts here
        """
        # find range comvered by current GTI and make subset DataFrame
        etime = data_management.MJD(self.df.time) # TODO convert before
        imin,imax = np.searchsorted(etime, [self.tstart[0],self.tstop[-1]]);
        setime = etime[imin:imax]
        print(f'Select {len(setime):,} photons within live time range')

        # get associated live time index for each photon
        lt_index = np.searchsorted(self.tstop, setime) # before which stop
        # make sure past start
        tdiff = setime - self.tstart[lt_index]
        in_gti = tdiff>=0
        print(f'... and exclude {sum(~in_gti):,} photons not in GTI')
        etime_ok = setime[in_gti] 

        # bin the data (all photons so far)
        return np.histogram(etime_ok, self.time_bins)[0]

    def plot_normalized_rate(self, cut=None):

        binned_data = self._data_binner(cut)

        bin_center = (self.time_bins[1:]+self.time_bins[:-1])/2
        mask = binned_data>10 
        ratio = binned_data[mask]/self.binned_exposure[mask]
        y = ratio/ratio.mean()
        fig, (ax1,ax2)= plt.subplots(2,1, figsize=(15,5), sharex=True,gridspec_kw=dict(hspace=0) )
        t = bin_center
        ax1.plot(t, self.binned_exposure, '+'); ax1.grid(alpha=0.5)
        ax1.set(ylabel=f'Exposure per {self.interval} day')
        dy = y/np.sqrt(binned_data[mask]) 
        ax2.errorbar(x =  t[mask],   y=y, yerr=dy,  fmt='+');
        ax2.set(xlabel='MJD', ylabel='Relative flux')
        ax2.axhline(1, color='grey');ax2.grid(alpha=0.5)
        fig.suptitle(f'Flux check for {self.name}')




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