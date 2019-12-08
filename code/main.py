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

from data_management import TimedData
from weightman import WeightedData, WeightModel
from light_curve import LightCurve, BayesianBlocks



class Main(object):
    """Top-level processing for photon data
    """
 
    plt.rc('font', size=12)
    defaults=(
        ('verbose', 1, 'verbosity level'),
        ('radius',  5, 'cone radius for selection [deg]'),
        ('interval', 2, 'Binning time interval [days]'),
        ('mjd_range', None, 'Range of MJD: default all data'),
        ('weight_file', None, 'file name to find weight'),
        ('fix_weights',  True, 'Set to supplement weights with model' ),
       )
    
    @keyword_options.decorate(defaults)
    def __init__(self, name, position=None, **kwargs  ):
        """Set up combined data from set of monthly files

        name :    string, source name 
        position : an (l,b) pair [optional- if not present, use name to look up] 

        """
        keyword_options.process(self,kwargs)

        self._set_geometry(name, position)
        self.data = TimedData(self, source_name=name, verbose=self.verbose)

        if self.weight_file is not None:
            # adds weights from map and replace data object
            self.data = WeightedData(self.data, self.weight_file, self.fix_weights)
            if self.fix_weights:
                if self.verbose>0:
                    print(f'Creating weight model')
                df = self.data.photon_data
                weight_model = WeightModel.from_data(df, plotit=False)
                tofix = pd.isna(df.weight) | (df.band>13) 
                fixme= df.loc[tofix,:]
                fixed = fixme.apply(lambda c: weight_model(c.band, c.radius), axis=1,)  
                df.loc[tofix, 'weight'] = fixed
                self.wm = weight_model

        self.df = self.data.photon_data

    @property
    def dataframe(self):
        return self.df
    
    def _set_geometry(self, name, position):
        self.name=name
        if position is None:
            skycoord = SkyCoord.from_name(name)
            gal = skycoord.galactic
            self.l,self.b = (gal.l.value, gal.b.value)
        else:
            self.l,self.b = position
        if self.verbose>0:
            print(f'Source {self.name} at: (l,b)=({self.l:.3f},{self.b:.3f}); ROI radius={self.radius}')
            
    def light_curve(self, bins=None, **kwargs):
        """ Rerurn a LIghtCurve object, containing a table of fluxes and other cell info
        bins: a list of bin edges | None
            if None use default
        """
        if bins is None and len(kwargs)==0: # set or use cached version unless new bins or a kwarg
            if not hasattr(self, 'basic_lc'):
                self.basic_lc = LightCurve(self.data.binned_weights(bins), **kwargs)
            return self.basic_lc
        return LightCurve(self.data.binned_weights(bins), **kwargs)

    def bayesian_blocks(self, lc=None, lc_kwargs={}, **kwargs):
        """
        lc : LightCurve object or None
            initial, presumably regular, binning to be used. If None, use default 
            bb_kwargs : dict
                parameters to pass to the BayesianBlocks class, especially fitness_func
            kwargs:
                parameter to psss for the light curve
        """
        bb = BayesianBlocks(lc or self.light_curve(), **kwargs)
        edges = bb.partition()
        lckw = dict(rep='poisson', min_exp=0.01)
        lckw.update(lc_kwargs)
        return bb.light_curve(edges, **lckw)
    

    def plot_time(self, radius_max=2, delta_t=2, xlim=None):
        """
        """
        df = self.df

        t = df.time.values
        ta,tb=t[0],t[-1]
        Nbins = int((tb-ta)/float(delta_t))

        fig,ax= plt.subplots(figsize=(15,5))
        hkw = dict(bins = np.linspace(ta,tb,Nbins), histtype='step')
        ax.hist(t, label='E>100 MeV', **hkw)
        ax.hist(t[(df.radius<radius_max) & (df.band>0)], label=f'radius<{radius_max} deg', **hkw)
        ax.set(xlabel=r'$\mathrm{MJD}$', ylabel='counts per {:.0f} day'.format(delta_t))
        if xlim is not None: ax.set(xlim=xlim)
        ax.legend()
        ax.set_title('{} counts vs. time'.format(self.name))

    def plot_radius(self, cumulative=False, squared=True):
        plt.rc('font', size=12)
        df = self.df
        fig,ax = plt.subplots(figsize=(6,3))
        x = df.radius**2 if squared else df.radius
        hkw = dict(bins=np.linspace(0, 25 if squared else 5, 100), 
                   histtype='step',lw=2,cumulative=cumulative)
        ax.hist(x, label='E>100 MeV', **hkw)
        ax.hist(x[df.band>8], label='E>1 GeV', **hkw)
        ax.set(yscale='log', xlabel='radius**2 [deg^2]' if squared else 'delta [deg]', 
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
