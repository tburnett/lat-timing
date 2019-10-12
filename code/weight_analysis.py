"""process weights for timed data
"""

import os, sys
from data_management import Data
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import corner
import keyword_options

class Main(object):

    defaults=(
        ('verbose', 1,),
        ('source_name', 'Geminga','name of the source'),
        ('mjd_range', (51910, 54800),'MJD limits to use'),
        ('weight_file', '../data/geminga_weights.pkl', 'location of pointlike weight file'),
    )
    
    @keyword_options.decorate(defaults)
    def __init__(self, **kwargs):
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

    def corner_plot(self):
        """produce a "corner" plot.
        """
        fig, axx=plt.subplots(3,3, figsize=(10,10))
        corner.corner(self.photons['eband radius weight'.split()], 
                    range=[(-0.75,7.25),(0,5),(0,1)],
                    bins=(16,20,20),
                    label_kwargs=dict(fontsize=14), 
                    labels=['Energy band index', 'Radius [deg]', 'weight'],
                    show_titles=False, top_ticks=True, color='blue', fig=fig);
        fig.set_edgecolor('white')

        fig.suptitle(f'Weight analysis, {len(self.photons)} events\nModel info'+self.model_info,
                    fontsize=14, ha='left',
                    #fontproperties=dict(family='monospace'), #didn't work
                    )
