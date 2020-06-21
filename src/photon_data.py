"""
package dev initialization
"""
import os, sys
import numpy as np
import matplotlib.pylab as plt
import pandas as pd

from jupydoc import DocPublisher

__docs__ = ['GammaDataPub']

from utilities import phase_plot, poiss_pars_hist, GammaData
#from utilities import PoissonTable
   
class GammaDataPub(DocPublisher): 
    """
    title: Photon data setup
   
    sections:
        title_page
        data_setup [data_load data_save]


    source_name: Geminga
    data_path: $HOME/work/data/photons

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def data_setup(self):
        """Data Setup
        Specified source name: {self.source_name}
        """
        self.publishme()

    def data_load(self):
        """Load data
        
        * {{self.gdata}}
        
        {self.gdata}
        
        * **photon data**: 
        {photons_head}

        * **light curve**
        {self.light_curve}

        """
        #-------------------------------------------------------------------
        self.gdata = GammaData(name=self.source_name)
        photons = self.gdata.photons
        photons_head = photons.head()
        self.light_curve = self.gdata.light_curve()
        #-------------------------------------------------------------------
        self.publishme()

    def data_save(self):
        """Save the Data

        This makes a pickle of the photon data and light curve to `{outfile}`

        Format is
        ```
        lc = self.light_curve
        outdict = dict(
            source_name=self.source_name, 
            photons=self.gdata.photons.to_dict('records'),
            light_curve =
                dict(
                    rep=lc.rep,
                    edges = lc.data.edges,
                    fit_dict = lc.fit_df.to_dict('records'),
                    ),
                )   
        ```

        Read back to check: keys are
        {pkl_keys}
        """
        #-------------------------------------------------------------------
        import pickle
        outfile = f'{os.path.expandvars(self.data_path)}/{self.source_name}.pkl'
        lc = self.light_curve
        outdict = dict(
                source_name=self.source_name, 
                photons=self.gdata.photons.to_dict('records'),
                light_curve =
                    dict(
                        rep=lc.rep,
                        edges = lc.data.edges,
                        fit_dict = lc.fit_df.to_dict('records'),
                        ),
                    )        
        with open(outfile, 'wb') as out:
            pickle.dump( outdict,  out)
        # check
        with open(outfile, 'rb') as inp:
            pkl = pickle.load(inp)
        pkl_keys = list(pkl.keys())
        #-------------------------------------------------------------------           
        self.publishme()
        
        