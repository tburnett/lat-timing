"""
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys
import core

data=None # make the Data object global here

class CoreInterface(object):
    """Simple interface to the Kerr code in godot.core
    """
    
    def __init__(self, binned_weights, min_exp=0.2):
        """
        parameters:
            binned_weights : a main.BinnedWeights object
            min_exp : float
                minimum fractional exposure accepted
        """
        global data
        data=binned_weights.data
   
        # Required parameters for the core.Cell constructor
        cell_pars=dict(tstart=0,tstop=0,exposure=1,photon_times=[],photon_weights=[],
                    source_to_background_ratio=0.5)

        cells=[]; rejected=0
        
        for c in binned_weights: # loop over interval information dicts
            t, tw, fexp, w, S, B = [c[x] for x in 't tw fexp w S B'.split()]
            if fexp<min_exp:
                rejected+=1
                continue
            cell_pars.update(tstart=t-0.5*tw, tstop=t+0.5*tw, 
                             photon_weights=w, 
                             exposure=S, 
                             source_to_background_ratio=S/B)
            cells.append(core.Cell(**cell_pars))

        if data.verbose>0:
            print(f'Loaded {len(cells)} cells for further analysis')
            if rejected>0:
                print(f'\trejected {rejected} with fractional exposure < {min_exp:.2f}')
        if len(cells)==0:
            raise Exception(f'Failed to find any cell with frational exposure> {min_exp}')
        
        self.cells = cells
        #self.cll = core.CellsLogLikelihood(cells) 

    def __repr__(self):
        return f'{self.__class__}\t {len(self.cells)} cells'
    
    def lightcurve(self, dataframe=True, *pars):
        """ Return a Bayesian blocks flux density light curve
        pars:
            tsmin=4,plot_years=False,plot_phase=False,
            get_ts=False
        """
        rvals =  self.cll.get_lightcurve(*pars)
        if not dataframe: return rvals
        return pd.DataFrame([rvals[:,0],rvals[:,2], 0.5*(rvals[:,3]+rvals[:,4]), ],
                            index='time rate error'.split()).T
    
    def bb_lightcurve(self, *pars):
        """Return a flux density light curve for the raw cells.
        pars:tsmin=4,plot_years=False,plot_phase=False,
            bb_prior=8

        """
        return self.cll.get_bb_lightcurve(*pars)
    
    def plot_bb(self, *pars):
        """tsmin=4,fignum=2,clear=True,color='C3',
            plot_raw_cells=True,bb_prior=4,plot_years=False,
            no_bb=False,log_scale=False,
            plot_phase=False,ax=None
        """
        return self.cll.plot_cells_bb(*pars)
    
    def plot_lc(self, *pars):
        return self.cll.plot_clls_lc(*pars)

    # @property
    # def cells(self):
    #     return self.cll.cells